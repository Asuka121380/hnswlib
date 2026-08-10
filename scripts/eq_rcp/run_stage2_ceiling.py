#!/usr/bin/env python3
"""Run the Stage 2 flat OAE representation ceiling experiment."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn

from operational_dataset import (
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
)
from stage2_quantization import (
    STAGE2_FORMAT,
    STAGE2_VERSION,
    Stage1OperationalData,
    code_oracle_scores,
    conservative_cutoff,
    decision_metrics,
    edge_training_set,
    fit_quantizer,
    fit_code_oracle,
    paired_bootstrap_ci,
    per_query_rows,
    projection_metrics,
    reconstruct_events,
    results_semantic_sha256,
    save_quantizer,
    smooth_threshold_weights,
)


def _empty_output(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"Stage 2 output directory must be empty: {path}")


def _semantic_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in {"stage1_dataset_dir", "notes"}
    }


def _method_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for method in sorted({str(row["method"]) for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        numeric = [
            "projection_mse",
            "projection_abs_q99",
            "projection_abs_q999",
            "reconstruction_mse_unique_edge",
            "calibrated_coverage",
            "calibrated_false_prune_rate_over_good",
            "code_oracle_coverage",
            "code_oracle_false_prune_rate_over_good",
        ]
        output[method] = {
            f"mean_{name}": float(np.mean([float(row[name]) for row in selected]))
            for name in numeric
        }
        output[method]["seed_count"] = len(selected)
    return output


def _semantic_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key != "training_seconds"}
        for row in rows
    ]


def _semantic_artifacts(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in artifact.items()
            if key not in {"relative_path", "artifact_sha256"}
        }
        for artifact in artifacts
    ]


def _per_query_metric(
    rows: list[dict[str, Any]], method: str, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["method"] == method]
    query_ids = sorted({int(row["query_id"]) for row in selected})
    values = []
    for query_id in query_ids:
        values.append(
            float(
                np.mean(
                    [
                        float(row[metric])
                        for row in selected
                        if int(row["query_id"]) == query_id
                    ]
                )
            )
        )
    return np.asarray(query_ids, dtype=np.int64), np.asarray(values, dtype=np.float64)


def _gate_report(
    config: dict[str, Any],
    data: Stage1OperationalData,
    rows: list[dict[str, Any]],
    per_query: list[dict[str, Any]],
    fully_sampled: bool,
) -> dict[str, Any]:
    summary = _method_summary(rows)
    baselines = [name for name in ["direct_delta_pq", "direct_delta_opq"] if name in summary]
    structured = [name for name in ["oae_diagonal", "oae_block"] if name in summary]
    if not baselines or not structured:
        raise ValueError("Stage 2 Gate requires PQ/OPQ and structured OAE methods")
    baseline = min(baselines, key=lambda name: summary[name]["mean_projection_abs_q99"])
    challenger = min(structured, key=lambda name: summary[name]["mean_projection_abs_q99"])
    baseline_q99 = summary[baseline]["mean_projection_abs_q99"]
    challenger_q99 = summary[challenger]["mean_projection_abs_q99"]
    q99_reduction = (baseline_q99 - challenger_q99) / max(baseline_q99, 1e-30)
    oracle_gain = (
        summary[challenger]["mean_code_oracle_coverage"]
        - summary[baseline]["mean_code_oracle_coverage"]
    )
    query_a, baseline_values = _per_query_metric(per_query, baseline, "projection_abs_q99")
    query_b, challenger_values = _per_query_metric(per_query, challenger, "projection_abs_q99")
    if not np.array_equal(query_a, query_b):
        raise ValueError("paired-query Gate inputs are not aligned")
    bootstrap = paired_bootstrap_ci(
        baseline_values,
        challenger_values,
        int(config["bootstrap"]["seed"]),
        int(config["bootstrap"]["samples"]),
        float(config["bootstrap"]["confidence"]),
    )
    by_seed: dict[str, float] = {}
    for seed in config["seeds"]:
        base_row = next(row for row in rows if row["method"] == baseline and row["seed"] == seed)
        challenge_row = next(row for row in rows if row["method"] == challenger and row["seed"] == seed)
        by_seed[str(seed)] = (
            float(base_row["projection_abs_q99"])
            - float(challenge_row["projection_abs_q99"])
        ) / max(float(base_row["projection_abs_q99"]), 1e-30)
    thresholds = config["gate"]
    criteria = {
        "q99_reduction": {
            "value": q99_reduction,
            "required": float(thresholds["minimum_q99_reduction"]),
            "pass": q99_reduction >= float(thresholds["minimum_q99_reduction"]),
        },
        "paired_query_ci_lower": {
            "value": bootstrap["ci_lower"],
            "required": float(thresholds["minimum_paired_ci_lower"]),
            "pass": bootstrap["ci_lower"] > float(thresholds["minimum_paired_ci_lower"]),
        },
        "code_oracle_coverage_gain": {
            "value": oracle_gain,
            "required": float(thresholds["minimum_code_oracle_coverage_gain"]),
            "pass": oracle_gain >= float(thresholds["minimum_code_oracle_coverage_gain"]),
        },
        "ideal_coverage": {
            "value": summary[challenger]["mean_code_oracle_coverage"],
            "required": float(thresholds["minimum_ideal_coverage"]),
            "pass": summary[challenger]["mean_code_oracle_coverage"]
            >= float(thresholds["minimum_ideal_coverage"]),
        },
        "seed_stability": {
            "values": by_seed,
            "required": "positive Q99 reduction for every seed",
            "pass": all(value > 0.0 for value in by_seed.values()),
        },
    }
    diagnostic_pass = all(bool(value["pass"]) for value in criteria.values())
    formal_eligible = (
        bool(config.get("allow_formal_gate_decision", False))
        and config.get("run_role") == "formal_ceiling"
        and data.manifest["dataset_role"] == "formal"
        and data.manifest["adjacency_validation"]["status"] == "PASS"
        and fully_sampled
    )
    decision = (
        ("GO_CPP" if diagnostic_pass else "NO_GO_CPP")
        if formal_eligible
        else "PENDING_FORMAL_STAGE2"
    )
    full_gap = None
    if "oae_full" in summary:
        full_value = summary["oae_full"]["mean_projection_abs_q99"]
        full_gap = (challenger_q99 - full_value) / max(full_value, 1e-30)
    return {
        "decision": decision,
        "formal_eligible": formal_eligible,
        "diagnostic_gate_pass": diagnostic_pass,
        "best_mse_baseline": baseline,
        "best_structured_oae": challenger,
        "criteria": criteria,
        "paired_query_bootstrap": bootstrap,
        "full_vs_structured_q99_gap": full_gap,
        "method_summary": summary,
        "warning": (
            None
            if formal_eligible
            else "Development/subsampled results cannot authorize C++ sidecar or LUT work."
        ),
    }


def run_experiment(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("format") != STAGE2_FORMAT or config.get("version") != STAGE2_VERSION:
        raise ValueError("unsupported Stage 2 config format/version")
    _empty_output(output_dir)
    stage1_dir = Path(config["stage1_dataset_dir"])
    data = Stage1OperationalData(stage1_dir)
    try:
        expected_dataset_hash = config.get("expected_stage1_dataset_semantic_sha256")
        actual_dataset_hash = data.manifest["dataset_semantic_sha256"]
        if expected_dataset_hash and expected_dataset_hash != actual_dataset_hash:
            raise ValueError("Stage 1 dataset semantic SHA-256 mismatch")
        selection = config["selection"]
        split_queries: dict[str, np.ndarray] = {}
        split_events: dict[str, np.ndarray] = {}
        for role in ["quantizer_train", "decoder_train", "evaluation"]:
            split = selection[role]["split"]
            maximum = selection[role].get("max_queries")
            query_ids = data.split_query_ids(
                split, None if maximum is None else int(maximum), int(selection["seed"])
            )
            split_queries[role] = query_ids
            split_events[role] = data.event_indices(query_ids)
        roles = list(split_queries)
        for left in range(len(roles)):
            for right in range(left + 1, len(roles)):
                if np.intersect1d(split_queries[roles[left]], split_queries[roles[right]]).size:
                    raise ValueError("Stage 2 query selections overlap")
        fully_sampled = all(selection[role].get("max_queries") is None for role in roles)

        train_values = data.event_values(split_events["quantizer_train"])
        train_x = data.event_x(split_events["quantizer_train"])
        weight_spec = config["objective_weight"]
        if weight_spec["mode"] == "raw":
            event_weights = train_values["raw_event_weight"].astype(np.float64)
        elif weight_spec["mode"] == "per_query":
            event_weights = train_values["per_query_weight"].astype(np.float64)
        elif weight_spec["mode"] == "smooth_threshold":
            event_weights = smooth_threshold_weights(
                train_values["margin"],
                float(weight_spec["minimum"]),
                float(weight_spec["tau"]),
            )
        else:
            raise ValueError(f"unsupported objective weight mode: {weight_spec['mode']}")
        train_edge_ids, _, edge_weights = edge_training_set(
            train_values["edge_id"].astype(np.int64), event_weights
        )
        train_vectors = data.edge_vectors(train_edge_ids)
        near_margin_limit = float(
            np.quantile(
                np.abs(train_values["margin"].astype(np.float64)),
                float(config["evaluation"]["near_margin_quantile"]),
            )
        )

        model_dir = output_dir / "models"
        model_dir.mkdir()
        rows: list[dict[str, Any]] = []
        per_query: list[dict[str, Any]] = []
        artifacts: list[dict[str, Any]] = []
        for seed_value in config["seeds"]:
            seed = int(seed_value)
            for method in config["methods"]:
                started = time.perf_counter()
                model = fit_quantizer(
                    str(method),
                    train_vectors,
                    edge_weights,
                    train_x,
                    event_weights,
                    config["quantizer"],
                    seed,
                )
                training_seconds = time.perf_counter() - started
                artifact_path = model_dir / f"{method}-seed{seed}.npz"
                artifact = save_quantizer(
                    model,
                    artifact_path,
                    {
                        "seed": seed,
                        "stage1_dataset_semantic_sha256": actual_dataset_hash,
                        "training_edge_count": int(len(train_edge_ids)),
                        "training_event_count": int(len(split_events["quantizer_train"])),
                    },
                )
                artifact["relative_path"] = str(artifact_path.relative_to(output_dir))
                artifacts.append(artifact)

                calibration = reconstruct_events(
                    model, data, split_events["decoder_train"]
                )
                alpha = float(config["evaluation"]["diagnostic_false_prune_alpha"])
                cutoff = conservative_cutoff(
                    calibration["estimated_margin"], calibration["good_candidate"], alpha
                )
                evaluation = reconstruct_events(model, data, split_events["evaluation"])
                oracle_cutoff = conservative_cutoff(
                    evaluation["estimated_margin"], evaluation["good_candidate"], alpha
                )
                score_oracle = decision_metrics(evaluation, oracle_cutoff)
                flexible_oracle = fit_code_oracle(
                    calibration, seed, config["flexible_code_oracle"]
                )
                flexible_scores = code_oracle_scores(flexible_oracle, evaluation)
                flexible_cutoff = conservative_cutoff(
                    flexible_scores, evaluation["good_candidate"], alpha
                )
                flexible_payload = dict(evaluation)
                flexible_payload["estimated_margin"] = flexible_scores
                projection = projection_metrics(evaluation, near_margin_limit)
                calibrated = decision_metrics(evaluation, cutoff)
                code_oracle = decision_metrics(flexible_payload, flexible_cutoff)
                row = {
                    "method": str(method),
                    "seed": seed,
                    "code_bytes": int(model.artifact_metadata()["code_bytes"]),
                    "training_seconds": float(training_seconds),
                    "training_edge_count": int(len(train_edge_ids)),
                    "training_event_count": int(len(split_events["quantizer_train"])),
                    "evaluation_query_count": int(len(split_queries["evaluation"])),
                    "evaluation_event_count": int(len(split_events["evaluation"])),
                    "near_margin_limit": near_margin_limit,
                    **projection,
                    "calibrated_cutoff": cutoff,
                    "calibrated_coverage": calibrated["coverage"],
                    "calibrated_oracle_opportunity_recall": calibrated[
                        "oracle_opportunity_recall"
                    ],
                    "calibrated_false_prune_rate_over_good": calibrated[
                        "false_prune_rate_over_good"
                    ],
                    "score_oracle_cutoff": oracle_cutoff,
                    "score_oracle_coverage": score_oracle["coverage"],
                    "score_oracle_false_prune_rate_over_good": score_oracle[
                        "false_prune_rate_over_good"
                    ],
                    "code_oracle_cutoff": flexible_cutoff,
                    "code_oracle_coverage": code_oracle["coverage"],
                    "code_oracle_oracle_opportunity_recall": code_oracle[
                        "oracle_opportunity_recall"
                    ],
                    "code_oracle_false_prune_rate_over_good": code_oracle[
                        "false_prune_rate_over_good"
                    ],
                }
                if row["code_bytes"] != int(config["quantizer"]["code_bytes"]):
                    raise ValueError(f"method {method} violated fixed code budget")
                rows.append(row)
                per_query.extend(per_query_rows(str(method), seed, evaluation))
                print(
                    f"completed {method} seed={seed}: "
                    f"Q99={row['projection_abs_q99']:.8g} "
                    f"oracle_coverage={row['code_oracle_coverage']:.4f}"
                )

        comparison_path = output_dir / "comparison.parquet"
        per_query_path = output_dir / "per_query_metrics.parquet"
        pq.write_table(pa.Table.from_pylist(rows), comparison_path, compression="zstd")
        pq.write_table(pa.Table.from_pylist(per_query), per_query_path, compression="zstd")
        gate = _gate_report(config, data, rows, per_query, fully_sampled)
        write_json(output_dir / "gate_report.json", gate)
        semantic_payload = {
            "format": STAGE2_FORMAT,
            "version": STAGE2_VERSION,
            "semantic_config": _semantic_config(config),
            "stage1_dataset_semantic_sha256": actual_dataset_hash,
            "query_selection": {
                role: split_queries[role].tolist() for role in sorted(split_queries)
            },
            "comparison": _semantic_comparison_rows(rows),
            "per_query": per_query,
            "artifacts": _semantic_artifacts(artifacts),
            "gate": gate,
        }
        manifest = {
            "format": STAGE2_FORMAT,
            "version": STAGE2_VERSION,
            "run_role": config["run_role"],
            "stage1_dataset_dir": str(stage1_dir.resolve()),
            "stage1_dataset_semantic_sha256": actual_dataset_hash,
            "stage2_results_semantic_sha256": results_semantic_sha256(semantic_payload),
            "semantic_config_sha256": sha256_bytes(
                canonical_json_bytes(_semantic_config(config))
            ),
            "selection": {
                role: {
                    "split": selection[role]["split"],
                    "query_count": int(len(split_queries[role])),
                    "event_count": int(len(split_events[role])),
                    "query_ids_sha256": sha256_bytes(
                        np.asarray(split_queries[role], dtype="<i8").tobytes()
                    ),
                }
                for role in roles
            },
            "fully_sampled": fully_sampled,
            "near_margin_limit": near_margin_limit,
            "artifacts": artifacts,
            "outputs": {
                "comparison": {
                    "relative_path": comparison_path.name,
                    "sha256": sha256_file(comparison_path),
                },
                "per_query_metrics": {
                    "relative_path": per_query_path.name,
                    "sha256": sha256_file(per_query_path),
                },
                "gate_report": {
                    "relative_path": "gate_report.json",
                    "sha256": sha256_file(output_dir / "gate_report.json"),
                },
            },
            "gate_decision": gate["decision"],
            "diagnostic_gate_pass": gate["diagnostic_gate_pass"],
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
            },
        }
        write_json(output_dir / "stage2_manifest.json", manifest)
        return manifest
    finally:
        data.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = run_experiment(args.config, args.output_dir)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "PASS",
                "gate_decision": manifest["gate_decision"],
                "diagnostic_gate_pass": manifest["diagnostic_gate_pass"],
                "stage2_results_semantic_sha256": manifest[
                    "stage2_results_semantic_sha256"
                ],
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
