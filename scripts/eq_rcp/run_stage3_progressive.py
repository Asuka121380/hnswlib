#!/usr/bin/env python3
"""Run frozen-representation Stage 3 progressive-code selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from operational_dataset import canonical_json_bytes, read_json, sha256_bytes, write_json
from stage2_quantization import Stage1OperationalData, edge_training_set
from stage3_quantization import (
    STAGE3_FORMAT,
    STAGE3_VERSION,
    ProgressiveGainShapeQuantizer,
    evaluate_stopping,
    fit_stopping_radius,
    projection_metrics,
    projection_payload,
    save_progressive_model,
)


def _prepare_output(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"Stage 3 output directory must be empty: {path}")
    (path / "models").mkdir()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_ready_input(manifest: dict[str, Any]) -> None:
    if manifest.get("format") != "eq_rcp_stage3_input_manifest" or manifest.get("version") != 1:
        raise ValueError("invalid Stage 3 input manifest")
    if manifest.get("status") != "READY_FOR_OFFLINE_STAGE3" or not manifest.get("experiment_ready"):
        raise ValueError("Stage 3 input is not ready; bind fresh evaluation queries first")
    if manifest.get("selected_flat_representation") != "gain_shape_opq":
        raise ValueError("Stage 3 runner only accepts the frozen gain_shape_opq representation")
    if manifest.get("risk_status") != "PENDING_RISK_ESTIMATOR":
        raise ValueError("unexpected Stage 3 risk status")
    if manifest.get("real_pruning_authorized") is not False:
        raise ValueError("Stage 3 input must not authorize real pruning")


def _event_payload(
    model: ProgressiveGainShapeQuantizer,
    data: Stage1OperationalData,
    event_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    values = data.event_values(event_indices)
    edge_ids, inverse = np.unique(values["edge_id"].astype(np.int64), return_inverse=True)
    payload = projection_payload(
        model,
        data.edge_vectors(edge_ids),
        inverse,
        data.event_x(event_indices),
        values["current_squared_distance"],
        values["edge_squared_length"],
        values["threshold"],
        values["good_candidate"],
        values["oracle_prunable"],
    )
    payload["query_id"] = values["query_id"].astype(np.int64)
    payload["edge_squared_length"] = values["edge_squared_length"].astype(np.float64)
    return payload


def _code_features(payload: dict[str, np.ndarray]) -> np.ndarray:
    inverse = payload["event_inverse"]
    coarse = payload["coarse_codes"][inverse].astype(np.float32) / 255.0
    if "residual_codes" in payload:
        residual = payload["residual_codes"][inverse].astype(np.float32) / 255.0
    else:
        residual = np.empty((len(inverse), 0), dtype=np.float32)
    scalar = np.column_stack(
        [
            payload["coarse_distance"],
            payload["full_distance"],
            payload["full_distance"] - payload["coarse_distance"],
            payload["threshold"],
            payload["full_distance"] - payload["threshold"],
            payload["edge_squared_length"],
        ]
    ).astype(np.float32)
    return np.column_stack([coarse, residual, scalar])


def _fit_code_information(
    payload: dict[str, np.ndarray], seed: int, spec: dict[str, Any]
) -> HistGradientBoostingClassifier:
    labels = payload["oracle_prunable"].astype(np.int32)
    if len(np.unique(labels)) != 2:
        raise ValueError("Stage 3 code-information model requires both classes")
    model = HistGradientBoostingClassifier(
        learning_rate=float(spec["learning_rate"]),
        max_iter=int(spec["max_iter"]),
        max_leaf_nodes=int(spec["max_leaf_nodes"]),
        min_samples_leaf=int(spec["min_samples_leaf"]),
        l2_regularization=float(spec["l2_regularization"]),
        early_stopping=False,
        random_state=seed,
    )
    model.fit(_code_features(payload), labels)
    return model


def _scores(model: HistGradientBoostingClassifier, payload: dict[str, np.ndarray]) -> np.ndarray:
    return model.predict_proba(_code_features(payload))[:, 1]


def _calibrate_score_cutoff(scores: np.ndarray, good: np.ndarray, alpha: float) -> float:
    unsafe = np.asarray(scores)[np.asarray(good, dtype=bool)]
    if unsafe.size == 0:
        raise ValueError("Stage 3 code-information calibration has no good candidates")
    return float(np.quantile(unsafe, 1.0 - alpha, method="higher"))


def _score_metrics(scores: np.ndarray, cutoff: float, payload: dict[str, np.ndarray]) -> dict[str, float]:
    decision = np.asarray(scores) > cutoff
    good = payload["good_candidate"]
    prunable = payload["oracle_prunable"]
    return {
        "code_information_coverage": float(np.mean(decision)),
        "code_information_false_prune_over_good": float(
            np.sum(decision & good) / max(np.sum(good), 1)
        ),
        "code_information_oracle_recall": float(
            np.sum(decision & prunable) / max(np.sum(prunable), 1)
        ),
    }


def _split_events(
    data: Stage1OperationalData, binding: dict[str, Any], seed: int
) -> np.ndarray:
    query_ids = data.split_query_ids(binding["split"], None, seed)
    if int(binding["query_count"]) != len(query_ids):
        raise ValueError(f"query count mismatch for fresh Stage 3 role {binding['split']}")
    query_hash = hashlib.sha256(np.asarray(query_ids, dtype="<i8").tobytes()).hexdigest()
    if query_hash != binding["query_ids_sha256"]:
        raise ValueError(f"query-ID hash mismatch for fresh Stage 3 role {binding['split']}")
    events = data.event_indices(query_ids)
    if int(binding["event_count"]) != len(events):
        raise ValueError(f"event count mismatch for fresh Stage 3 role {binding['split']}")
    return events


def _candidate_key(layout: str, strategy: str) -> str:
    return "flat_32__frozen_flat" if layout == "flat_32" else f"{layout}__{strategy}"


def _load_selected_artifact(input_manifest: dict[str, Any], seed: int) -> dict[str, Any]:
    matches = [
        item for item in input_manifest["frozen_artifacts"]["selected"]
        if int(item["seed"]) == int(seed)
    ]
    if len(matches) != 1:
        raise ValueError(f"missing frozen selected artifact for seed {seed}")
    return matches[0]


def run_experiment(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("format") != STAGE3_FORMAT or config.get("version") != STAGE3_VERSION:
        raise ValueError("invalid Stage 3 run config")
    input_path = Path(config["stage3_input_manifest"])
    input_manifest = read_json(input_path)
    _require_ready_input(input_manifest)
    if _sha256(Path(config["experiment_contract"])) != input_manifest["contract_sha256"]:
        raise ValueError("Stage 3 experiment contract hash mismatch")

    training = Stage1OperationalData(Path(config["training_stage1_dataset_dir"]))
    evaluation = Stage1OperationalData(Path(config["evaluation_stage1_dataset_dir"]))
    try:
        if training.manifest["dataset_semantic_sha256"] != input_manifest["source_stage2_1"][
            "stage1_dataset_semantic_sha256"
        ]:
            raise ValueError("Stage 3 training dataset is not the frozen Stage 2.1 dataset")
        if evaluation.manifest["dataset_semantic_sha256"] != input_manifest["evaluation_queries"][
            "dataset_semantic_sha256"
        ]:
            raise ValueError("fresh Stage 3 evaluation dataset hash mismatch")
        _prepare_output(output_dir)
        seed_for_splits = int(config["selection_seed"])
        train_queries = training.split_query_ids(
            config["quantizer_train"]["split"],
            config["quantizer_train"].get("max_queries"),
            seed_for_splits,
        )
        train_events = training.event_indices(train_queries)
        train_values = training.event_values(train_events)
        edge_ids, edge_inverse, edge_weights = edge_training_set(
            train_values["edge_id"].astype(np.int64),
            train_values["per_query_weight"].astype(np.float64),
        )
        train_edges = training.edge_vectors(edge_ids)
        train_x = training.event_x(train_events)
        role_bindings = input_manifest["evaluation_queries"]["roles"]
        selection_events = _split_events(evaluation, role_bindings["bit_allocation_selection"], seed_for_splits)
        calibration_events = _split_events(evaluation, role_bindings["stopping_calibration"], seed_for_splits)
        final_events = _split_events(evaluation, role_bindings["final_test"], seed_for_splits)

        stage2_result = Path(config["stage2_1_result_dir"])
        layouts = config["candidate_layouts"]
        strategies = config["allocation_strategies"]
        seeds = [int(seed) for seed in input_manifest["frozen_artifacts"]["matched_seeds"]]
        selection_rows: list[dict[str, Any]] = []
        artifact_rows: list[dict[str, Any]] = []
        model_index: dict[tuple[int, str], tuple[Path, dict[str, Any]]] = {}
        for seed in seeds:
            binding = _load_selected_artifact(input_manifest, seed)
            frozen_path = stage2_result / Path(
                str(binding["relative_path"]).replace("\\", "/")
            )
            if _sha256(frozen_path) != binding["artifact_sha256"]:
                raise ValueError(f"frozen Stage 2.1 artifact changed for seed {seed}")
            flat = ProgressiveGainShapeQuantizer.from_frozen_flat(frozen_path)
            rotation = flat.rotation
            for layout in layouts:
                layout_name = layout["name"]
                active_strategies = ["frozen_flat"] if layout_name == "flat_32" else strategies
                for strategy in active_strategies:
                    key = _candidate_key(layout_name, strategy)
                    if layout_name == "flat_32":
                        model = flat
                    else:
                        model = ProgressiveGainShapeQuantizer.fit(
                            train_edges,
                            edge_weights,
                            rotation,
                            int(layout["coarse_bytes"]),
                            int(layout["residual_bytes"]),
                            strategy,
                            train_x,
                            seed,
                            config["quantizer"],
                            layout_name,
                        )
                    payload = _event_payload(model, evaluation, selection_events)
                    metrics = projection_metrics(payload)
                    row = {"candidate": key, "layout": layout_name, "strategy": strategy, "seed": seed, **metrics}
                    selection_rows.append(row)
                    artifact_path = output_dir / "models" / f"{key}-seed{seed}.npz"
                    metadata = save_progressive_model(
                        model,
                        artifact_path,
                        {
                            "candidate": key,
                            "seed": seed,
                            "relative_path": str(artifact_path.relative_to(output_dir)),
                            "source_stage2_1_artifact_sha256": binding["artifact_sha256"],
                        },
                    )
                    artifact_rows.append(metadata)
                    model_index[(seed, key)] = (artifact_path, metadata)

        summary: dict[str, dict[str, float]] = {}
        for key in sorted({row["candidate"] for row in selection_rows}):
            rows = [row for row in selection_rows if row["candidate"] == key]
            summary[key] = {
                "mean_full_projection_abs_q99": float(np.mean([row["full_projection_abs_q99"] for row in rows])),
                "mean_full_projection_abs_q999": float(np.mean([row["full_projection_abs_q999"] for row in rows])),
                "mean_coarse_projection_abs_q99": float(np.mean([row["coarse_projection_abs_q99"] for row in rows])),
            }
        baseline_key = "flat_32__frozen_flat"
        baseline_q99 = summary[baseline_key]["mean_full_projection_abs_q99"]
        eligible = [
            key for key, values in summary.items()
            if key != baseline_key
            and values["mean_full_projection_abs_q99"]
            <= baseline_q99 * (1.0 + float(config["gate"]["max_final_q99_relative_degradation"]))
        ]
        selected_key = min(
            eligible,
            key=lambda key: (
                summary[key]["mean_coarse_projection_abs_q99"],
                summary[key]["mean_full_projection_abs_q99"],
            ),
        ) if eligible else baseline_key
        selection_report = {
            "selected_candidate": selected_key,
            "fallback_candidate": baseline_key,
            "summary": summary,
            "selection_rows": selection_rows,
            "representation_was_frozen_before_calibration": True,
        }
        write_json(output_dir / "selection_report.json", selection_report)

        calibration_entries: list[dict[str, Any]] = []
        final_rows: list[dict[str, Any]] = []
        final_candidates = [baseline_key] if selected_key == baseline_key else [baseline_key, selected_key]
        for seed in seeds:
            for key in final_candidates:
                artifact_path, metadata = model_index[(seed, key)]
                model = ProgressiveGainShapeQuantizer.load(artifact_path, metadata)
                train_payload = _event_payload(model, evaluation, selection_events)
                calibration_payload = _event_payload(model, evaluation, calibration_events)
                code_model = _fit_code_information(train_payload, seed, config["code_information"])
                code_cutoff = _calibrate_score_cutoff(
                    _scores(code_model, calibration_payload),
                    calibration_payload["good_candidate"],
                    float(config["diagnostic_false_prune_alpha"]),
                )
                radius = fit_stopping_radius(
                    calibration_payload["coarse_projection_error"],
                    float(config["stopping_projection_coverage"]),
                )
                calibration_entries.append(
                    {"candidate": key, "seed": seed, "stopping_radius": radius, "code_score_cutoff": code_cutoff}
                )
                final_payload = _event_payload(model, evaluation, final_events)
                final_rows.append(
                    {
                        "candidate": key,
                        "seed": seed,
                        **projection_metrics(final_payload),
                        **evaluate_stopping(final_payload, radius),
                        **_score_metrics(_scores(code_model, final_payload), code_cutoff, final_payload),
                    }
                )
        calibration_manifest = {
            "entries": calibration_entries,
            "risk_unit": "event_conditioned_on_good_candidate_diagnostic_only",
            "cutoffs_frozen_before_final_test": True,
        }
        write_json(output_dir / "calibration_manifest.json", calibration_manifest)

        def mean(candidate: str, metric: str) -> float:
            return float(np.mean([row[metric] for row in final_rows if row["candidate"] == candidate]))

        selected_final_q99 = mean(selected_key, "full_projection_abs_q99")
        baseline_final_q99 = mean(baseline_key, "full_projection_abs_q99")
        criteria = {
            "final_q99_noninferior": selected_final_q99 <= baseline_final_q99 * (
                1.0 + float(config["gate"]["max_final_q99_relative_degradation"])
            ),
            "coarse_only_fraction": mean(selected_key, "coarse_only_fraction") >= float(
                config["gate"]["min_coarse_only_fraction"]
            ),
            "code_information_noninferior": mean(selected_key, "code_information_coverage") >= mean(
                baseline_key, "code_information_coverage"
            ),
        }
        selected_strategy = next(row["strategy"] for row in selection_rows if row["candidate"] == selected_key)
        if selected_strategy in {"frozen_flat", "uniform"}:
            criteria["stable_non_uniform_gain"] = selected_key == baseline_key
        else:
            selected_layout = next(row["layout"] for row in selection_rows if row["candidate"] == selected_key)
            uniform_key = _candidate_key(selected_layout, "uniform")
            criteria["stable_non_uniform_gain"] = all(
                next(row for row in selection_rows if row["candidate"] == selected_key and row["seed"] == seed)[
                    "full_projection_abs_q99"
                ]
                < next(row for row in selection_rows if row["candidate"] == uniform_key and row["seed"] == seed)[
                    "full_projection_abs_q99"
                ]
                for seed in seeds
            )
        gate_pass = all(criteria.values())
        decision = "GO_PROGRESSIVE_STAGE3_DEV" if gate_pass and selected_key != baseline_key else "RETAIN_FROZEN_FLAT_32B"
        gate_report = {
            "decision": decision,
            "selected_candidate": selected_key if decision == "GO_PROGRESSIVE_STAGE3_DEV" else baseline_key,
            "criteria": criteria,
            "selected_final_q99": selected_final_q99,
            "baseline_final_q99": baseline_final_q99,
            "risk_status": "PENDING_RISK_ESTIMATOR",
            "real_pruning_authorized": False,
            "final_test_consumed": True,
        }
        write_json(output_dir / "final_metrics.json", {"rows": final_rows})
        write_json(output_dir / "gate_report.json", gate_report)
        semantic = {
            "format": STAGE3_FORMAT,
            "version": STAGE3_VERSION,
            "input_manifest_sha256": _sha256(input_path),
            "contract_sha256": input_manifest["contract_sha256"],
            "selected_flat_representation": "gain_shape_opq",
            "selected_progressive_candidate": gate_report["selected_candidate"],
            "selection_report": selection_report,
            "calibration_manifest": calibration_manifest,
            "final_rows": final_rows,
            "gate_report": gate_report,
        }
        manifest = {
            **semantic,
            "stage3_results_semantic_sha256": sha256_bytes(canonical_json_bytes(semantic)),
            "artifacts": artifact_rows,
            "outputs": {
                name: _sha256(output_dir / name)
                for name in ["selection_report.json", "calibration_manifest.json", "final_metrics.json", "gate_report.json"]
            },
            "protocol": {
                "fresh_evaluation_queries": True,
                "flat_representation_reselection": False,
                "selection_frozen_before_calibration": True,
                "cutoffs_frozen_before_final_test": True,
                "risk_is_diagnostic_only": True,
            },
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
        }
        write_json(output_dir / "stage3_manifest.json", manifest)
        return manifest
    finally:
        training.close()
        evaluation.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = run_experiment(args.config, args.output_dir)
    print(json.dumps({
        "decision": manifest["gate_report"]["decision"],
        "selected": manifest["selected_progressive_candidate"],
        "semantic_sha256": manifest["stage3_results_semantic_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
