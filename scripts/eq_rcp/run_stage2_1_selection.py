#!/usr/bin/env python3
"""Run the EQ-RCP Stage 2.1 causal representation-selection experiment."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from scipy.stats import beta

from operational_dataset import (
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
)
from stage2_quantization import (
    Stage1OperationalData,
    code_oracle_scores,
    conservative_cutoff,
    decision_metrics,
    edge_training_set,
    fit_code_oracle,
    paired_bootstrap_ci,
    per_query_rows,
    projection_metrics,
    reconstruct_events,
    results_semantic_sha256,
    save_quantizer,
    smooth_threshold_weights,
)
from stage2_1_quantization import (
    STAGE21_FORMAT,
    STAGE21_VERSION,
    fit_stage21_quantizer,
    operational_loss_decomposition,
    radial_tangential_diagnostics,
)


def _prepare_output(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"Stage 2.1 output directory must be empty: {path}")


def _semantic_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in config.items()
        if key not in {"stage1_dataset_dir", "notes"}
    }


def _weights(values: dict[str, np.ndarray], spec: dict[str, Any]) -> np.ndarray:
    mode = spec["mode"]
    if mode == "raw":
        return values["raw_event_weight"].astype(np.float64)
    if mode == "per_query":
        return values["per_query_weight"].astype(np.float64)
    if mode == "smooth_threshold":
        return smooth_threshold_weights(
            values["margin"], float(spec["minimum"]), float(spec["tau"])
        )
    raise ValueError(f"unsupported Stage 2.1 objective weight mode: {mode}")


def _binomial_upper(errors: int, trials: int, confidence: float) -> float:
    if trials <= 0:
        raise ValueError("risk calibration requires good-candidate trials")
    if errors >= trials:
        return 1.0
    return float(beta.ppf(confidence, errors + 1, trials - errors))


def _risk_controlled_cutoff(
    scores: np.ndarray, good_candidate: np.ndarray,
    target: float, confidence: float,
) -> tuple[float, dict[str, Any]]:
    unsafe = np.asarray(scores, dtype=np.float64)[np.asarray(good_candidate, dtype=bool)]
    if unsafe.size == 0:
        raise ValueError("risk calibration has no good candidates")
    allowed = -1
    upper = 1.0
    for errors in range(len(unsafe) + 1):
        candidate_upper = _binomial_upper(errors, len(unsafe), confidence)
        if candidate_upper <= target:
            allowed, upper = errors, candidate_upper
        else:
            break
    if allowed < 0:
        cutoff = float("inf")
        observed_errors = 0
        upper = _binomial_upper(0, len(unsafe), confidence)
    else:
        ordered = np.sort(unsafe)[::-1]
        cutoff = float(ordered[allowed]) if allowed < len(ordered) else float("-inf")
        observed_errors = int(np.sum(unsafe > cutoff))
        upper = _binomial_upper(observed_errors, len(unsafe), confidence)
    return cutoff, {
        "target": target, "confidence": confidence,
        "good_candidate_trials": int(len(unsafe)),
        "observed_false_prunes": observed_errors,
        "observed_rate": float(observed_errors / len(unsafe)),
        "upper_confidence_bound": upper,
        "pass": upper <= target,
    }


def _edge_payload(
    data: Stage1OperationalData,
    event_indices: np.ndarray,
    event_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = data.event_values(event_indices)
    edge_ids, inverse, edge_weights = edge_training_set(
        values["edge_id"].astype(np.int64), event_weights
    )
    return edge_ids, inverse, edge_weights, data.edge_vectors(edge_ids)


def _method_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    numeric = [
        "projection_mse", "projection_abs_q99", "projection_abs_q999",
        "threshold_near_projection_abs_q99", "calibrated_coverage",
        "calibrated_false_prune_rate_over_good", "code_oracle_coverage",
        "code_oracle_false_prune_rate_over_good", "radial_mse",
        "tangential_mse",
        "model_parameter_bytes",
    ]
    result: dict[str, dict[str, float]] = {}
    for method in sorted({str(row["method"]) for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        result[method] = {
            f"mean_{metric}": float(np.nanmean([float(row[metric]) for row in selected]))
            for metric in numeric
        }
        result[method]["seed_count"] = len(selected)
    return result


def _per_query_metric(
    rows: list[dict[str, Any]], method: str, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["method"] == method]
    query_ids = sorted({int(row["query_id"]) for row in selected})
    values = [
        np.mean([
            float(row[metric]) for row in selected
            if int(row["query_id"]) == query_id
        ])
        for query_id in query_ids
    ]
    return np.asarray(query_ids), np.asarray(values, dtype=np.float64)


def _candidate_criteria(
    name: str,
    baseline: str,
    summary: dict[str, dict[str, float]],
    rows: list[dict[str, Any]],
    per_query: list[dict[str, Any]],
    config: dict[str, Any],
    enforce_calibrated_risk: bool = True,
) -> dict[str, Any]:
    base = summary[baseline]
    candidate = summary[name]
    base_q99 = base["mean_projection_abs_q99"]
    q99_reduction = (
        base_q99 - candidate["mean_projection_abs_q99"]
    ) / max(base_q99, 1e-30)
    query_a, base_values = _per_query_metric(per_query, baseline, "projection_abs_q99")
    query_b, candidate_values = _per_query_metric(per_query, name, "projection_abs_q99")
    if not np.array_equal(query_a, query_b):
        raise ValueError("Stage 2.1 paired-query inputs are not aligned")
    bootstrap = paired_bootstrap_ci(
        base_values, candidate_values, int(config["bootstrap"]["seed"]),
        int(config["bootstrap"]["samples"]),
        float(config["bootstrap"]["confidence"]),
    )
    by_seed: dict[str, float] = {}
    for seed in config["seeds"]:
        base_row = next(row for row in rows if row["method"] == baseline and row["seed"] == seed)
        candidate_row = next(row for row in rows if row["method"] == name and row["seed"] == seed)
        by_seed[str(seed)] = (
            float(base_row["projection_abs_q99"]) - float(candidate_row["projection_abs_q99"])
        ) / max(float(base_row["projection_abs_q99"]), 1e-30)
    gate = config["gate"]
    near_ratio = candidate["mean_threshold_near_projection_abs_q99"] / max(
        base["mean_threshold_near_projection_abs_q99"], 1e-30
    )
    q999_ratio = candidate["mean_projection_abs_q999"] / max(
        base["mean_projection_abs_q999"], 1e-30
    )
    criteria = {
        "positive_q99": q99_reduction > 0.0,
        "target_q99": q99_reduction >= float(gate["target_q99_reduction"]),
        "paired_ci": bootstrap["ci_lower"] > float(gate["minimum_paired_ci_lower"]),
        "threshold_near_noninferior": near_ratio <= float(gate["maximum_near_q99_ratio"]),
        "q999_control": q999_ratio <= float(gate["maximum_q999_ratio"]),
        "diagnostic_coverage": candidate["mean_calibrated_coverage"] >= float(gate["minimum_diagnostic_coverage"]),
        "diagnostic_risk": candidate["mean_calibrated_false_prune_rate_over_good"] <= float(gate["maximum_false_prune_rate_over_good"]),
        "code_oracle_information": candidate["mean_code_oracle_coverage"] >= float(gate["minimum_code_oracle_coverage"]),
        "seed_stability": all(value > 0.0 for value in by_seed.values()),
    }
    maximum_metadata = int(gate.get("maximum_model_parameter_bytes", 2**63 - 1))
    criteria["runtime_metadata_budget"] = candidate["mean_model_parameter_bytes"] <= maximum_metadata
    required = [
        key for key in criteria
        if enforce_calibrated_risk or key != "diagnostic_risk"
    ]
    return {
        "method": name,
        "pass": all(bool(criteria[key]) for key in required),
        "target_q99_achieved": bool(criteria["target_q99"]),
        "q99_reduction": q99_reduction,
        "near_q99_ratio": near_ratio,
        "q999_ratio": q999_ratio,
        "paired_query_bootstrap": bootstrap,
        "q99_reduction_by_seed": by_seed,
        "criteria": criteria,
        "calibrated_risk_deferred": not enforce_calibrated_risk,
    }


REQUIRED_ROLES = [
    "quantizer_train", "decoder_train", "model_selection",
    "risk_calibration", "final_test",
]


def _selection_report(
    config: dict[str, Any], rows: list[dict[str, Any]],
    per_query: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = _method_summary(rows)
    baseline = "direct_delta_opq"
    if baseline not in summary:
        raise ValueError("Stage 2.1 requires direct_delta_opq as the frozen baseline")
    candidates = [name for name in summary if name not in {"direct_delta_pq", baseline}]
    reports = {
        name: _candidate_criteria(
            name, baseline, summary, rows, per_query, config,
            enforce_calibrated_risk=False,
        )
        for name in candidates
    }
    passing = [name for name, report in reports.items() if report["pass"]]
    selected = min(
        passing, key=lambda name: summary[name]["mean_projection_abs_q99"]
    ) if passing else None
    fallback_ok = (
        summary[baseline]["mean_calibrated_coverage"]
        >= float(config["gate"]["minimum_diagnostic_coverage"])
        and summary[baseline]["mean_code_oracle_coverage"]
        >= float(config["gate"]["minimum_code_oracle_coverage"])
        and summary[baseline]["mean_calibrated_false_prune_rate_over_good"]
        <= float(config["gate"]["maximum_false_prune_rate_over_good"])
        and summary[baseline]["mean_model_parameter_bytes"]
        <= int(config["gate"].get("maximum_model_parameter_bytes", 2**63 - 1))
    )
    if selected is None and bool(config["gate"].get("allow_opq_fallback", True)) and fallback_ok:
        selected = baseline
    return {
        "selected_flat_representation": selected,
        "selected_parameterization_contract": (
            next(row["parameterization_contract"] for row in rows if row["method"] == selected)
            if selected else None
        ),
        "safety_fallback": baseline,
        "raw_opq_fallback_eligible": fallback_ok,
        "candidate_reports": reports,
        "method_summary": summary,
        "selection_role": "model_selection",
        "frozen": True,
    }


def _final_gate_report(
    config: dict[str, Any], data: Stage1OperationalData,
    selection_report: dict[str, Any], rows: list[dict[str, Any]],
    per_query: list[dict[str, Any]], fully_sampled: bool,
    calibration_manifest: dict[str, Any],
) -> dict[str, Any]:
    selected = selection_report["selected_flat_representation"]
    baseline = "direct_delta_opq"
    summary = _method_summary(rows)
    if selected is None:
        final_pass = False
        final_criteria = None
    elif selected == baseline:
        value = summary[baseline]
        gate = config["gate"]
        criteria = {
            "diagnostic_coverage": value["mean_calibrated_coverage"] >= float(gate["minimum_diagnostic_coverage"]),
            "diagnostic_risk": value["mean_calibrated_false_prune_rate_over_good"] <= float(gate["maximum_false_prune_rate_over_good"]),
            "code_oracle_information": value["mean_code_oracle_coverage"] >= float(gate["minimum_code_oracle_coverage"]),
            "runtime_metadata_budget": value["mean_model_parameter_bytes"] <= int(gate.get("maximum_model_parameter_bytes", 2**63 - 1)),
        }
        final_pass = all(criteria.values())
        final_criteria = {"method": baseline, "pass": final_pass, "criteria": criteria, "opq_fallback": True}
    else:
        final_criteria = _candidate_criteria(
            selected, baseline, summary, rows, per_query, config
        )
        final_pass = bool(final_criteria["pass"])
    calibration_pass = all(
        bool(entry["scalar_risk"]["pass"] and entry["code_oracle_risk"]["pass"])
        for entry in calibration_manifest["entries"]
    )
    final_pass = final_pass and calibration_pass
    formal = (
        bool(config.get("allow_formal_gate_decision", False))
        and config.get("run_role") == "formal_representation_selection"
        and data.manifest["dataset_role"] == "formal"
        and data.manifest["adjacency_validation"]["status"] == "PASS"
        and fully_sampled and len(config["seeds"]) >= 3
    )
    development = (
        config.get("run_role") == "development_full"
        and data.manifest["dataset_role"] == "development"
        and fully_sampled and len(config["seeds"]) >= 3
    )
    if formal:
        decision = "GO_CPP" if final_pass else "NO_GO_REPRESENTATION"
    elif development:
        decision = "GO_STAGE3_DEV" if final_pass else "NO_GO_REPRESENTATION"
    else:
        decision = "PENDING_FORMAL_STAGE2_1"
    return {
        "decision": decision,
        "formal_eligible": formal,
        "development_full_eligible": development,
        "selected_flat_representation": selected,
        "selected_parameterization_contract": selection_report["selected_parameterization_contract"],
        "safety_fallback": selection_report["safety_fallback"],
        "final_test_pass": final_pass,
        "calibration_risk_ucb_pass": calibration_pass,
        "final_test_criteria": final_criteria,
        "method_summary": summary,
        "selection_frozen_before_calibration": True,
        "cutoffs_frozen_before_final_test": True,
        "gate_role": "final_test",
    }


def _evaluate_model(
    model: Any, oracle: Any, data: Stage1OperationalData,
    event_indices: np.ndarray, method: str, seed: int,
    metadata: dict[str, Any], artifact: dict[str, Any], near_limit: float,
    scalar_cutoff: float, oracle_cutoff: float, config: dict[str, Any],
    training_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    payload = reconstruct_events(model, data, event_indices)
    calibrated = decision_metrics(payload, scalar_cutoff)
    oracle_scores = code_oracle_scores(oracle, payload)
    oracle_payload = dict(payload)
    oracle_payload["estimated_margin"] = oracle_scores
    code_oracle = decision_metrics(oracle_payload, oracle_cutoff)
    projection = projection_metrics(payload, near_limit)
    delta = data.edge_vectors(payload["edge_ids"])
    reconstructed, _ = model.reconstruct(delta)
    x = data.event_x(event_indices)
    radial, radial_rows = radial_tangential_diagnostics(
        delta, reconstructed, payload["edge_inverse"], x, payload["margin"],
        int(config["analysis"]["length_bins"]),
        int(config["analysis"]["margin_bins"]),
    )
    operational = operational_loss_decomposition(
        delta, reconstructed, payload["edge_inverse"], x,
        int(metadata["subquantizers"]),
    )
    row = {
        "method": method, "seed": seed,
        "code_bytes": int(metadata["code_bytes"]),
        "physical_code_bytes": int(metadata["code_bytes"]),
        "metadata_float_count": int(metadata.get("metadata_float_count", 0)),
        "model_parameter_bytes": artifact["model_parameter_bytes"],
        "compressed_artifact_bytes": artifact["compressed_artifact_bytes"],
        "parameterization_contract": metadata["kind"],
        "training_seconds": training_seconds,
        "evaluation_query_count": int(len(np.unique(payload["query_id"]))),
        "evaluation_event_count": int(len(event_indices)),
        **projection,
        "calibrated_cutoff": scalar_cutoff,
        "calibrated_coverage": calibrated["coverage"],
        "calibrated_false_prune_rate_over_good": calibrated["false_prune_rate_over_good"],
        "code_oracle_cutoff": oracle_cutoff,
        "code_oracle_coverage": code_oracle["coverage"],
        "code_oracle_false_prune_rate_over_good": code_oracle["false_prune_rate_over_good"],
        **radial, **operational,
    }
    return row, per_query_rows(method, seed, payload), radial_rows


def run_experiment(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("format") != STAGE21_FORMAT or config.get("version") != STAGE21_VERSION:
        raise ValueError("unsupported Stage 2.1 config format/version")
    _prepare_output(output_dir)
    data = Stage1OperationalData(Path(config["stage1_dataset_dir"]))
    try:
        actual_hash = data.manifest["dataset_semantic_sha256"]
        expected_hash = config.get("expected_stage1_dataset_semantic_sha256")
        if expected_hash and expected_hash != actual_hash:
            raise ValueError("Stage 1 dataset semantic SHA-256 mismatch")
        selection = config["selection"]
        roles = REQUIRED_ROLES
        if set(selection) - {"seed"} != set(roles):
            raise ValueError("Stage 2.1 requires exactly five selection roles")
        split_queries: dict[str, np.ndarray] = {}
        split_events: dict[str, np.ndarray] = {}
        for role in roles:
            entry = selection[role]
            if not str(entry.get("split", "")).endswith(role):
                raise ValueError(f"Stage 2.1 split/role mismatch: {role}")
            maximum = entry.get("max_queries")
            split_queries[role] = data.split_query_ids(
                entry["split"], None if maximum is None else int(maximum),
                int(selection["seed"]),
            )
            split_events[role] = data.event_indices(split_queries[role])
        for index, left in enumerate(roles):
            for right in roles[index + 1:]:
                if np.intersect1d(split_queries[left], split_queries[right]).size:
                    raise ValueError("Stage 2.1 query selections overlap")
        fully_sampled = all(selection[role].get("max_queries") is None for role in roles)

        train_values = data.event_values(split_events["quantizer_train"])
        train_weights = _weights(train_values, config["objective_weight"])
        train_edge_ids, train_inverse, edge_weights, train_vectors = _edge_payload(
            data, split_events["quantizer_train"], train_weights
        )
        train_x = data.event_x(split_events["quantizer_train"])
        decoder_values = data.event_values(split_events["decoder_train"])
        decoder_weights = _weights(decoder_values, config["objective_weight"])
        decoder_edge_ids, decoder_inverse, _, decoder_vectors = _edge_payload(
            data, split_events["decoder_train"], decoder_weights
        )
        decoder_x = data.event_x(split_events["decoder_train"])
        validation = {
            "edge_vectors": decoder_vectors,
            "event_inverse": decoder_inverse,
            "x": decoder_x,
            "y": decoder_values["y_exact"],
            "weights": decoder_weights,
        }
        near_limit = float(np.quantile(
            np.abs(train_values["margin"]),
            float(config["evaluation"]["near_margin_quantile"]),
        ))

        model_dir = output_dir / "models"
        model_dir.mkdir()
        selection_rows: list[dict[str, Any]] = []
        selection_per_query: list[dict[str, Any]] = []
        artifacts: list[dict[str, Any]] = []
        models: dict[tuple[int, str], Any] = {}
        oracles: dict[tuple[int, str], Any] = {}
        model_metadata: dict[tuple[int, str], dict[str, Any]] = {}
        artifact_by_key: dict[tuple[int, str], dict[str, Any]] = {}
        training_time: dict[tuple[int, str], float] = {}
        for seed_value in config["seeds"]:
            seed = int(seed_value)
            for method_value in config["methods"]:
                method = str(method_value)
                started = time.perf_counter()
                model, details = fit_stage21_quantizer(
                    method, train_vectors, edge_weights, train_inverse, train_x,
                    train_values["y_exact"], train_weights, config["quantizer"],
                    seed, validation,
                )
                training_seconds = time.perf_counter() - started
                metadata = model.artifact_metadata()
                if int(metadata["code_bytes"]) != int(config["quantizer"]["code_bytes"]):
                    raise ValueError(f"method {method} violated fixed code budget")
                artifact_path = model_dir / f"{method}-seed{seed}.npz"
                artifact = save_quantizer(model, artifact_path, {
                    "stage": "2.1", "seed": seed,
                    "stage1_dataset_semantic_sha256": actual_hash,
                    "training_edge_count": int(len(train_edge_ids)),
                    "encoder_inputs": ["Delta", "static_group"],
                    "forbidden_encoder_inputs": ["query", "threshold", "label"],
                    "parameterization_contract": metadata["kind"],
                    **details,
                })
                artifact["relative_path"] = str(artifact_path.relative_to(output_dir))
                artifact["model_parameter_bytes"] = int(sum(
                    np.asarray(value).nbytes for value in model.artifact_arrays().values()
                ))
                artifact["compressed_artifact_bytes"] = int(artifact_path.stat().st_size)
                artifacts.append(artifact)
                key = (seed, method)
                models[key] = model
                model_metadata[key] = metadata
                artifact_by_key[key] = artifact
                training_time[key] = training_seconds

                decoder = reconstruct_events(model, data, split_events["decoder_train"])
                alpha = float(config["evaluation"]["diagnostic_false_prune_alpha"])
                diagnostic_cutoff = conservative_cutoff(
                    decoder["estimated_margin"], decoder["good_candidate"], alpha
                )
                oracle = fit_code_oracle(decoder, seed, config["flexible_code_oracle"])
                oracles[key] = oracle
                decoder_oracle_scores = code_oracle_scores(oracle, decoder)
                diagnostic_oracle_cutoff = conservative_cutoff(
                    decoder_oracle_scores, decoder["good_candidate"], alpha
                )
                row, query_rows, _ = _evaluate_model(
                    model, oracle, data, split_events["model_selection"], method,
                    seed, metadata, artifact, near_limit, diagnostic_cutoff,
                    diagnostic_oracle_cutoff, config, training_seconds,
                )
                row["evaluation_role"] = "model_selection"
                selection_rows.append(row)
                selection_per_query.extend(query_rows)
                print(f"completed {method} seed={seed}: Q99={row['projection_abs_q99']:.8g}")

        selection_comparison_path = output_dir / "selection_comparison.parquet"
        selection_per_query_path = output_dir / "selection_per_query_metrics.parquet"
        pq.write_table(pa.Table.from_pylist(selection_rows), selection_comparison_path, compression="zstd")
        pq.write_table(pa.Table.from_pylist(selection_per_query), selection_per_query_path, compression="zstd")
        selection_report = _selection_report(config, selection_rows, selection_per_query)
        write_json(output_dir / "selection_report.json", selection_report)
        selected = selection_report["selected_flat_representation"]
        audit_methods = sorted({name for name in [selected, "direct_delta_opq"] if name is not None})

        calibration_entries: list[dict[str, Any]] = []
        frozen_cutoffs: dict[tuple[int, str], tuple[float, float]] = {}
        for seed_value in config["seeds"]:
            seed = int(seed_value)
            for method in audit_methods:
                key = (seed, method)
                risk_payload = reconstruct_events(models[key], data, split_events["risk_calibration"])
                risk_confidence = float(config["evaluation"].get("risk_confidence", 0.95))
                scalar_cutoff, scalar_risk = _risk_controlled_cutoff(
                    risk_payload["estimated_margin"], risk_payload["good_candidate"],
                    float(config["gate"]["maximum_false_prune_rate_over_good"]),
                    risk_confidence,
                )
                oracle_scores = code_oracle_scores(oracles[key], risk_payload)
                oracle_cutoff, oracle_risk = _risk_controlled_cutoff(
                    oracle_scores, risk_payload["good_candidate"],
                    float(config["gate"]["maximum_false_prune_rate_over_good"]),
                    risk_confidence,
                )
                frozen_cutoffs[key] = (scalar_cutoff, oracle_cutoff)
                calibration_entries.append({
                    "method": method, "seed": seed,
                    "scalar_cutoff": scalar_cutoff,
                    "code_oracle_cutoff": oracle_cutoff,
                    "scalar_risk": scalar_risk,
                    "code_oracle_risk": oracle_risk,
                    "calibration_query_count": int(len(split_queries["risk_calibration"])),
                    "calibration_event_count": int(len(split_events["risk_calibration"])),
                })
        calibration_manifest = {
            "role": "risk_calibration", "frozen": True,
            "risk_unit": "event_conditioned_on_good_candidate",
            "selected_flat_representation": selected,
            "safety_fallback": "direct_delta_opq",
            "selection_report_sha256": sha256_file(output_dir / "selection_report.json"),
            "entries": calibration_entries,
        }
        write_json(output_dir / "calibration_manifest.json", calibration_manifest)

        final_rows: list[dict[str, Any]] = []
        final_per_query: list[dict[str, Any]] = []
        strata: list[dict[str, Any]] = []
        for seed_value in config["seeds"]:
            seed = int(seed_value)
            for method in audit_methods:
                key = (seed, method)
                scalar_cutoff, oracle_cutoff = frozen_cutoffs[key]
                row, query_rows, radial_rows = _evaluate_model(
                    models[key], oracles[key], data, split_events["final_test"],
                    method, seed, model_metadata[key], artifact_by_key[key], near_limit,
                    scalar_cutoff, oracle_cutoff, config, training_time[key],
                )
                row["evaluation_role"] = "final_test"
                final_rows.append(row)
                final_per_query.extend(query_rows)
                strata.extend({"method": method, "seed": seed, **item} for item in radial_rows)

        comparison_path = output_dir / "comparison.parquet"
        per_query_path = output_dir / "per_query_metrics.parquet"
        strata_path = output_dir / "radial_tangential_strata.parquet"
        pq.write_table(pa.Table.from_pylist(final_rows), comparison_path, compression="zstd")
        pq.write_table(pa.Table.from_pylist(final_per_query), per_query_path, compression="zstd")
        pq.write_table(pa.Table.from_pylist(strata), strata_path, compression="zstd")
        gate = _final_gate_report(
            config, data, selection_report, final_rows, final_per_query,
            fully_sampled, calibration_manifest,
        )
        write_json(output_dir / "gate_report.json", gate)
        attribution = {
            "raw_opq_q99": selection_report["method_summary"]["direct_delta_opq"]["mean_projection_abs_q99"],
            "norm_corrected_opq_q99": selection_report["method_summary"].get("norm_corrected_opq", {}).get("mean_projection_abs_q99"),
            "gain_shape_opq_q99": selection_report["method_summary"].get("gain_shape_opq", {}).get("mean_projection_abs_q99"),
            "raw_pq_q99": selection_report["method_summary"].get("direct_delta_pq", {}).get("mean_projection_abs_q99"),
            "norm_corrected_pq_q99": selection_report["method_summary"].get("norm_corrected_pq", {}).get("mean_projection_abs_q99"),
            "direction_pq_q99": selection_report["method_summary"].get("direction_pq_exact_length", {}).get("mean_projection_abs_q99"),
            "rule": "Attribute effects only through optimizer-matched contrasts: B1->B3 for exact length, B3->B5 for OPQ shape allocation, and B0->B2->B4 for PQ.",
        }
        raw_opq = attribution["raw_opq_q99"]
        norm_opq = attribution["norm_corrected_opq_q99"]
        shape_opq = attribution["gain_shape_opq_q99"]
        if norm_opq is not None:
            attribution["exact_length_opq_q99_reduction"] = float(
                (raw_opq - norm_opq) / max(float(raw_opq), 1e-30)
            )
        if norm_opq is not None and shape_opq is not None:
            attribution["shape_allocation_incremental_q99_reduction"] = float(
                (norm_opq - shape_opq) / max(float(norm_opq), 1e-30)
            )
            tolerance = float(config["analysis"].get("attribution_tracking_tolerance", 0.02))
            attribution["shape_increment_distinguishable"] = (
                attribution["shape_allocation_incremental_q99_reduction"] > tolerance
            )
            attribution["causal_conclusion"] = (
                "exact_length_plus_distinguishable_shape_increment"
                if attribution["shape_increment_distinguishable"]
                else "exact_length_explains_opq_gain_at_configured_tolerance"
            )
        write_json(output_dir / "causal_attribution.json", attribution)
        semantic_payload = {
            "format": STAGE21_FORMAT, "version": STAGE21_VERSION,
            "semantic_config": _semantic_config(config),
            "stage1_dataset_semantic_sha256": actual_hash,
            "query_selection": {role: split_queries[role].tolist() for role in roles},
            "selection_comparison": [{k: v for k, v in row.items() if k != "training_seconds"} for row in selection_rows],
            "selection_per_query": selection_per_query,
            "final_comparison": [{k: v for k, v in row.items() if k != "training_seconds"} for row in final_rows],
            "final_per_query": final_per_query, "strata": strata,
            "artifacts": [{k: v for k, v in item.items() if k not in {"relative_path", "artifact_sha256"}} for item in artifacts],
            "selection_report": selection_report,
            "calibration_manifest": calibration_manifest,
            "gate": gate, "causal_attribution": attribution,
        }
        manifest = {
            "format": STAGE21_FORMAT, "version": STAGE21_VERSION,
            "run_role": config["run_role"],
            "stage1_dataset_dir": str(data.dataset_dir),
            "stage1_dataset_semantic_sha256": actual_hash,
            "stage2_1_results_semantic_sha256": results_semantic_sha256(semantic_payload),
            "semantic_config_sha256": sha256_bytes(canonical_json_bytes(_semantic_config(config))),
            "fully_sampled": fully_sampled,
            "selection": {role: {"split": selection[role]["split"], "query_count": int(len(split_queries[role])), "event_count": int(len(split_events[role])), "query_ids_sha256": sha256_bytes(np.asarray(split_queries[role], dtype="<i8").tobytes())} for role in roles},
            "artifacts": artifacts,
            "protocol": {
                "evaluator_protocol_version": 2,
                "required_roles": roles,
                "representation_frozen_before_risk_calibration": True,
                "risk_cutoffs_frozen_before_final_test": True,
                "final_test_consumed": True,
                "selection_report_sha256": sha256_file(output_dir / "selection_report.json"),
                "calibration_manifest_sha256": sha256_file(output_dir / "calibration_manifest.json"),
            },
            "outputs": {path.name: sha256_file(path) for path in [selection_comparison_path, selection_per_query_path, comparison_path, per_query_path, strata_path, output_dir / "selection_report.json", output_dir / "calibration_manifest.json", output_dir / "gate_report.json", output_dir / "causal_attribution.json"]},
            "gate_decision": gate["decision"],
            "selected_flat_representation": gate["selected_flat_representation"],
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
        }
        write_json(output_dir / "stage2_1_manifest.json", manifest)
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
    manifest = run_experiment(args.config, args.output_dir)
    print(json.dumps({
        "gate_decision": manifest["gate_decision"],
        "selected_flat_representation": manifest["selected_flat_representation"],
        "semantic_sha256": manifest["stage2_1_results_semantic_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
