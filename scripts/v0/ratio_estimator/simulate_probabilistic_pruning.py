#!/usr/bin/env python3
"""Simulate frozen Phase-2 probabilistic pruning on held-out records.

This is an observe-only, record-level counterfactual.  It never changes the
HNSW trajectory and therefore reports potential exact-distance savings, not
real search speedup or recall at a changed control flow.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hnsw-ratio-phase3-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import apply_probabilistic_bound as phase2_apply
import probabilistic_bound_core as bound_core
import ratio_estimator_core as estimator_core


PHASE3_VERSION = "v0_ratio_offline_pruning_phase3_v1"
SIMULATION_MANIFEST_FORMAT = "v0_ratio_offline_pruning_simulation"
RAW_ESTIMATE_COLUMN = "distance_hat_raw_clipped"
SAMPLING_HASH_ID = "v0_search_runner_mix64_xor_query_current_candidate_v1"
NEAR_MARGIN_RELATIVE_THRESHOLD = 1e-2
KAPPA_BINS = (-math.inf, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, math.inf)
KAPPA_LABELS = ("<0.5", "[0.5,0.6)", "[0.6,0.7)", "[0.7,0.8)", "[0.8,0.9)", "[0.9,1.0)", ">=1.0")


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return math.nan if denominator == 0 else float(numerator) / float(denominator)


def stable_decision_sha256(data: pd.DataFrame, lb_column: str, decision_column: str) -> str:
    ordered = data[["query_id", "current_node_id", "candidate_id", lb_column, decision_column]].copy()
    for name in ("query_id", "current_node_id", "candidate_id"):
        ordered[name] = ordered[name].astype(str)
    ordered[lb_column] = ordered[lb_column].astype(float).map(float.hex)
    ordered[decision_column] = ordered[decision_column].astype(bool).astype(int)
    ordered = ordered.sort_values(
        ["query_id", "current_node_id", "candidate_id", lb_column, decision_column],
        kind="mergesort",
    )
    return estimator_core.sha256_text(ordered.to_csv(index=False, lineterminator="\n"))


def normalize_pruning_inputs(data: pd.DataFrame) -> pd.DataFrame:
    required = {"current_would_prune", "raw_would_prune", "oracle_would_prune"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError("Phase-3 input is missing pruning flags: " + ", ".join(missing))
    result = data.copy()
    for name in sorted(required):
        result[f"input_{name}"] = estimator_core.normalize_bool(result[name], name)
    result["current_prune"] = result["current_lb_safe"] > result["threshold"]
    result["raw_point_prune"] = (
        result["analysis_valid"]
        & np.isfinite(result[RAW_ESTIMATE_COLUMN])
        & (result[RAW_ESTIMATE_COLUMN] > result["threshold"])
    )
    result["oracle_prune"] = result["shadow_exact_squared_distance"] > result["threshold"]
    result["oracle_margin"] = result["shadow_exact_squared_distance"] - result["threshold"]
    result["near_margin"] = result["relative_abs_margin"] <= NEAR_MARGIN_RELATIVE_THRESHOLD
    return result


def apply_frozen_quantile(
    data: pd.DataFrame,
    quantile: float,
    *,
    kappa_min: float | None = None,
) -> pd.DataFrame:
    if not math.isfinite(float(quantile)):
        raise ValueError("quantile must be finite")
    result = data.copy()
    if kappa_min is None:
        eligible = result["ratio_eligible"].astype(bool)
    else:
        eligible = (
            result["analysis_valid"]
            & np.isfinite(result["kappa_meta"])
            & (result["kappa_meta"] >= float(kappa_min))
        )
    ratio_lb = np.maximum(
        0.0,
        result["deployment_distance_hat"].to_numpy(float) - float(quantile),
    )
    result["simulation_ratio_eligible"] = eligible.to_numpy(bool)
    result["simulation_ratio_lb"] = ratio_lb
    result["simulation_lb"] = np.where(
        result["simulation_ratio_eligible"],
        ratio_lb,
        result["current_lb_safe"].to_numpy(float),
    )
    result["simulation_used_current_fallback"] = ~result["simulation_ratio_eligible"]
    # Strictly greater-than is the frozen HNSW threshold semantic.
    result["simulation_prune"] = result["simulation_lb"] > result["threshold"]
    result["simulation_extra_prune"] = result["simulation_prune"] & ~result["current_prune"]
    result["simulation_false_prune"] = result["simulation_prune"] & ~result["oracle_prune"]
    result["simulation_false_prune_magnitude"] = np.maximum(
        0.0,
        result["simulation_lb"] - result["shadow_exact_squared_distance"],
    )
    if (result["simulation_lb"] < 0.0).any():
        raise AssertionError("simulated lower bound became negative")
    if (result.loc[~result["analysis_valid"], "simulation_ratio_eligible"]).any():
        raise AssertionError("invalid records became estimator eligible")
    return result


def summarize_decisions(
    data: pd.DataFrame,
    *,
    operating_point_id: str,
    strategy: str,
    strategy_role: str,
    lb_column: str,
    decision_column: str,
    formal_candidate: bool,
    calibration_level: str | None = None,
    nominal_alpha: float | None = None,
    quantile: float | None = None,
    calibrator_sha256: str | None = None,
    status: str = "supported",
) -> dict[str, object]:
    decision = data[decision_column].astype(bool)
    current = data["current_prune"].astype(bool)
    oracle = data["oracle_prune"].astype(bool)
    false_prune = decision & ~oracle
    extra = decision & ~current
    safe_extra = decision & oracle & ~current
    n_records = int(len(data))
    n_queries = int(data["query_id"].nunique())
    prune_count = int(decision.sum())
    current_count = int(current.sum())
    oracle_count = int(oracle.sum())
    false_count = int(false_prune.sum())
    exposed_queries = int(data.assign(_false=false_prune).groupby("query_id", sort=False)["_false"].any().sum())
    oracle_gap = max(0, oracle_count - current_count)
    magnitude = (data.loc[false_prune, lb_column] - data.loc[false_prune, "shadow_exact_squared_distance"]).clip(lower=0.0)
    median_true_distance = float(data["shadow_exact_squared_distance"].median())
    max_magnitude = float(magnitude.max()) if len(magnitude) else 0.0
    row: dict[str, object] = {
        "operating_point_id": operating_point_id,
        "strategy": strategy,
        "strategy_role": strategy_role,
        "formal_candidate": bool(formal_candidate),
        "calibration_level": calibration_level,
        "nominal_alpha": nominal_alpha,
        "status": status,
        "quantile": quantile,
        "calibrator_sha256": calibrator_sha256,
        "record_count": n_records,
        "query_count": n_queries,
        "prune_count": prune_count,
        "coverage_rate": safe_ratio(prune_count, n_records),
        "potential_exact_distance_savings_count": prune_count,
        "potential_exact_distance_savings_rate": safe_ratio(prune_count, n_records),
        "current_prune_count": current_count,
        "extra_prune_count": int(extra.sum()),
        "extra_coverage_rate": safe_ratio(int(extra.sum()), n_records),
        "safe_extra_prune_count": int(safe_extra.sum()),
        "oracle_prune_count": oracle_count,
        "oracle_recovery_count": int((decision & oracle).sum()),
        "oracle_recovery_rate": safe_ratio(int((decision & oracle).sum()), oracle_count),
        "oracle_gap_count": oracle_gap,
        "oracle_gap_recovery_count": int(safe_extra.sum()),
        "oracle_gap_recovery_rate": safe_ratio(int(safe_extra.sum()), oracle_gap),
        "false_prune_count": false_count,
        "false_prune_event_rate": safe_ratio(false_count, n_records),
        "false_prune_decision_fraction": safe_ratio(false_count, prune_count),
        "false_prune_query_count": exposed_queries,
        "false_prune_query_exposure_rate": safe_ratio(exposed_queries, n_queries),
        "false_prune_magnitude_mean": float(magnitude.mean()) if len(magnitude) else 0.0,
        "false_prune_magnitude_p95": float(magnitude.quantile(0.95)) if len(magnitude) else 0.0,
        "false_prune_magnitude_p99": float(magnitude.quantile(0.99)) if len(magnitude) else 0.0,
        "false_prune_magnitude_max": max_magnitude,
        "severe_tail_flag": bool(max_magnitude > max(median_true_distance, 1e-12)),
        "oracle_positive_margin_count": int((data["oracle_margin"] > 0.0).sum()),
        "oracle_nonpositive_margin_count": int((data["oracle_margin"] <= 0.0).sum()),
        "near_positive_margin_count": int(((data["oracle_margin"] > 0.0) & data["near_margin"]).sum()),
        "near_negative_margin_count": int(((data["oracle_margin"] <= 0.0) & data["near_margin"]).sum()),
        "decision_sha256": stable_decision_sha256(data, lb_column, decision_column),
    }
    if nominal_alpha is not None and status == "supported":
        record_ci = bound_core.clopper_pearson(false_count, n_records)
        query_ci = bound_core.clopper_pearson(exposed_queries, n_queries)
        primary_lower = record_ci[0] if calibration_level == "record" else query_ci[0]
        row.update({
            "record_false_prune_ci95_lower": record_ci[0],
            "record_false_prune_ci95_upper": record_ci[1],
            "query_false_prune_ci95_lower": query_ci[0],
            "query_false_prune_ci95_upper": query_ci[1],
            "nominal_statistically_compatible": bool(primary_lower <= float(nominal_alpha)),
        })
    return row


def unsupported_row(calibrator: Mapping[str, object], calibrator_sha256: str) -> dict[str, object]:
    return {
        "operating_point_id": calibrator["calibrator_id"],
        "strategy": "probabilistic_ratio_lb",
        "strategy_role": "formal_candidate",
        "formal_candidate": True,
        "calibration_level": calibrator["calibration_level"],
        "nominal_alpha": calibrator["nominal_alpha"],
        "status": calibrator["status"],
        "quantile": None,
        "calibrator_sha256": calibrator_sha256,
        "record_count": None,
        "query_count": None,
    }


def summarize_per_query(
    data: pd.DataFrame,
    query_ids: Sequence[str],
    query_metrics: pd.DataFrame,
    *,
    operating_point_id: str,
    strategy: str,
    decision_column: str,
) -> pd.DataFrame:
    frame = data.assign(
        _prune=data[decision_column].astype(bool),
        _extra=data[decision_column].astype(bool) & ~data["current_prune"],
        _false=data[decision_column].astype(bool) & ~data["oracle_prune"],
        _safe=data[decision_column].astype(bool) & data["oracle_prune"],
    )
    grouped = frame.groupby("query_id", sort=False).agg(
        sampled_records=("query_id", "size"),
        prune_count=("_prune", "sum"),
        extra_prune_count=("_extra", "sum"),
        false_prune_count=("_false", "sum"),
        safe_prune_count=("_safe", "sum"),
        current_prune_count=("current_prune", "sum"),
        raw_point_prune_count=("raw_point_prune", "sum"),
        oracle_prune_count=("oracle_prune", "sum"),
        ratio_eligible_count=("ratio_eligible", "sum"),
    ).reindex([str(value) for value in query_ids], fill_value=0)
    grouped.index.name = "query_id"
    result = grouped.reset_index()
    result.insert(0, "strategy", strategy)
    result.insert(0, "operating_point_id", operating_point_id)
    result["false_prune_exposed"] = result["false_prune_count"] > 0
    result["extra_prune_exposed"] = result["extra_prune_count"] > 0
    result["coverage_rate"] = result["prune_count"] / result["sampled_records"].replace(0, np.nan)
    result["extra_coverage_rate"] = result["extra_prune_count"] / result["sampled_records"].replace(0, np.nan)
    result["false_prune_event_rate"] = result["false_prune_count"] / result["sampled_records"].replace(0, np.nan)
    metrics = query_metrics.copy()
    metrics["query_id"] = metrics["query_id"].astype(str)
    result = result.merge(metrics, on="query_id", how="left", validate="one_to_one")
    if result["bound_evaluated"].isna().any():
        missing = result.loc[result["bound_evaluated"].isna(), "query_id"].tolist()
        raise ValueError(f"query metrics are missing frozen test queries: {missing[:10]}")
    result["sampled_support_fraction"] = result["sampled_records"] / result["bound_evaluated"].replace(0, np.nan)
    return result


def false_prune_records(
    applied: pd.DataFrame,
    calibrator: Mapping[str, object],
) -> pd.DataFrame:
    bad = applied[applied["simulation_false_prune"]].copy()
    columns = [
        "query_id", "current_node_id", "candidate_id", "split", "threshold",
        "shadow_exact_squared_distance", "oracle_margin", "simulation_lb",
        "simulation_ratio_lb", "simulation_false_prune_magnitude",
        "simulation_ratio_eligible", "simulation_used_current_fallback",
        "ratio_fallback_reason", "current_lb_safe", "deployment_distance_hat",
        RAW_ESTIMATE_COLUMN, "kappa_meta", "relative_abs_margin",
        "current_prune", "raw_point_prune", "oracle_prune",
    ]
    if bad.empty:
        result = pd.DataFrame(columns=columns)
    else:
        result = bad[columns].copy()
    result.insert(0, "nominal_alpha", calibrator["nominal_alpha"])
    result.insert(0, "calibration_level", calibrator["calibration_level"])
    result.insert(0, "operating_point_id", calibrator["calibrator_id"])
    return result


def kappa_precision_rows(applied: pd.DataFrame, calibrator: Mapping[str, object]) -> list[dict[str, object]]:
    labels = pd.cut(
        applied["kappa_meta"],
        bins=KAPPA_BINS,
        labels=KAPPA_LABELS,
        right=False,
    ).astype("string").fillna("invalid")
    rows: list[dict[str, object]] = []
    for label in list(KAPPA_LABELS) + ["invalid"]:
        subset = applied[labels == label]
        if subset.empty:
            continue
        decisions = subset["simulation_prune"].astype(bool)
        false = subset["simulation_false_prune"].astype(bool)
        rows.append({
            "operating_point_id": calibrator["calibrator_id"],
            "calibration_level": calibrator["calibration_level"],
            "nominal_alpha": calibrator["nominal_alpha"],
            "kappa_bin": label,
            "record_count": int(len(subset)),
            "ratio_eligible_count": int(subset["simulation_ratio_eligible"].sum()),
            "prune_count": int(decisions.sum()),
            "false_prune_count": int(false.sum()),
            "pruning_precision": safe_ratio(int((decisions & subset["oracle_prune"]).sum()), int(decisions.sum())),
            "false_prune_event_rate": safe_ratio(int(false.sum()), int(len(subset))),
        })
    return rows


def make_margin_plot(applied: pd.DataFrame, operating_point_id: str, output_dir: Path) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    x = applied["oracle_margin"].to_numpy(float)
    y = (applied["simulation_lb"] - applied["threshold"]).to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.hexbin(x, y, gridsize=70, bins="log", mincnt=1, cmap="viridis")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Oracle margin D - threshold")
    ax.set_ylabel("Simulated margin LB - threshold")
    ax.set_title(f"Margin confusion: {operating_point_id}")
    fig.tight_layout()
    fig.savefig(figures / "margin_confusion_plot.png", dpi=180)
    plt.close(fig)


def read_auxiliary_inputs(
    input_path: Path,
    metadata_path: Path | None,
    query_metrics_path: Path | None,
) -> tuple[Path, dict[str, object], Path, pd.DataFrame]:
    metadata_path = (metadata_path or input_path.parent / "metadata.json").resolve()
    query_metrics_path = (query_metrics_path or input_path.parent / "query_metrics.csv").resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if not query_metrics_path.is_file():
        raise FileNotFoundError(query_metrics_path)
    metadata = bound_core.read_json(metadata_path)
    query_metrics = pd.read_csv(query_metrics_path)
    required = {"query_id", "bound_evaluated", "baseline_recall_at_k", "baseline_latency_ns"}
    missing = sorted(required - set(query_metrics.columns))
    if missing:
        raise ValueError("query metrics are missing columns: " + ", ".join(missing))
    query_metrics["query_id"] = query_metrics["query_id"].astype(str)
    if query_metrics["query_id"].duplicated().any():
        raise ValueError("query metrics contain duplicate query_id values")
    return metadata_path, metadata, query_metrics_path, query_metrics


def validate_phase2_application(
    phase2_dir: Path,
    contract: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    application_path = phase2_dir / "test_application_manifest.json"
    application = bound_core.read_json(application_path)
    if application.get("status") != "PASS" or application.get("decision") != "GO_TO_PHASE3":
        raise ValueError("Phase 2 is not PASS / GO_TO_PHASE3")
    if application.get("input_sha256") != contract.get("input_sha256"):
        raise ValueError("Phase-2 application/input SHA mismatch")
    if application.get("query_split_manifest_sha256") != contract.get("query_split_manifest_sha256"):
        raise ValueError("Phase-2 application/query split SHA mismatch")
    calibration_path = phase2_dir / "calibration_manifest.json"
    if estimator_core.sha256_file(calibration_path) != application.get("calibration_manifest_sha256"):
        raise ValueError("Phase-2 calibration manifest SHA mismatch")
    calibration_manifest, calibrators = phase2_apply.load_frozen_calibrators(phase2_dir, contract)
    return application, calibration_manifest, calibrators


def calibrator_sha_lookup(calibration_manifest: Mapping[str, object]) -> dict[str, str]:
    artifacts = calibration_manifest.get("calibrators")
    if not isinstance(artifacts, list):
        raise ValueError("calibration manifest has no calibrator artifacts")
    result: dict[str, str] = {}
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise ValueError("invalid calibrator artifact entry")
        result[str(item["calibrator_id"])] = str(item["sha256"])
    return result


def simulate(
    input_path: Path,
    phase1_dir: Path,
    phase2_dir: Path,
    output_dir: Path,
    *,
    metadata_path: Path | None = None,
    query_metrics_path: Path | None = None,
    sensitivity_kappa: Sequence[float] | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_path.resolve()
    phase1_dir = phase1_dir.resolve()
    phase2_dir = phase2_dir.resolve()
    contract = bound_core.load_phase1_contract(input_path, phase1_dir)
    phase2_application, calibration_manifest, calibrators = validate_phase2_application(phase2_dir, contract)
    data, schema_audit = bound_core.load_derived_input(contract, required_split="test")
    data = normalize_pruning_inputs(data)

    split = contract["query_split_manifest"]
    query_ids_obj = split.get("query_ids") if isinstance(split, Mapping) else None
    if not isinstance(query_ids_obj, Mapping):
        raise ValueError("query split manifest has no query_ids")
    calibration_ids = {str(value) for value in query_ids_obj["calibration"]}
    test_query_ids = [str(value) for value in query_ids_obj["test"]]
    test_ids = set(test_query_ids)
    if calibration_ids & test_ids:
        raise ValueError("calibration/test query leakage")
    if set(data["query_id"].astype(str)) != test_ids:
        raise ValueError("loaded records do not cover the exact frozen test-query set")
    if set(str(value) for value in phase2_application.get("test_query_ids", [])) != test_ids:
        raise ValueError("Phase-2 application test-query set mismatch")
    for calibrator in calibrators:
        frozen_ids = {str(value) for value in calibrator.get("calibration_query_ids", [])}
        if frozen_ids != calibration_ids or frozen_ids & test_ids:
            raise ValueError(f"calibrator query isolation failure: {calibrator.get('calibrator_id')}")

    metadata_path, metadata, query_metrics_path, query_metrics = read_auxiliary_inputs(
        input_path, metadata_path, query_metrics_path
    )
    run_ids = sorted(set(data["run_id"].astype(str))) if "run_id" in data.columns else []
    if len(run_ids) != 1 or str(metadata.get("run_id")) != run_ids[0]:
        raise ValueError("run metadata/run_id mismatch")
    test_query_metrics = query_metrics[query_metrics["query_id"].isin(test_ids)].copy()
    if set(test_query_metrics["query_id"]) != test_ids:
        raise ValueError("query metrics do not cover the exact frozen test-query set")

    calibrator_hashes = calibrator_sha_lookup(calibration_manifest)
    summary_rows: list[dict[str, object]] = []
    per_query_frames: list[pd.DataFrame] = []
    false_frames: list[pd.DataFrame] = []
    kappa_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    applied_by_id: dict[str, pd.DataFrame] = {}

    baseline_specs = [
        ("current_lb", "current_safe_lb", "baseline", "current_lb_safe", "current_prune", False),
        ("raw_point_estimate", "raw_point_estimate", "diagnostic_upper_bound", RAW_ESTIMATE_COLUMN, "raw_point_prune", False),
        ("oracle", "oracle_exact_distance", "diagnostic_ceiling", "shadow_exact_squared_distance", "oracle_prune", False),
    ]
    if "cap_lb" in data.columns:
        cap = pd.to_numeric(data["cap_lb"], errors="coerce")
        if np.isfinite(cap).any():
            data["cap_lb_numeric"] = cap.where(np.isfinite(cap), 0.0).clip(lower=0.0)
            data["cap_prune"] = data["cap_lb_numeric"] > data["threshold"]
            baseline_specs.insert(1, ("cap_lb", "cap_lb", "baseline", "cap_lb_numeric", "cap_prune", False))

    for op_id, strategy, role, lb_column, decision_column, formal in baseline_specs:
        summary_rows.append(summarize_decisions(
            data,
            operating_point_id=op_id,
            strategy=strategy,
            strategy_role=role,
            lb_column=lb_column,
            decision_column=decision_column,
            formal_candidate=formal,
        ))
        per_query_frames.append(summarize_per_query(
            data,
            test_query_ids,
            test_query_metrics,
            operating_point_id=op_id,
            strategy=strategy,
            decision_column=decision_column,
        ))

    for calibrator in calibrators:
        calibrator_id = str(calibrator["calibrator_id"])
        calibrator_sha = calibrator_hashes[calibrator_id]
        if calibrator.get("status") != "supported":
            summary_rows.append(unsupported_row(calibrator, calibrator_sha))
            continue
        quantile = calibrator.get("quantile")
        if quantile is None:
            raise ValueError(f"supported calibrator has no quantile: {calibrator_id}")
        applied = apply_frozen_quantile(data, float(quantile))
        applied_by_id[calibrator_id] = applied
        summary_rows.append(summarize_decisions(
            applied,
            operating_point_id=calibrator_id,
            strategy="probabilistic_ratio_lb",
            strategy_role="formal_candidate",
            lb_column="simulation_lb",
            decision_column="simulation_prune",
            formal_candidate=True,
            calibration_level=str(calibrator["calibration_level"]),
            nominal_alpha=float(calibrator["nominal_alpha"]),
            quantile=float(quantile),
            calibrator_sha256=calibrator_sha,
        ))
        per_query_frames.append(summarize_per_query(
            applied,
            test_query_ids,
            test_query_metrics,
            operating_point_id=calibrator_id,
            strategy="probabilistic_ratio_lb",
            decision_column="simulation_prune",
        ))
        false_frames.append(false_prune_records(applied, calibrator))
        kappa_rows.extend(kappa_precision_rows(applied, calibrator))

    formal_kappa = float(contract["kappa_min"])
    sensitivity_values = list(sensitivity_kappa or (formal_kappa - 0.1, formal_kappa + 0.1))
    sensitivity_values = sorted({float(value) for value in sensitivity_values if math.isfinite(float(value)) and float(value) > 0.0 and float(value) != formal_kappa})
    for calibrator in calibrators:
        if calibrator.get("status") != "supported":
            continue
        for kappa_value in sensitivity_values:
            applied = apply_frozen_quantile(data, float(calibrator["quantile"]), kappa_min=kappa_value)
            row = summarize_decisions(
                applied,
                operating_point_id=f"{calibrator['calibrator_id']}_kappa_{kappa_value:g}",
                strategy="probabilistic_ratio_lb_kappa_sensitivity",
                strategy_role="sensitivity_only",
                lb_column="simulation_lb",
                decision_column="simulation_prune",
                formal_candidate=False,
                calibration_level=str(calibrator["calibration_level"]),
                nominal_alpha=float(calibrator["nominal_alpha"]),
                quantile=float(calibrator["quantile"]),
                calibrator_sha256=calibrator_hashes[str(calibrator["calibrator_id"])],
            )
            row["sensitivity_kappa_min"] = kappa_value
            row["frozen_calibrator_kappa_min"] = formal_kappa
            row["valid_for_formal_selection"] = False
            sensitivity_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    per_query = pd.concat(per_query_frames, ignore_index=True)
    for op_id in summary.loc[summary["status"].eq("supported"), "operating_point_id"]:
        values = per_query.loc[per_query["operating_point_id"].eq(op_id), "prune_count"].astype(int).sort_values(ascending=False)
        total = int(values.sum())
        top_count = max(1, int(math.ceil(len(values) * 0.10))) if len(values) else 0
        share = safe_ratio(int(values.head(top_count).sum()), total)
        exposed_fraction = safe_ratio(int((values > 0).sum()), int(len(values)))
        summary.loc[summary["operating_point_id"].eq(op_id), "top_10_percent_query_savings_share"] = share
        summary.loc[summary["operating_point_id"].eq(op_id), "queries_with_savings_fraction"] = exposed_fraction

    summary = summary.sort_values(
        ["strategy_role", "calibration_level", "nominal_alpha", "operating_point_id"],
        ascending=[True, True, False, True],
        na_position="last",
        kind="mergesort",
    )
    per_query = per_query.sort_values(["operating_point_id", "query_id"], kind="mergesort")
    false_records = pd.concat(false_frames, ignore_index=True) if false_frames else pd.DataFrame()
    if not false_records.empty:
        false_records = false_records.sort_values(
            ["operating_point_id", "simulation_false_prune_magnitude", "query_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
    top_risk = per_query[
        per_query["strategy"].eq("probabilistic_ratio_lb")
        & per_query["false_prune_exposed"]
    ].copy()
    top_risk = top_risk.sort_values(
        ["false_prune_count", "false_prune_event_rate", "extra_prune_count"],
        ascending=[False, False, False],
        kind="mergesort",
    ).head(100)

    summary.to_csv(output_dir / "operating_point_summary.csv", index=False)
    per_query.to_csv(output_dir / "per_query_summary.csv", index=False)
    false_records.to_csv(output_dir / "false_prune_records.csv", index=False)
    top_risk.to_csv(output_dir / "top_risk_queries.csv", index=False)
    pd.DataFrame(kappa_rows).to_csv(output_dir / "kappa_pruning_summary.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(output_dir / "kappa_sensitivity_summary.csv", index=False)

    representative = sorted(
        (
            (float(calibrator["nominal_alpha"]), str(calibrator["calibrator_id"]))
            for calibrator in calibrators
            if calibrator.get("status") == "supported" and calibrator.get("calibration_level") == "query"
        ),
        key=lambda item: (item[0], item[1]),
    )
    if not representative:
        representative = sorted(
            (
                (float(calibrator["nominal_alpha"]), str(calibrator["calibrator_id"]))
                for calibrator in calibrators if calibrator.get("status") == "supported"
            ),
            key=lambda item: (item[0], item[1]),
        )
    if not representative:
        raise ValueError("Phase 2 has no supported calibrator")
    margin_plot_id = representative[0][1]
    make_margin_plot(applied_by_id[margin_plot_id], margin_plot_id, output_dir)

    raw_total_bound = int(pd.to_numeric(query_metrics["bound_evaluated"], errors="raise").sum())
    test_total_bound = int(pd.to_numeric(test_query_metrics["bound_evaluated"], errors="raise").sum())
    sample_modulus = int(metadata.get("shadow_sample_modulus", 0))
    sample_remainder = int(metadata.get("shadow_sample_remainder", -1))
    if sample_modulus <= 0 or not 0 <= sample_remainder < sample_modulus:
        raise ValueError("run metadata has invalid shadow sampling parameters")
    if int(metadata.get("query_count", 0)) < 1000:
        formal_representative = False
        representativeness_reason = "query_count_below_1000"
    else:
        formal_representative = True
        representativeness_reason = "q1000_or_greater_with_frozen_edge_sampling"

    manifest: dict[str, object] = {
        "format": SIMULATION_MANIFEST_FORMAT,
        "format_version": 1,
        "phase3_version": PHASE3_VERSION,
        "status": "SIMULATED",
        "decision": "PENDING_SWEEP_SUMMARY",
        "decision_reasons": [],
        "conclusion_scope": "candidate_record_level_observe_only_potential_savings",
        "real_search_recall_claim_allowed": False,
        "input_path": str(input_path),
        "input_sha256": contract["input_sha256"],
        "phase1_dir": str(phase1_dir),
        "phase2_dir": str(phase2_dir),
        "query_split_manifest_sha256": contract["query_split_manifest_sha256"],
        "phase1_run_manifest_sha256": contract["phase1_run_manifest_sha256"],
        "phase1_summary_sha256": contract["phase1_summary_sha256"],
        "phase2_calibration_manifest_sha256": estimator_core.sha256_file(phase2_dir / "calibration_manifest.json"),
        "phase2_application_manifest_sha256": estimator_core.sha256_file(phase2_dir / "test_application_manifest.json"),
        "phase2_coverage_summary_sha256": estimator_core.sha256_file(phase2_dir / "test_coverage_summary.csv"),
        "estimator_formula_version": contract["estimator_formula_version"],
        "estimator_variant": estimator_core.DEPLOYMENT_ESTIMATOR,
        "clipping_policy": estimator_core.DEPLOYMENT_CLIPPING,
        "kappa_min": formal_kappa,
        "fallback_policy": estimator_core.FALLBACK_POLICY,
        "threshold_comparison": "strict_greater_than",
        "raw_point_estimate_formal_operating_point": False,
        "raw_point_estimate_definition": "Phase-1 distance_hat_raw_clipped",
        "input_raw_would_prune_definition": "upstream C++ operational approximate_squared_distance > threshold",
        "input_flags_used_for_formal_probabilistic_decisions": False,
        "calibration_test_disjoint": True,
        "test_quantile_refit": False,
        "test_query_count": len(test_query_ids),
        "test_record_count": int(len(data)),
        "ratio_eligible_count": int(data["ratio_eligible"].sum()),
        "ratio_eligible_fraction": float(data["ratio_eligible"].mean()),
        "fallback_count": int((~data["ratio_eligible"]).sum()),
        "invalid_fail_closed_count": int((~data["analysis_valid"]).sum()),
        "record_support_sha256": estimator_core.sha256_text(
            data[["query_id", "current_node_id", "candidate_id"]]
            .astype(str)
            .sort_values(["query_id", "current_node_id", "candidate_id"], kind="mergesort")
            .to_csv(index=False, lineterminator="\n")
        ),
        "input_flag_mismatches": {
            "current_would_prune": int((data["current_prune"] != data["input_current_would_prune"]).sum()),
            "raw_would_prune": int((data["raw_point_prune"] != data["input_raw_would_prune"]).sum()),
            "oracle_would_prune": int((data["oracle_prune"] != data["input_oracle_would_prune"]).sum()),
        },
        "schema_audit": schema_audit,
        "run_metadata": {
            "path": str(metadata_path),
            "sha256": estimator_core.sha256_file(metadata_path),
            "run_id": metadata.get("run_id"),
            "dataset": metadata.get("dataset"),
            "mode": metadata.get("mode"),
            "query_count": metadata.get("query_count"),
            "ef_search": metadata.get("ef_search"),
            "index_sha256": metadata.get("index_sha256"),
            "sidecar_sha256": metadata.get("sidecar_sha256"),
            "real_pruning_enabled": metadata.get("real_pruning_enabled"),
            "bound_pruned_semantics": metadata.get("bound_pruned_semantics"),
        },
        "query_metrics": {
            "path": str(query_metrics_path),
            "sha256": estimator_core.sha256_file(query_metrics_path),
            "all_query_bound_evaluated": raw_total_bound,
            "test_query_bound_evaluated": test_total_bound,
            "test_sampled_support_fraction": safe_ratio(int(len(data)), test_total_bound),
        },
        "sampling": {
            "hash_id": SAMPLING_HASH_ID,
            "formula": "mix64(query_id) XOR mix64(current_node_id + 0x632be59bd9b4e019) XOR mix64(candidate_id + 0x8cb92baa3f3d8dd7) modulo modulus",
            "modulus": sample_modulus,
            "remainder": sample_remainder,
            "selection_mechanism": "deterministic sampled candidate records on the unchanged observe-only HNSW trajectory",
            "does_not_represent_all_bound_evaluations": sample_modulus != 1,
        },
        "representativeness": {
            "formal": formal_representative,
            "reason": representativeness_reason,
            "same_record_support_for_all_strategies": True,
            "diagnostic_invalid_and_fallback_in_denominator": True,
            "cap_baseline_available": any(spec[0] == "cap_lb" for spec in baseline_specs),
        },
        "sensitivity_kappa_min": sensitivity_values,
        "sensitivity_used_for_selection": False,
        "near_margin_relative_threshold": NEAR_MARGIN_RELATIVE_THRESHOLD,
        "margin_plot_operating_point": margin_plot_id,
        "multiple_selection_bias": {
            "present": True,
            "reason": "up to three Phase-4 candidates are selected after comparing held-out Phase-3 results",
            "required_mitigation": "reserve fresh final evaluation queries for Phase 4",
        },
        "cost_model": {
            "status": "not_evaluated",
            "reason": "observe-only records do not isolate estimator and exact-distance cycle costs",
            "required_phase4_measurement": "C++ estimator parity plus end-to-end latency and distance-count benchmark",
        },
        "producer": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "simulation_script_sha256": estimator_core.sha256_file(Path(__file__).resolve()),
            "bound_core_sha256": estimator_core.sha256_file(Path(bound_core.__file__).resolve()),
            "estimator_core_sha256": estimator_core.sha256_file(Path(estimator_core.__file__).resolve()),
        },
        "artifacts": {
            "operating_point_summary": "operating_point_summary.csv",
            "per_query_summary": "per_query_summary.csv",
            "false_prune_records": "false_prune_records.csv",
            "top_risk_queries": "top_risk_queries.csv",
            "kappa_pruning_summary": "kappa_pruning_summary.csv",
            "kappa_sensitivity_summary": "kappa_sensitivity_summary.csv",
        },
    }
    bound_core.write_json(output_dir / "simulation_manifest.json", manifest)

    import summarize_probability_sweep as sweep

    final_manifest = sweep.summarize(output_dir)
    print(json.dumps(estimator_core.json_safe(final_manifest), indent=2, sort_keys=True, allow_nan=False))
    return final_manifest


def parse_float_list(value: str) -> tuple[float, ...]:
    if not value.strip():
        return ()
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("sensitivity kappa values must be comma-separated floats") from exc
    if any(not math.isfinite(item) or item <= 0.0 for item in parsed):
        raise ValueError("sensitivity kappa values must be positive and finite")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--phase1-dir", required=True, type=Path)
    parser.add_argument("--phase2-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--query-metrics", type=Path)
    parser.add_argument("--sensitivity-kappa", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = simulate(
            args.input,
            args.phase1_dir,
            args.phase2_dir,
            args.output_dir,
            metadata_path=args.metadata,
            query_metrics_path=args.query_metrics,
            sensitivity_kappa=parse_float_list(args.sensitivity_kappa) or None,
        )
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0 if manifest["decision"] == "GO_TO_PHASE4" else 3


if __name__ == "__main__":
    raise SystemExit(main())
