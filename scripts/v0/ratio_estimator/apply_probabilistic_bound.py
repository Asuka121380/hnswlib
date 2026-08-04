#!/usr/bin/env python3
"""Apply frozen Phase-2 calibrators to held-out test queries."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hnsw-ratio-phase2-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import probabilistic_bound_core as bound_core
import ratio_estimator_core as estimator_core


def load_frozen_calibrators(
    calibration_dir: Path,
    contract: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest_path = calibration_dir / "calibration_manifest.json"
    manifest = bound_core.read_json(manifest_path)
    if manifest.get("format") != bound_core.CALIBRATION_MANIFEST_FORMAT:
        raise ValueError("unsupported calibration manifest")
    if manifest.get("status") != "frozen" or manifest.get("test_queries_read") is not False:
        raise ValueError("calibration manifest was not frozen before test application")
    if manifest.get("input_sha256") != contract.get("input_sha256"):
        raise ValueError("calibration manifest/input SHA mismatch")
    if manifest.get("query_split_manifest_sha256") != contract.get("query_split_manifest_sha256"):
        raise ValueError("calibration manifest/query split SHA mismatch")

    artifacts = manifest.get("calibrators")
    if not isinstance(artifacts, list):
        raise ValueError("calibration manifest has no calibrator list")
    calibrators: list[dict[str, object]] = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ValueError("invalid calibrator artifact entry")
        relative = artifact.get("path")
        if not isinstance(relative, str):
            raise ValueError("calibrator artifact has no path")
        path = (calibration_dir / relative).resolve()
        if calibration_dir.resolve() not in path.parents:
            raise ValueError("calibrator path escapes calibration directory")
        if estimator_core.sha256_file(path) != artifact.get("sha256"):
            raise ValueError(f"calibrator artifact SHA mismatch: {relative}")
        calibrator = bound_core.read_json(path)
        bound_core.validate_calibrator(calibrator, contract)
        if calibrator.get("calibrator_id") != artifact.get("calibrator_id"):
            raise ValueError("calibrator ID mismatch")
        calibrators.append(calibrator)
    return manifest, calibrators


def summarize_application(
    applied: pd.DataFrame,
    calibrator: Mapping[str, object],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    violations = applied[applied["interval_violation"]].copy()
    query_group = applied.groupby("query_id", sort=False)["interval_violation"].any()
    violating_query_ids = query_group[query_group].index.astype(str)
    violating_queries = applied[applied["query_id"].astype(str).isin(set(violating_query_ids))]
    query_violations = (
        violating_queries.groupby("query_id", sort=False)
        .agg(
            violation_records=("interval_violation", "sum"),
            max_violation_magnitude=("violation_magnitude", "max"),
            ratio_eligible_records=("ratio_eligible", "sum"),
            records=("query_id", "size"),
        )
        .reset_index()
    )

    n_records = int(len(applied))
    n_record_violations = int(len(violations))
    n_queries = int(applied["query_id"].nunique())
    n_query_violations = int(len(violating_query_ids))
    record_ci = bound_core.clopper_pearson(n_record_violations, n_records)
    query_ci = bound_core.clopper_pearson(n_query_violations, n_queries)
    alpha = float(calibrator["nominal_alpha"])
    level = str(calibrator["calibration_level"])
    primary_lower = record_ci[0] if level == "record" else query_ci[0]
    compatible = bool(primary_lower <= alpha)
    magnitude = violations["violation_magnitude"]
    median_true_distance = float(applied["shadow_exact_squared_distance"].median())
    max_magnitude = float(magnitude.max()) if len(magnitude) else 0.0
    severe_tail = bool(max_magnitude > max(median_true_distance, 1e-12))
    fallback = applied["used_current_lb_fallback"]
    row: dict[str, object] = {
        "calibrator_id": calibrator["calibrator_id"],
        "calibration_level": level,
        "nominal_alpha": alpha,
        "status": calibrator["status"],
        "sample_count": calibrator["sample_count"],
        "order_statistic_index_one_based": calibrator["order_statistic_index_one_based"],
        "quantile": calibrator["quantile"],
        "test_record_count": n_records,
        "record_violation_count": n_record_violations,
        "record_violation_rate": n_record_violations / n_records if n_records else math.nan,
        "record_ci95_lower": record_ci[0],
        "record_ci95_upper": record_ci[1],
        "test_query_count": n_queries,
        "query_violation_count": n_query_violations,
        "query_violation_rate": n_query_violations / n_queries if n_queries else math.nan,
        "query_ci95_lower": query_ci[0],
        "query_ci95_upper": query_ci[1],
        "nominal_statistically_compatible": compatible,
        "ratio_eligible_count": int(applied["ratio_eligible"].sum()),
        "ratio_eligible_fraction": float(applied["ratio_eligible"].mean()),
        "current_lb_fallback_count": int(fallback.sum()),
        "current_lb_fallback_fraction": float(fallback.mean()),
        "invalid_fail_closed_count": int((~applied["analysis_valid"]).sum()),
        "invalid_fail_closed_fraction": float((~applied["analysis_valid"]).mean()),
        "fallback_violation_count": int((violations["used_current_lb_fallback"]).sum()) if len(violations) else 0,
        "positive_lb_fraction": float((applied["probabilistic_lb"] > 0.0).mean()),
        "stronger_than_current_fraction": float((applied["probabilistic_lb"] > applied["current_lb_safe"]).mean()),
        "violation_magnitude_mean": float(magnitude.mean()) if len(magnitude) else 0.0,
        "violation_magnitude_p50": float(magnitude.quantile(0.50)) if len(magnitude) else 0.0,
        "violation_magnitude_p95": float(magnitude.quantile(0.95)) if len(magnitude) else 0.0,
        "violation_magnitude_p99": float(magnitude.quantile(0.99)) if len(magnitude) else 0.0,
        "violation_magnitude_max": max_magnitude,
        "median_true_distance": median_true_distance,
        "severe_tail_flag": severe_tail,
    }

    if len(violations):
        violation_records = violations[[
            "query_id", "current_node_id", "candidate_id", "split",
            "ratio_eligible", "used_current_lb_fallback", "ratio_fallback_reason",
            "deployment_distance_hat", "probabilistic_lb",
            "shadow_exact_squared_distance", "violation_magnitude",
            "kappa_meta", "threshold", "current_lb_safe",
        ]].copy()
    else:
        violation_records = pd.DataFrame(columns=[
            "query_id", "current_node_id", "candidate_id", "split",
            "ratio_eligible", "used_current_lb_fallback", "ratio_fallback_reason",
            "deployment_distance_hat", "probabilistic_lb",
            "shadow_exact_squared_distance", "violation_magnitude",
            "kappa_meta", "threshold", "current_lb_safe",
        ])
    violation_records.insert(0, "calibrator_id", calibrator["calibrator_id"])
    if query_violations.empty:
        query_violations = pd.DataFrame(columns=[
            "query_id", "violation_records", "max_violation_magnitude",
            "ratio_eligible_records", "records",
        ])
    query_violations.insert(0, "calibrator_id", calibrator["calibrator_id"])
    return row, violation_records, query_violations


def make_figures(summary: pd.DataFrame, violations: pd.DataFrame, output_dir: Path) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    supported = summary[summary["status"] == "supported"].copy()

    def coverage_plot(level: str, metric: str, filename: str, title: str) -> None:
        subset = supported[supported["calibration_level"] == level].sort_values("nominal_alpha")
        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        if not subset.empty:
            ax.loglog(subset["nominal_alpha"], subset[metric], marker="o", label="empirical")
            low = float(subset["nominal_alpha"].min())
            high = float(subset["nominal_alpha"].max())
            ax.plot([low, high], [low, high], linestyle="--", color="black", label="nominal")
        ax.set_xlabel("Nominal alpha")
        ax.set_ylabel("Empirical violation rate")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=180)
        plt.close(fig)

    coverage_plot("record", "record_violation_rate", "nominal_vs_empirical_record_coverage.png", "Record-level held-out coverage")
    coverage_plot("query", "query_violation_rate", "nominal_vs_empirical_query_coverage.png", "Query-level held-out coverage")

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for level, group in supported.groupby("calibration_level", sort=True):
        group = group.sort_values("nominal_alpha")
        ax.semilogx(group["nominal_alpha"], group["quantile"], marker="o", label=level)
    ax.set_xlabel("Nominal alpha")
    ax.set_ylabel("One-sided error quantile")
    ax.set_title("Calibrated quantile versus risk")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "quantile_vs_alpha.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    if not violations.empty:
        ordered_ids = list(dict.fromkeys(violations["calibrator_id"].astype(str)))
        values = [violations.loc[violations["calibrator_id"] == item, "violation_magnitude"].to_numpy(float) for item in ordered_ids]
        ax.boxplot(values, tick_labels=ordered_ids, showfliers=False)
        ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("LB - true distance for violations")
    ax.set_title("Held-out violation magnitude")
    fig.tight_layout()
    fig.savefig(figures / "violation_magnitude.png", dpi=180)
    plt.close(fig)


def decide(summary: pd.DataFrame) -> tuple[str, list[str]]:
    supported = summary[summary["status"] == "supported"]
    reasons: list[str] = []
    record = supported[
        (supported["calibration_level"] == "record")
        & supported["nominal_statistically_compatible"]
        & ~supported["severe_tail_flag"]
    ]
    query = supported[
        (supported["calibration_level"] == "query")
        & supported["nominal_statistically_compatible"]
        & ~supported["severe_tail_flag"]
    ]
    if record.empty:
        reasons.append("no_supported_compatible_record_calibrator")
    if query.empty:
        reasons.append("no_supported_compatible_query_calibrator")
    elif not (query["stronger_than_current_fraction"] > 0.0).any():
        reasons.append("query_calibrator_uninformative_vs_current_lb")
    if int(supported["fallback_violation_count"].sum()) > 0:
        reasons.append("current_lb_fallback_violation")
    return ("GO_TO_PHASE3" if not reasons else "NO_GO_EMPIRICAL_BOUND"), reasons


def write_report(output_dir: Path, manifest: Mapping[str, object], summary: pd.DataFrame) -> None:
    decision = manifest["decision"]
    reasons = manifest["decision_reasons"]
    supported = summary[summary["status"] == "supported"]
    lines = [
        "# Phase 2 empirical probabilistic-bound calibration",
        "",
        f"- Decision: **{decision}**",
        f"- Reasons: `{', '.join(reasons) if reasons else 'none'}`",
        f"- Frozen kappa_min: `{manifest['kappa_min']}`",
        f"- Test records / queries: `{manifest['test_record_count']}` / `{manifest['test_query_count']}`",
        f"- Ratio-eligible fraction: `{manifest['ratio_eligible_fraction']:.8%}`",
        "",
        "## Supported held-out working points",
        "",
        "| ID | level | alpha | record violations | query violations | stronger than current | compatible |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in supported.sort_values(["calibration_level", "nominal_alpha"], ascending=[True, False]).iterrows():
        lines.append(
            f"| `{row['calibrator_id']}` | {row['calibration_level']} | {row['nominal_alpha']:.1e} "
            f"| {int(row['record_violation_count'])}/{int(row['test_record_count'])} "
            f"| {int(row['query_violation_count'])}/{int(row['test_query_count'])} "
            f"| {row['stronger_than_current_fraction']:.4%} "
            f"| {bool(row['nominal_statistically_compatible'])} |"
        )
    lines.extend([
        "",
        "These are empirical held-out guarantees for the frozen sampled search distribution.",
        "They are not deterministic lower bounds and do not predict recall under changed HNSW control flow.",
        "",
    ])
    (output_dir / "PHASE2_CALIBRATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def apply(
    input_path: Path,
    phase1_dir: Path,
    calibration_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = bound_core.load_phase1_contract(input_path, phase1_dir)
    calibration_manifest, calibrators = load_frozen_calibrators(calibration_dir, contract)
    data, _ = bound_core.load_derived_input(contract, required_split="test")
    split = contract["query_split_manifest"]
    query_ids_obj = split["query_ids"]  # type: ignore[index]
    calibration_ids = {str(value) for value in query_ids_obj["calibration"]}  # type: ignore[index]
    test_ids = {str(value) for value in query_ids_obj["test"]}  # type: ignore[index]
    if calibration_ids & test_ids:
        raise ValueError("calibration/test query leakage")
    test = data.copy()
    if test.empty:
        raise ValueError("frozen test split has no records")

    rows: list[dict[str, object]] = []
    violation_frames: list[pd.DataFrame] = []
    query_frames: list[pd.DataFrame] = []
    for calibrator in calibrators:
        frozen_ids = {str(value) for value in calibrator.get("calibration_query_ids", [])}
        if frozen_ids != calibration_ids or frozen_ids & test_ids:
            raise ValueError(f"calibrator query isolation failure: {calibrator.get('calibrator_id')}")
        if calibrator.get("status") != "supported":
            rows.append({
                "calibrator_id": calibrator["calibrator_id"],
                "calibration_level": calibrator["calibration_level"],
                "nominal_alpha": calibrator["nominal_alpha"],
                "status": calibrator["status"],
                "sample_count": calibrator["sample_count"],
                "order_statistic_index_one_based": calibrator["order_statistic_index_one_based"],
                "quantile": None,
                "test_record_count": int(len(test)),
                "test_query_count": int(test["query_id"].nunique()),
            })
            continue
        quantile = calibrator.get("quantile")
        if quantile is None:
            raise ValueError("supported calibrator has no quantile")
        applied = bound_core.apply_calibrator_to_frame(test, float(quantile))
        row, violation_records, violation_queries = summarize_application(applied, calibrator)
        rows.append(row)
        violation_frames.append(violation_records)
        query_frames.append(violation_queries)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "test_coverage_summary.csv", index=False)
    violations = pd.concat(violation_frames, ignore_index=True) if violation_frames else pd.DataFrame()
    violation_queries = pd.concat(query_frames, ignore_index=True) if query_frames else pd.DataFrame()
    violations.to_csv(output_dir / "violation_records.csv", index=False)
    violation_queries.to_csv(output_dir / "violation_queries.csv", index=False)
    make_figures(summary, violations, output_dir)
    decision, reasons = decide(summary)
    manifest: dict[str, object] = {
        "format": "v0_ratio_probabilistic_test_application",
        "format_version": 1,
        "phase2_version": bound_core.PHASE2_VERSION,
        "status": "PASS" if decision == "GO_TO_PHASE3" else "FAIL",
        "decision": decision,
        "decision_reasons": reasons,
        "input_sha256": contract["input_sha256"],
        "query_split_manifest_sha256": contract["query_split_manifest_sha256"],
        "calibration_manifest_sha256": estimator_core.sha256_file(calibration_dir / "calibration_manifest.json"),
        "estimator_formula_version": contract["estimator_formula_version"],
        "estimator_variant": estimator_core.DEPLOYMENT_ESTIMATOR,
        "clipping_policy": estimator_core.DEPLOYMENT_CLIPPING,
        "kappa_min": contract["kappa_min"],
        "fallback_policy": estimator_core.FALLBACK_POLICY,
        "test_query_ids": sorted(test_ids),
        "test_query_count": int(test["query_id"].nunique()),
        "test_record_count": int(len(test)),
        "ratio_eligible_count": int(test["ratio_eligible"].sum()),
        "ratio_eligible_fraction": float(test["ratio_eligible"].mean()),
        "current_lb_fallback_count": int((~test["ratio_eligible"]).sum()),
        "invalid_fail_closed_count": int((~test["analysis_valid"]).sum()),
        "calibration_test_disjoint": True,
        "calibration_manifest_test_queries_read": calibration_manifest.get("test_queries_read"),
        "artifacts": {
            "coverage_summary": "test_coverage_summary.csv",
            "violation_records": "violation_records.csv",
            "violation_queries": "violation_queries.csv",
            "report": "PHASE2_CALIBRATION_REPORT.md",
        },
    }
    bound_core.write_json(output_dir / "test_application_manifest.json", manifest)
    write_report(output_dir, manifest, summary)
    print(json.dumps(estimator_core.json_safe(manifest), indent=2, sort_keys=True, allow_nan=False))
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--phase1-dir", required=True, type=Path)
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        manifest = apply(args.input, args.phase1_dir, args.calibration_dir, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0 if manifest["decision"] == "GO_TO_PHASE3" else 3


if __name__ == "__main__":
    raise SystemExit(main())
