#!/usr/bin/env python3
"""Summarize a Phase-3 probability sweep and freeze Phase-4 candidates."""

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
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hnsw-ratio-phase3-sweep-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import probabilistic_bound_core as bound_core
import ratio_estimator_core as estimator_core


MIN_POTENTIAL_SAVINGS_RATE = 0.05
MIN_EXTRA_COVERAGE_RATE = 0.01
MIN_ORACLE_GAP_RECOVERY_RATE = 0.05
MIN_RATIO_ELIGIBLE_FRACTION = 0.99
MAX_TOP_10_PERCENT_QUERY_SHARE = 0.50
MIN_QUERIES_WITH_SAVINGS_FRACTION = 0.10


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype("string").str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False}).fillna(False).astype(bool)


def candidate_frame(summary: pd.DataFrame) -> pd.DataFrame:
    formal = bool_series(summary["formal_candidate"])
    supported = summary["status"].eq("supported")
    candidates = summary[formal & supported].copy()
    for name in (
        "nominal_alpha", "coverage_rate", "potential_exact_distance_savings_rate",
        "extra_coverage_rate", "oracle_gap_recovery_rate", "false_prune_event_rate",
        "false_prune_query_exposure_rate", "top_10_percent_query_savings_share",
        "queries_with_savings_fraction",
    ):
        candidates[name] = pd.to_numeric(candidates[name], errors="coerce")
    candidates["risk_compatible"] = bool_series(candidates["nominal_statistically_compatible"])
    candidates["tail_acceptable"] = ~bool_series(candidates["severe_tail_flag"])
    candidates["trusted"] = candidates["risk_compatible"] & candidates["tail_acceptable"]
    candidates["beneficial"] = (
        (candidates["potential_exact_distance_savings_rate"] >= MIN_POTENTIAL_SAVINGS_RATE)
        | (candidates["oracle_gap_recovery_rate"] >= MIN_ORACLE_GAP_RECOVERY_RATE)
    )
    candidates["distributed"] = (
        (candidates["top_10_percent_query_savings_share"] <= MAX_TOP_10_PERCENT_QUERY_SHARE)
        & (candidates["queries_with_savings_fraction"] >= MIN_QUERIES_WITH_SAVINGS_FRACTION)
    )
    return candidates


def row_to_operating_point(row: pd.Series, role: str) -> dict[str, object]:
    return {
        "operating_point_id": str(row["operating_point_id"]),
        "roles": [role],
        "alpha": float(row["nominal_alpha"]),
        "calibration_level": str(row["calibration_level"]),
        "quantile": float(row["quantile"]),
        "calibrator_sha256": str(row["calibrator_sha256"]),
        "strategy": str(row["strategy"]),
        "potential_exact_distance_savings_rate": float(row["potential_exact_distance_savings_rate"]),
        "extra_coverage_rate": float(row["extra_coverage_rate"]),
        "oracle_recovery_rate": float(row["oracle_recovery_rate"]),
        "oracle_gap_recovery_rate": None if pd.isna(row["oracle_gap_recovery_rate"]) else float(row["oracle_gap_recovery_rate"]),
        "false_prune_event_rate": float(row["false_prune_event_rate"]),
        "false_prune_query_exposure_rate": float(row["false_prune_query_exposure_rate"]),
        "top_10_percent_query_savings_share": float(row["top_10_percent_query_savings_share"]),
        "queries_with_savings_fraction": float(row["queries_with_savings_fraction"]),
        "risk_compatible": bool(row["risk_compatible"]),
        "severe_tail_flag": not bool(row["tail_acceptable"]),
        "trusted": bool(row["trusted"]),
        "diagnostic_only": role == "aggressive" and not bool(row["trusted"]),
    }


def select_operating_points(candidates: pd.DataFrame) -> list[dict[str, object]]:
    selected: dict[str, dict[str, object]] = {}

    trusted_query = candidates[
        candidates["trusted"] & candidates["calibration_level"].eq("query")
    ].sort_values(
        ["nominal_alpha", "potential_exact_distance_savings_rate", "operating_point_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    if not trusted_query.empty:
        row = trusted_query.iloc[0]
        selected[str(row["operating_point_id"])] = row_to_operating_point(row, "conservative")

    balanced = candidates[
        candidates["trusted"] & candidates["beneficial"] & candidates["distributed"]
    ].sort_values(
        ["extra_coverage_rate", "false_prune_query_exposure_rate", "nominal_alpha", "operating_point_id"],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    if not balanced.empty:
        row = balanced.iloc[0]
        op_id = str(row["operating_point_id"])
        if op_id in selected:
            selected[op_id]["roles"].append("balanced")  # type: ignore[index]
        else:
            selected[op_id] = row_to_operating_point(row, "balanced")

    aggressive = candidates.sort_values(
        ["potential_exact_distance_savings_rate", "extra_coverage_rate", "false_prune_query_exposure_rate", "operating_point_id"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    if not aggressive.empty:
        row = aggressive.iloc[0]
        op_id = str(row["operating_point_id"])
        if op_id in selected:
            selected[op_id]["roles"].append("aggressive")  # type: ignore[index]
        else:
            selected[op_id] = row_to_operating_point(row, "aggressive")

    return list(selected.values())[:3]


def decide(
    candidates: pd.DataFrame,
    selected: Sequence[Mapping[str, object]],
    manifest: Mapping[str, object],
) -> tuple[str, list[str], list[str]]:
    reasons: list[str] = []
    caveats: list[str] = []
    representativeness = manifest.get("representativeness")
    if not isinstance(representativeness, Mapping) or not bool(representativeness.get("formal")):
        reasons.append("formal_q1000_representativeness_requirement_not_met")
    if float(manifest.get("ratio_eligible_fraction", 0.0)) < MIN_RATIO_ELIGIBLE_FRACTION:
        reasons.append("ratio_eligible_fraction_below_gate")

    trusted = candidates[candidates["trusted"]]
    if trusted.empty:
        reasons.append("no_supported_risk_compatible_tail_acceptable_calibrator")
    else:
        if (trusted["extra_coverage_rate"] < MIN_EXTRA_COVERAGE_RATE).all():
            reasons.append("all_trusted_working_points_extra_coverage_below_one_percent")
        beneficial = trusted[trusted["beneficial"]]
        if beneficial.empty:
            reasons.append("no_trusted_working_point_has_five_percent_savings_or_oracle_gap_recovery")
        elif not beneficial["distributed"].any():
            reasons.append("all_beneficial_working_points_are_query_concentrated")

    selected_roles = {role for point in selected for role in point.get("roles", [])}
    if not ({"conservative", "balanced"} & selected_roles):
        reasons.append("no_conservative_or_balanced_operating_point_for_phase4")
    if not any(point.get("calibration_level") == "query" and point.get("extra_coverage_rate", 0.0) > 0.0 for point in selected):
        caveats.append("no_selected_query_level_point_has_extra_coverage")
    cost_model = manifest.get("cost_model")
    if isinstance(cost_model, Mapping) and cost_model.get("status") != "evaluated":
        caveats.append("estimator_cost_gate_deferred_to_phase4_benchmark")
    caveats.append("phase3_selection_used_held_out_test_results_reserve_fresh_phase4_queries")
    return ("GO_TO_PHASE4" if not reasons else "NO_GO_OFFLINE_PRUNING"), reasons, caveats


def make_sweep_figures(
    summary: pd.DataFrame,
    per_query: pd.DataFrame,
    kappa: pd.DataFrame,
    output_dir: Path,
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    candidates = candidate_frame(summary)
    baselines = summary[summary["strategy_role"].eq("baseline") | summary["strategy_role"].eq("diagnostic_ceiling")]

    def line_plot(metric: str, filename: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        for level, group in candidates.groupby("calibration_level", sort=True):
            group = group.sort_values("nominal_alpha")
            ax.semilogx(group["nominal_alpha"], group[metric], marker="o", label=str(level))
        if metric == "coverage_rate":
            for baseline_id in ("current_lb", "oracle"):
                row = baselines[baselines["operating_point_id"].eq(baseline_id)]
                if not row.empty:
                    ax.axhline(float(row.iloc[0][metric]), linestyle="--", label=baseline_id)
        ax.set_xlabel("Nominal alpha")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=180)
        plt.close(fig)

    line_plot("coverage_rate", "coverage_vs_risk.png", "Potential prune coverage")
    line_plot("oracle_recovery_rate", "oracle_recovery_vs_risk.png", "Oracle recovery")

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for level, group in candidates.groupby("calibration_level", sort=True):
        group = group.sort_values("nominal_alpha")
        metric = "false_prune_event_rate" if level == "record" else "false_prune_query_exposure_rate"
        ax.loglog(group["nominal_alpha"], group[metric].clip(lower=1e-12), marker="o", label=f"{level} primary risk")
    low = float(candidates["nominal_alpha"].min()) if not candidates.empty else 1e-5
    high = float(candidates["nominal_alpha"].max()) if not candidates.empty else 1e-1
    ax.plot([low, high], [low, high], linestyle="--", color="black", label="nominal")
    ax.set_xlabel("Nominal alpha")
    ax.set_ylabel("Empirical false-prune risk")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "false_prune_vs_risk.png", dpi=180)
    plt.close(fig)

    formal_ids = candidates["operating_point_id"].astype(str).tolist()
    query_data = per_query[per_query["operating_point_id"].astype(str).isin(formal_ids)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    if not query_data.empty:
        labels = formal_ids
        extra_values = [
            query_data.loc[query_data["operating_point_id"].eq(op_id), "extra_coverage_rate"].dropna().to_numpy(float)
            for op_id in labels
        ]
        false_values = [
            query_data.loc[query_data["operating_point_id"].eq(op_id), "false_prune_event_rate"].dropna().to_numpy(float)
            for op_id in labels
        ]
        axes[0].boxplot(extra_values, tick_labels=labels, showfliers=False)
        axes[1].boxplot(false_values, tick_labels=labels, showfliers=False)
        for ax in axes:
            ax.tick_params(axis="x", rotation=35)
    axes[0].set_title("Per-query extra prune distribution")
    axes[0].set_ylabel("Extra coverage")
    axes[1].set_title("Per-query false-prune distribution")
    axes[1].set_ylabel("False-prune event rate")
    fig.tight_layout()
    fig.savefig(figures / "per_query_exposure.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    if not kappa.empty:
        pivot = kappa.pivot(index="kappa_bin", columns="operating_point_id", values="pruning_precision")
        pivot = pivot.reindex([label for label in ("<0.5", "[0.5,0.6)", "[0.6,0.7)", "[0.7,0.8)", "[0.8,0.9)", "[0.9,1.0)", ">=1.0", "invalid") if label in pivot.index])
        for name in pivot.columns:
            ax.plot(range(len(pivot.index)), pivot[name], marker="o", label=str(name))
        ax.set_xticks(range(len(pivot.index)), pivot.index, rotation=30)
    ax.set_xlabel("kappa_meta bin")
    ax.set_ylabel("Pruning precision")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "kappa_pruning_precision.png", dpi=180)
    plt.close(fig)


def write_report(
    output_dir: Path,
    manifest: Mapping[str, object],
    summary: pd.DataFrame,
    selected: Sequence[Mapping[str, object]],
) -> None:
    candidates = candidate_frame(summary)
    sampling = manifest.get("sampling") if isinstance(manifest.get("sampling"), Mapping) else {}
    query_metrics = manifest.get("query_metrics") if isinstance(manifest.get("query_metrics"), Mapping) else {}
    flag_mismatches = manifest.get("input_flag_mismatches") if isinstance(manifest.get("input_flag_mismatches"), Mapping) else {}
    lines = [
        "# Phase 3 offline probabilistic-pruning simulation",
        "",
        f"- Decision: **{manifest['decision']}**",
        f"- Reasons: `{', '.join(manifest['decision_reasons']) if manifest['decision_reasons'] else 'none'}`",
        f"- Caveats: `{', '.join(manifest['decision_caveats'])}`",
        f"- Test records / queries: `{manifest['test_record_count']}` / `{manifest['test_query_count']}`",
        f"- Frozen kappa_min: `{manifest['kappa_min']}`",
        f"- Ratio-eligible fraction: `{float(manifest['ratio_eligible_fraction']):.8%}`",
        f"- Edge sampling: modulus `{sampling.get('modulus')}`, remainder `{sampling.get('remainder')}`; test support `{float(query_metrics.get('test_sampled_support_fraction', math.nan)):.4%}` of observed bound evaluations",
        f"- Input flag mismatches (current/raw/oracle): `{flag_mismatches.get('current_would_prune')}` / `{flag_mismatches.get('raw_would_prune')}` / `{flag_mismatches.get('oracle_would_prune')}`",
        "",
        "The Phase-3 raw diagnostic is the frozen Phase-1 `distance_hat_raw_clipped` formula. The exported `raw_would_prune` flag uses the older C++ operational approximate-distance diagnostic; their mismatches are recorded and neither is a formal operating point.",
        "",
        "## Frozen formal working points",
        "",
        "| ID | level | alpha | coverage | extra | oracle recovery | false event | query exposure | top-10% query share | trusted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in candidates.sort_values(["calibration_level", "nominal_alpha"], ascending=[True, False]).iterrows():
        lines.append(
            f"| `{row['operating_point_id']}` | {row['calibration_level']} | {row['nominal_alpha']:.1e} "
            f"| {row['coverage_rate']:.4%} | {row['extra_coverage_rate']:.4%} "
            f"| {row['oracle_recovery_rate']:.4%} | {row['false_prune_event_rate']:.4%} "
            f"| {row['false_prune_query_exposure_rate']:.4%} "
            f"| {row['top_10_percent_query_savings_share']:.4%} | {bool(row['trusted'])} |"
        )
    lines.extend(["", "## Selected Phase-4 candidates", ""])
    if selected:
        lines.extend([
            "| ID | roles | level | alpha | potential savings | extra coverage | diagnostic only |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for point in selected:
            lines.append(
                f"| `{point['operating_point_id']}` | {', '.join(point['roles'])} "
                f"| {point['calibration_level']} | {float(point['alpha']):.1e} "
                f"| {float(point['potential_exact_distance_savings_rate']):.4%} "
                f"| {float(point['extra_coverage_rate']):.4%} | {bool(point['diagnostic_only'])} |"
            )
    else:
        lines.append("No Phase-4 operating point was selected.")
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "The estimator has potential candidate-record pruning benefit on the frozen observe-only trajectory.",
        "This does not show that HNSW can be safely accelerated at a target recall: real pruning changes subsequent traversal.",
        "The selected points were compared on held-out Phase-3 data, so Phase 4 must reserve fresh final-evaluation queries.",
        "The input is a deterministic edge sample rather than every bound evaluation; results must not be extrapolated without Phase-4 validation.",
        "Estimator compute cost is not isolated here and remains an explicit Phase-4 benchmark gate.",
        "",
    ])
    (output_dir / "PHASE3_OFFLINE_PRUNING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def summarize(output_dir: Path) -> dict[str, object]:
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "simulation_manifest.json"
    manifest = bound_core.read_json(manifest_path)
    if manifest.get("format") != "v0_ratio_offline_pruning_simulation":
        raise ValueError("unsupported Phase-3 simulation manifest")
    summary = pd.read_csv(output_dir / "operating_point_summary.csv")
    per_query = pd.read_csv(output_dir / "per_query_summary.csv", dtype={"query_id": "string"})
    kappa = pd.read_csv(output_dir / "kappa_pruning_summary.csv")
    candidates = candidate_frame(summary)
    selected = select_operating_points(candidates)
    for point in selected:
        point.update({
            "kappa_min": manifest.get("kappa_min"),
            "clipping_policy": manifest.get("clipping_policy"),
            "fallback_policy": manifest.get("fallback_policy"),
            "estimator_formula_version": manifest.get("estimator_formula_version"),
        })
    decision, reasons, caveats = decide(candidates, selected, manifest)

    selected_payload = {
        "format": "v0_ratio_phase4_selected_operating_points",
        "format_version": 1,
        "phase3_version": manifest.get("phase3_version"),
        "decision": decision,
        "selection_count": len(selected),
        "input_sha256": manifest.get("input_sha256"),
        "query_split_manifest_sha256": manifest.get("query_split_manifest_sha256"),
        "phase2_calibration_manifest_sha256": manifest.get("phase2_calibration_manifest_sha256"),
        "phase2_application_manifest_sha256": manifest.get("phase2_application_manifest_sha256"),
        "estimator_formula_version": manifest.get("estimator_formula_version"),
        "kappa_min": manifest.get("kappa_min"),
        "clipping_policy": manifest.get("clipping_policy"),
        "fallback_policy": manifest.get("fallback_policy"),
        "test_quantile_refit": False,
        "sensitivity_used_for_selection": False,
        "multiple_selection_bias": manifest.get("multiple_selection_bias"),
        "requires_fresh_phase4_evaluation_queries": True,
        "operating_points": selected,
    }
    bound_core.write_json(output_dir / "selected_operating_points.json", selected_payload)

    baseline_ids = {"current_lb", "cap_lb", "raw_point_estimate", "oracle"}
    potential_rows = summary[
        summary["operating_point_id"].astype(str).isin(baseline_ids)
        | (bool_series(summary["formal_candidate"]) & summary["status"].eq("supported"))
    ]
    potential = {
        "format": "v0_ratio_phase3_potential_savings_summary",
        "format_version": 1,
        "definition": "counterfactual prune count on unchanged observe-only sampled records",
        "is_realized_speedup": False,
        "record_count": manifest.get("test_record_count"),
        "query_count": manifest.get("test_query_count"),
        "strategies": estimator_core.json_safe(potential_rows.to_dict(orient="records")),
        "selected_operating_point_ids": [point["operating_point_id"] for point in selected],
    }
    bound_core.write_json(output_dir / "potential_savings_summary.json", potential)

    make_sweep_figures(summary, per_query, kappa, output_dir)
    manifest.update({
        "status": "PASS" if decision == "GO_TO_PHASE4" else "FAIL",
        "decision": decision,
        "decision_reasons": reasons,
        "decision_caveats": caveats,
        "gate_thresholds": {
            "minimum_potential_savings_rate": MIN_POTENTIAL_SAVINGS_RATE,
            "minimum_extra_coverage_rate": MIN_EXTRA_COVERAGE_RATE,
            "minimum_oracle_gap_recovery_rate": MIN_ORACLE_GAP_RECOVERY_RATE,
            "minimum_ratio_eligible_fraction": MIN_RATIO_ELIGIBLE_FRACTION,
            "maximum_top_10_percent_query_savings_share": MAX_TOP_10_PERCENT_QUERY_SHARE,
            "minimum_queries_with_savings_fraction": MIN_QUERIES_WITH_SAVINGS_FRACTION,
        },
        "selected_operating_point_count": len(selected),
        "selected_operating_point_ids": [point["operating_point_id"] for point in selected],
        "summarizer_sha256": estimator_core.sha256_file(Path(__file__).resolve()),
    })
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, dict):
        artifacts.update({
            "potential_savings_summary": "potential_savings_summary.json",
            "selected_operating_points": "selected_operating_points.json",
            "report": "PHASE3_OFFLINE_PRUNING_REPORT.md",
        })
    bound_core.write_json(manifest_path, manifest)
    write_report(output_dir, manifest, summary, selected)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = summarize(args.phase3_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(estimator_core.json_safe(manifest), indent=2, sort_keys=True, allow_nan=False))
    return 0 if manifest["decision"] == "GO_TO_PHASE4" else 3


if __name__ == "__main__":
    raise SystemExit(main())
