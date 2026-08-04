#!/usr/bin/env python3
"""Aggregate Phase-4 HNSW shadow/real-pruning runs and enforce gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONSISTENCY_KEYS = (
    "dataset", "dataset_config_sha256", "query_dataset_sha256",
    "ground_truth_sha256", "index_sha256", "sidecar_sha256", "dimension",
    "n_base", "k",
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_runs(explicit: Iterable[Path], root: Path | None) -> list[Path]:
    found = {path.resolve() for path in explicit}
    if root:
        found.update(path.parent.resolve() for path in root.rglob("complete.json"))
    return sorted(found)


def validate_fresh_queries(run_frames: list[pd.DataFrame], manifest_path: Path) -> dict:
    manifest = load_json(manifest_path)
    fresh = {int(value) for value in manifest.get("query_ids", [])}
    phase3 = {int(value) for value in manifest.get("phase3_query_ids", [])}
    if not fresh or not phase3:
        raise ValueError(
            "fresh-query manifest must contain non-empty query_ids and phase3_query_ids")
    overlap = sorted(fresh.intersection(phase3))
    if overlap:
        raise ValueError(f"fresh queries overlap Phase-3 queries: {overlap[:10]}")
    for frame in run_frames:
        observed = {int(value) for value in frame["query_id"]}
        if observed != fresh:
            raise ValueError("run query IDs do not exactly match fresh-query manifest")
    return {
        "path": str(manifest_path.resolve()),
        "sha256": sha256(manifest_path),
        "fresh_query_count": len(fresh),
        "phase3_query_count": len(phase3),
        "overlap_count": 0,
    }


def load_runs(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    rows, per_query, frames = [], [], []
    reference = None
    for path in paths:
        required = ["complete.json", "metadata.json", "summary.json", "query_metrics.csv"]
        missing = [name for name in required if not (path / name).is_file()]
        if missing:
            raise ValueError(f"incomplete run {path}: missing {missing}")
        complete = load_json(path / "complete.json")
        metadata = load_json(path / "metadata.json")
        summary = load_json(path / "summary.json")
        if complete.get("status") != "complete" or summary.get("status") != "valid":
            raise ValueError(f"run is not complete/valid: {path}")
        current = {key: metadata.get(key) for key in CONSISTENCY_KEYS}
        if reference is None:
            reference = current
        elif current != reference:
            differing = [key for key in CONSISTENCY_KEYS if current[key] != reference[key]]
            raise ValueError(f"run provenance mismatch {path}: {differing}")

        query = pd.read_csv(path / "query_metrics.csv")
        if len(query) != int(summary["query_count"]):
            raise ValueError(f"query count mismatch: {path}")
        frames.append(query)
        mode = metadata["mode"]
        point = metadata.get("ratio_calibrator_id", "current")
        repetition = metadata.get("repetition")
        if repetition is None:
            repetition = metadata.get("run_id", path.name)
        baseline_ns = float(summary["baseline_latency_ns"])
        v0_ns = float(summary["v0_latency_ns"])
        query_count = int(summary["query_count"])
        candidate_count = int(summary.get("candidate_expansions", 0))
        visited_count = int(summary.get("visited_nodes", 0))
        saved = int(summary.get("ratio_exact_distance_saved", 0))
        row = {
            "run_dir": str(path), "run_id": metadata.get("run_id", path.name),
            "mode": mode, "operating_point_id": point,
            "ef_search": int(metadata["ef_search"]), "repetition": str(repetition),
            "query_count": query_count,
            "baseline_recall_at_k": float(summary["mean_baseline_recall_at_k"]),
            "v0_recall_at_k": float(summary["mean_v0_recall_at_k"]),
            "baseline_recall_at_1": float(summary.get("mean_baseline_recall_at_1", math.nan)),
            "v0_recall_at_1": float(summary.get("mean_v0_recall_at_1", math.nan)),
            "recall_loss_pp": float(summary["recall_at_k_loss_percentage_points"]),
            "worst_v0_recall_at_k": float(summary["worst_v0_recall_at_k"]),
            "mismatch_query_fraction": float(summary["mismatch_query_fraction"]),
            "baseline_qps": query_count * 1e9 / baseline_ns if baseline_ns else math.nan,
            "v0_qps": query_count * 1e9 / v0_ns if v0_ns else math.nan,
            "qps_gain_percent": 100.0 * (baseline_ns / v0_ns - 1.0) if v0_ns else math.nan,
            "baseline_latency_p50_ms": float(query["baseline_latency_ns"].quantile(.50)) / 1e6,
            "baseline_latency_p95_ms": float(query["baseline_latency_ns"].quantile(.95)) / 1e6,
            "baseline_latency_p99_ms": float(query["baseline_latency_ns"].quantile(.99)) / 1e6,
            "v0_latency_p50_ms": float(query["v0_latency_ns"].quantile(.50)) / 1e6,
            "v0_latency_p95_ms": float(query["v0_latency_ns"].quantile(.95)) / 1e6,
            "v0_latency_p99_ms": float(query["v0_latency_ns"].quantile(.99)) / 1e6,
            "peak_rss_bytes": int(summary.get("peak_rss_bytes", 0)),
            "sidecar_bytes_per_edge": float(metadata.get("sidecar_bytes_per_edge", math.nan)),
            "current_lb_time_ns": int(summary.get("current_lb_time_ns", 0)),
            "estimator_time_ns": int(summary.get("estimator_time_ns", 0)),
            "exact_distance_time_ns": int(summary.get("exact_distance_time_ns", 0)),
            "ratio_bound_evaluated": int(summary.get("ratio_bound_evaluated", 0)),
            "ratio_eligible": int(summary.get("ratio_eligible", 0)),
            "ratio_bound_pruned": int(summary.get("ratio_bound_pruned", 0)),
            "ratio_exact_distance_saved": saved,
            "visited_nodes": visited_count,
            "candidate_expansions": candidate_count,
            "exact_savings_percent": 100.0 * saved / visited_count if visited_count else 0.0,
            "ratio_interval_violation": int(summary.get("ratio_interval_violation", 0)),
            "ratio_false_prune": int(summary.get("ratio_false_prune", 0)),
            "ratio_false_prune_query_exposure": int(summary.get("ratio_false_prune_query_exposure", 0)),
            "lost_ground_truth_neighbors": int(summary.get("lost_ground_truth_neighbors", 0)),
            "ratio_current_lb_evaluated": int(summary.get("ratio_current_lb_evaluated", 0)),
            "ratio_current_lb_skipped_eligible": int(summary.get("ratio_current_lb_skipped_eligible", 0)),
            "ratio_current_lb_valid": int(summary.get("ratio_current_lb_valid", 0)),
            "ratio_current_lb_invalid": int(summary.get("ratio_current_lb_invalid", 0)),
            "ratio_current_lb_fallback": int(summary.get("ratio_current_lb_fallback", 0)),
            "ratio_invalid_fallback": int(summary.get("ratio_invalid_fallback", 0)),
            "ratio_trusted": bool(metadata.get("ratio_trusted", False)),
            "ratio_diagnostic_only": bool(metadata.get("ratio_diagnostic_only", False)),
            "ratio_calibrator_sha256": metadata.get("ratio_calibrator_sha256", ""),
        }
        rows.append(row)
        query = query.copy()
        query["run_id"] = row["run_id"]
        query["mode"] = mode
        query["operating_point_id"] = point
        query["ef_search"] = row["ef_search"]
        per_query.append(query)
    return pd.DataFrame(rows), pd.concat(per_query, ignore_index=True), frames


def matrix_diagnostics(runs: pd.DataFrame, expected_repetitions: int,
                       expected_ef: list[int], expected_points: list[str]) -> list[str]:
    reasons = []
    for ef in expected_ef:
        count = runs[(runs["mode"] == "correctness") &
                     (runs["ef_search"] == ef)]["repetition"].nunique()
        if count < expected_repetitions:
            reasons.append(f"current ef={ef}: {count}/{expected_repetitions} repetitions")
        shadow = runs[(runs["mode"] == "ratio-shadow") &
                      (runs["ef_search"] == ef)]["repetition"].nunique()
        if shadow < expected_repetitions:
            reasons.append(f"ratio-shadow ef={ef}: {shadow}/{expected_repetitions} repetitions")
        for point in expected_points:
            count = runs[(runs["mode"] == "ratio-prune") &
                         (runs["ef_search"] == ef) &
                         (runs["operating_point_id"] == point)]["repetition"].nunique()
            if count < expected_repetitions:
                reasons.append(f"{point} ef={ef}: {count}/{expected_repetitions} repetitions")
    return reasons


def aggregate(runs: pd.DataFrame, recall_budget_pp: float,
              min_savings_percent: float, min_qps_gain_percent: float) -> pd.DataFrame:
    fields = ["mode", "operating_point_id", "ef_search"]
    numeric = [column for column in runs.columns if column not in {
        *fields, "run_dir", "run_id", "repetition", "ratio_calibrator_sha256"} and
        pd.api.types.is_numeric_dtype(runs[column])]
    result = runs.groupby(fields, dropna=False)[numeric].mean().reset_index()
    counts = runs.groupby(fields, dropna=False).size().rename("repetitions").reset_index()
    result = result.merge(counts, on=fields)
    dispersion = runs.groupby(fields, dropna=False).agg(
        v0_qps_median=("v0_qps", "median"),
        v0_qps_std=("v0_qps", "std"),
        recall_loss_pp_median=("recall_loss_pp", "median"),
        exact_savings_percent_median=("exact_savings_percent", "median"),
    ).reset_index()
    result = result.merge(dispersion, on=fields)
    result["recall_gate_pass"] = result["recall_loss_pp"] <= recall_budget_pp
    result["benefit_gate_pass"] = (
        (result["exact_savings_percent"] >= min_savings_percent) |
        (result["qps_gain_percent"] >= min_qps_gain_percent))
    result["scientific_gate_pass"] = (
        result["recall_gate_pass"] & result["benefit_gate_pass"] &
        (result["ratio_diagnostic_only"] < 0.5))
    return result


def make_figures(summary: pd.DataFrame, per_query: pd.DataFrame, output: Path) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    real = summary[summary["mode"] == "ratio-prune"]
    for x, y, name, xlabel, ylabel in [
        ("qps_gain_percent", "recall_loss_pp", "recall_vs_qps.png", "QPS gain (%)", "Recall@K loss (pp)"),
        ("exact_savings_percent", "recall_loss_pp", "recall_loss_vs_distance_saved.png", "Exact-distance savings (%)", "Recall@K loss (pp)"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for point, group in real.groupby("operating_point_id"):
            ax.plot(group[x], group[y], marker="o", label=point)
            for _, row in group.iterrows():
                ax.annotate(f"ef={int(row.ef_search)}", (row[x], row[y]), fontsize=7)
        ax.axhline(0.1, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=.25)
        if len(real): ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(figures / name, dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for point, group in real.groupby("operating_point_id"):
        ax.plot(group.ef_search, group.v0_qps, marker="o", label=point)
    ax.set_xlabel("efSearch"); ax.set_ylabel("QPS"); ax.grid(alpha=.25)
    if len(real): ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figures / "qps_vs_ef.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    latency = per_query[per_query["mode"] == "ratio-prune"]
    for point, group in latency.groupby("operating_point_id"):
        ax.hist(group["v0_latency_ns"] / 1e6, bins=40, alpha=.35, label=point)
    ax.set_xlabel("Per-query latency (ms)"); ax.set_ylabel("Count"); ax.grid(alpha=.2)
    if len(latency): ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figures / "latency_distribution.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if {"baseline_recall_at_k", "v0_recall_at_k"}.issubset(per_query.columns):
        loss = 100.0 * (per_query["baseline_recall_at_k"] - per_query["v0_recall_at_k"])
        ax.hist(loss, bins=40)
    ax.set_xlabel("Per-query Recall@K loss (percentage points)")
    ax.set_ylabel("Count"); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(figures / "per_query_recall_loss.png", dpi=160); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", default=[], type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--fresh-query-manifest", type=Path)
    parser.add_argument("--expected-repetitions", type=int, default=5)
    parser.add_argument("--expected-ef-search", default="100,200,400")
    parser.add_argument("--expected-operating-points", default="query_alpha_1e-2,query_alpha_1e-1,record_alpha_1e-1")
    parser.add_argument("--recall-loss-budget-pp", type=float, default=0.1)
    parser.add_argument("--min-savings-percent", type=float, default=5.0)
    parser.add_argument("--min-qps-gain-percent", type=float, default=5.0)
    args = parser.parse_args()

    paths = discover_runs(args.run_dir, args.runs_root)
    if not paths:
        raise ValueError("no complete run directories found")
    runs, per_query, frames = load_runs(paths)
    fresh = None
    if args.fresh_query_manifest:
        fresh = validate_fresh_queries(frames, args.fresh_query_manifest)
    elif args.mode == "formal":
        raise ValueError("formal mode requires --fresh-query-manifest")

    expected_ef = [int(value) for value in args.expected_ef_search.split(",") if value]
    expected_points = [value for value in args.expected_operating_points.split(",") if value]
    incomplete = matrix_diagnostics(runs, args.expected_repetitions, expected_ef, expected_points)
    summary = aggregate(runs, args.recall_loss_budget_pp,
                        args.min_savings_percent, args.min_qps_gain_percent)
    real = summary[summary["mode"] == "ratio-prune"]
    trusted_benefit = real[(real.ratio_trusted >= 0.5) & real.scientific_gate_pass]
    trusted_qps = trusted_benefit[
        trusted_benefit.qps_gain_percent >= args.min_qps_gain_percent]
    trusted_savings_only = trusted_benefit[
        (trusted_benefit.exact_savings_percent >= args.min_savings_percent) &
        (trusted_benefit.qps_gain_percent < args.min_qps_gain_percent)]
    if args.mode == "formal" and not incomplete and not trusted_qps.empty:
        decision = "GO_TO_PHASE5"
    elif args.mode == "formal" and not incomplete and not trusted_savings_only.empty:
        decision = "CONDITIONAL_GO_MICROBENCHMARK"
    elif not trusted_benefit.empty:
        decision = "CONDITIONAL_GO_MORE_RUNS"
    else:
        decision = "NO_GO_PHASE4"
    reasons = incomplete.copy()
    if trusted_benefit.empty:
        reasons.append("no trusted real-pruning configuration passed recall and benefit gates")
    elif decision == "CONDITIONAL_GO_MICROBENCHMARK":
        reasons.append("exact-distance work decreased but measured QPS gain is below threshold")

    out = args.output_dir
    for directory in ("manifests", "shadow", "real_pruning", "performance", "tables"):
        (out / directory).mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "tables" / "run_metrics.csv", index=False)
    summary.to_csv(out / "tables" / "recall_performance_frontier.csv", index=False)
    risk_columns = [column for column in (
        "run_id", "mode", "operating_point_id", "ef_search", "query_id",
        "baseline_recall_at_k", "v0_recall_at_k", "lost_ground_truth_neighbors",
        "ratio_false_prune", "ratio_bound_pruned", "ratio_exact_distance_saved")
        if column in per_query.columns]
    per_query[risk_columns].to_csv(out / "tables" / "per_query_quality.csv", index=False)
    work_columns = [column for column in summary.columns if column in {
        "mode", "operating_point_id", "ef_search", "repetitions",
        "ratio_bound_evaluated", "ratio_eligible", "ratio_bound_pruned",
        "ratio_exact_distance_saved", "visited_nodes", "candidate_expansions",
        "ratio_current_lb_evaluated", "ratio_current_lb_skipped_eligible",
        "ratio_current_lb_valid", "ratio_current_lb_invalid",
        "ratio_current_lb_fallback", "ratio_invalid_fallback",
        "current_lb_time_ns", "estimator_time_ns", "exact_distance_time_ns",
        "exact_savings_percent"}]
    summary[work_columns].to_csv(out / "tables" / "work_counters.csv", index=False)
    runs[runs["mode"] == "ratio-shadow"].to_csv(out / "shadow" / "shadow_run_summary.csv", index=False)
    runs[runs["mode"] == "ratio-prune"].to_csv(out / "real_pruning" / "real_run_summary.csv", index=False)
    latency_columns = [column for column in runs.columns if "latency" in column or column in {
        "run_id", "mode", "operating_point_id", "ef_search", "v0_qps", "baseline_qps", "peak_rss_bytes"}]
    runs[latency_columns].to_csv(out / "performance" / "latency_summary.csv", index=False)
    inventory = {"run_directories": [str(path) for path in paths], "fresh_query_manifest": fresh}
    (out / "manifests" / "run_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_figures(summary, per_query, out)

    result = {
        "status": "PASS" if decision != "NO_GO_PHASE4" else "FAIL",
        "decision": decision, "mode": args.mode, "run_count": len(runs),
        "matrix_complete": not incomplete, "decision_reasons": reasons,
        "recall_loss_budget_pp": args.recall_loss_budget_pp,
        "min_savings_percent": args.min_savings_percent,
        "min_qps_gain_percent": args.min_qps_gain_percent,
        "passing_configurations": trusted_benefit[["operating_point_id", "ef_search"]].to_dict("records"),
        "fresh_query_manifest": fresh,
    }
    (out / "decision.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# Phase 4 HNSW shadow and real-pruning report", "",
        f"Decision: **{decision}**", "",
        f"Runs: {len(runs)}; formal matrix complete: {not incomplete}.", "",
        "The recall gate is evaluated against the baseline result from the same run. "
        "The benefit gate requires either exact-distance savings or QPS improvement.", "",
        "## Decision reasons", "",
    ] + ([f"- {reason}" for reason in reasons] or ["- All requested gates passed."])
    (out / "PHASE4_HNSW_PROBABILISTIC_PRUNING_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8")
    decision_report = [
        "# Final Phase 4 Go/No-Go decision", "", f"**{decision}**", "",
        "This is an empirical result for the frozen dataset, index, sidecar, "
        "calibration distribution and risk setting. It is not a deterministic "
        "safety theorem.", "",
    ] + ([f"- {reason}" for reason in reasons] or ["- All formal gates passed."])
    (out / "FINAL_GO_NO_GO_DECISION.md").write_text(
        "\n".join(decision_report) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if decision != "NO_GO_PHASE4" else 3


if __name__ == "__main__":
    raise SystemExit(main())
