#!/usr/bin/env python3
"""Summarize one completed real-data trace experiment across efSearch values."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hnsw-summary-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    "queries",
    "dco_records",
    "exact_dco_count",
    "mean_recall_at_k",
    "mean_recall_at_k_ci_low",
    "mean_recall_at_k_ci_high",
    "mean_n_dist",
    "mean_n_dist_ci_low",
    "mean_n_dist_ci_high",
    "median_n_dist",
    "p95_n_dist",
    "p99_n_dist",
    "rejected_at_evaluation_count",
    "rejected_at_evaluation_ratio",
    "sampled_fully_state_neutral_count",
    "sampled_fully_state_neutral_ratio",
    "mean_per_query_sampled_fully_state_neutral_ratio",
    "mean_per_query_sampled_fully_state_neutral_ratio_ci_low",
    "mean_per_query_sampled_fully_state_neutral_ratio_ci_high",
    "sampled_threshold_valid_count",
    "sampled_threshold_valid_ratio",
    "sampled_inserted_candidate_count",
    "sampled_inserted_candidate_ratio",
    "sampled_threshold_changed_count",
    "sampled_threshold_changed_ratio",
    "sampled_expanded_later_count",
    "sampled_final_topk_count",
    "valid_margin_count",
    "valid_neutral_count",
    "valid_neutral_margin_gt_1pct_count",
    "valid_neutral_margin_gt_5pct_count",
    "valid_neutral_margin_gt_10pct_count",
    "valid_neutral_margin_gt_1pct",
    "valid_neutral_margin_gt_5pct",
    "valid_neutral_margin_gt_10pct",
    "edge_direction_sample_count",
    "edge_direction_dimension",
    "edge_pca_components",
    "edge_pca_cumulative_variance",
]

INTEGER_COLUMNS = {
    "queries", "dco_records", "exact_dco_count", "rejected_at_evaluation_count",
    "sampled_fully_state_neutral_count", "valid_margin_count", "valid_neutral_count",
    "valid_neutral_margin_gt_1pct_count", "valid_neutral_margin_gt_5pct_count",
    "valid_neutral_margin_gt_10pct_count", "edge_direction_sample_count",
    "edge_direction_dimension", "edge_pca_components",
    "sampled_threshold_valid_count", "sampled_inserted_candidate_count",
    "sampled_threshold_changed_count", "sampled_expanded_later_count", "sampled_final_topk_count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Required file is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_text(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"Required provenance file is missing: {path}")
    return path.read_text(encoding="utf-8").strip()


def analysis_provenance() -> tuple[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(repo_root), "status", "--short"], text=True
    ).strip()
    return commit, status


def finite_number(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not numeric: {value!r}") from exc
    if not np.isfinite(result):
        raise RuntimeError(f"{label} is not finite: {value!r}")
    return result


def summarize_performance(path: Path, expected_repeats: int) -> dict[str, float | int]:
    if not path.is_file():
        raise RuntimeError(f"Required file is missing: {path}")
    frame = pd.read_csv(path)
    required = {"repeat", "query_count", "total_latency_ns", "qps", "mean_recall_at_k"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{path} is missing columns: {', '.join(missing)}")
    if len(frame) != expected_repeats:
        raise RuntimeError(f"{path} has {len(frame)} repeats; expected {expected_repeats}")
    if frame.empty or (frame["query_count"] <= 0).any():
        raise RuntimeError(f"{path} contains no valid benchmark repeats")
    if frame["query_count"].nunique() != 1:
        raise RuntimeError(f"{path} uses inconsistent query counts across repeats")
    return {
        "performance_repeats": len(frame),
        "performance_query_count": int(frame["query_count"].iloc[0]),
        "latency_ms_mean": float(frame["total_latency_ns"].mean() / 1e6),
        "latency_ms_std": float(frame["total_latency_ns"].std(ddof=1) / 1e6) if len(frame) > 1 else 0.0,
        "qps_mean": float(frame["qps"].mean()),
        "qps_std": float(frame["qps"].std(ddof=1)) if len(frame) > 1 else 0.0,
        "performance_recall_mean": float(frame["mean_recall_at_k"].mean()),
    }


def save_line_plot(frame: pd.DataFrame, columns: list[tuple[str, str]], ylabel: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for column, label in columns:
        ax.plot(frame["ef_search"], frame[column], marker="o", label=label)
    ax.set(xlabel="efSearch", ylabel=ylabel)
    ax.set_xticks(frame["ef_search"])
    ax.grid(alpha=0.3)
    if len(columns) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    config_path = args.experiment_config or run_root / "config" / "experiment.json"
    config = load_json(config_path)
    dataset_config_path = run_root / "config" / "dataset.json"
    dataset_config = load_json(dataset_config_path)
    experiment_commit = load_text(run_root / "config" / "git_commit.txt")
    experiment_git_status = load_text(run_root / "config" / "git_status.txt")
    analysis_commit, analysis_git_status = analysis_provenance()
    ef_values = [int(value) for value in config["ef_search_values"]]
    if not ef_values or len(ef_values) != len(set(ef_values)):
        raise RuntimeError("ef_search_values must be non-empty and unique")
    expected_repeats = int(config["performance_repeats"])
    expected_queries = int(config["query_count"])

    rows: list[dict] = []
    dataset: str | None = None
    for ef_search in ef_values:
        metrics_path = run_root / f"ef{ef_search}" / "analysis" / "metrics.json"
        metrics = load_json(metrics_path)
        if dataset is None:
            dataset = str(metrics["dataset"])
        elif str(metrics["dataset"]) != dataset:
            raise RuntimeError(f"Dataset mismatch in {metrics_path}")
        if int(metrics["queries"]) != expected_queries:
            raise RuntimeError(
                f"{metrics_path} reports {metrics['queries']} queries; expected {expected_queries}"
            )
        row: dict[str, float | int | str] = {"dataset": dataset, "ef_search": ef_search}
        for column in METRIC_COLUMNS:
            if column not in metrics:
                raise RuntimeError(f"{metrics_path} is missing metric: {column}")
            value = finite_number(metrics[column], f"{metrics_path}:{column}")
            row[column] = int(value) if column in INTEGER_COLUMNS else value
        performance_path = run_root / "performance" / f"ef{ef_search}" / "performance.csv"
        row.update(summarize_performance(performance_path, expected_repeats))
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("ef_search").reset_index(drop=True)
    output_dir = (args.output_dir or run_root / "summary").resolve()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    records = json.loads(summary.to_json(orient="records"))
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_root.name,
                "run_root": str(run_root),
                "experiment_config": str(config_path.resolve()),
                "dataset_config": str(dataset_config_path.resolve()),
                "dataset": dataset,
                "dimension": int(dataset_config["dimension"]),
                "n_base": int(dataset_config["n_base"]),
                "n_query": int(dataset_config["n_query"]),
                "distance_kind": dataset_config["distance_kind"],
                "dataset_checksums": dataset_config.get("checksums", {}),
                "experiment_commit": experiment_commit,
                "experiment_git_status": experiment_git_status,
                "analysis_commit": analysis_commit,
                "analysis_git_status": analysis_git_status,
                "metric_schema_version": 2,
                "sampling": {
                    "dco_sample_modulus": int(config["dco_sample_modulus"]),
                    "trace_shards": int(config["trace_shards"]),
                    "edge_samples_per_shard": int(config["edge_samples_per_shard"]),
                },
                "results": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    save_line_plot(summary, [("mean_recall_at_k", "trace"), ("performance_recall_mean", "trace-disabled")],
                   "Mean Recall@K", plots_dir / "recall_vs_ef_search.png")
    save_line_plot(summary, [("mean_n_dist", "N_dist")], "Mean distance computations",
                   plots_dir / "n_dist_vs_ef_search.png")
    save_line_plot(summary, [("rejected_at_evaluation_ratio", "rejected at evaluation"),
                             ("sampled_fully_state_neutral_ratio", "fully state-neutral")],
                   "DCO ratio", plots_dir / "neutral_ratio_vs_ef_search.png")
    save_line_plot(summary, [("valid_neutral_margin_gt_1pct", ">1%"),
                             ("valid_neutral_margin_gt_5pct", ">5%"),
                             ("valid_neutral_margin_gt_10pct", ">10%")],
                   "Fraction of valid fully-neutral DCOs", plots_dir / "neutral_margin_vs_ef_search.png")
    save_line_plot(summary, [("qps_mean", "trace-disabled")], "Queries per second",
                   plots_dir / "qps_vs_ef_search.png")

    columns = [
        "ef_search", "mean_recall_at_k", "mean_n_dist", "rejected_at_evaluation_ratio",
        "sampled_fully_state_neutral_ratio", "valid_neutral_margin_gt_5pct",
        "edge_pca_cumulative_variance", "qps_mean", "latency_ms_mean",
    ]
    display = summary[columns].copy()
    for column in columns[1:]:
        display[column] = display[column].map(lambda value: f"{value:.6g}")
    report = [
        "# Cross-efSearch Real-Dataset Experiment Summary",
        "",
        f"- Dataset: {dataset}",
        f"- Run ID: {run_root.name}",
        f"- Dimension: {dataset_config['dimension']}",
        f"- Experiment commit: `{experiment_commit}`",
        f"- Experiment working tree clean: {'yes' if not experiment_git_status else 'no'}",
        f"- Queries per trace configuration: {expected_queries}",
        f"- Trace-disabled performance repeats: {expected_repeats}",
        f"- efSearch values: {', '.join(map(str, sorted(ef_values)))}",
        "",
        "## Core results",
        "",
        markdown_table(display),
        "",
        "## Interpretation boundary",
        "",
        "The rejected-at-evaluation ratio is an exact all-DCO measure. The fully-state-neutral ratio "
        "is computed from deterministic sampled DCO rows. Margin percentages use only sampled DCOs "
        "that are fully state-neutral and had a valid threshold before evaluation. "
        "These trace statistics characterize distance-computation and state-neutrality distributions. "
        "The performance columns come from the separately built trace-disabled binary. They establish "
        "the unquantized search baseline, but do not by themselves establish quantization speedup or "
        "break-even points; those require lookup/decode and table-construction measurements.",
        "",
        "## Generated artifacts",
        "",
        "- `summary.csv`: analysis-ready table",
        "- `summary.json`: machine-readable summary with run provenance",
        "- `plots/`: cross-efSearch comparison plots",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"cross_ef_summary_ok output_dir={output_dir}")


if __name__ == "__main__":
    main()
