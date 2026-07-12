#!/usr/bin/env python3
"""Validate and analyze a baseline HNSW trace produced by baseline_trace_runner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hnsw-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA


QUERY_REQUIRED = {
    "query_id",
    "n_edge_scan",
    "n_duplicate",
    "n_unique_neighbor",
    "n_dist",
    "n_expanded",
    "n_strict_state_neutral",
    "recall_at_k",
}

DCO_REQUIRED = {
    "query_id",
    "dco_index",
    "dist_qc",
    "dist_qd",
    "threshold_before",
    "threshold_valid_before",
    "relative_margin",
    "edge_length_cd",
    "edge_dot_qcd",
    "edge_cosine_qcd",
    "triangle_lower_bound",
    "is_state_neutral",
    "search_progress_fraction",
    "threshold_stability_indicator",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--exact-distance-ns", type=float, default=420.0)
    parser.add_argument("--lookup-ns", type=float, default=80.0)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def save_cdf(values: np.ndarray, title: str, xlabel: str, path: Path) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"No finite values for {title}")
    values.sort()
    y = np.arange(1, values.size + 1) / values.size
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(values, y)
    ax.set(title=title, xlabel=xlabel, ylabel="CDF")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarize_queries(query: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "n_dist",
        "n_edge_scan",
        "n_duplicate",
        "n_expanded",
        "n_strict_state_neutral",
        "baseline_query_latency_ns",
        "trace_query_latency_ns",
        "exact_distance_time_ns",
        "recall_at_k",
    ]
    rows: list[dict[str, float | str]] = []
    for metric in metrics:
        values = query[metric].astype(float)
        rows.append(
            {
                "metric": metric,
                "mean": values.mean(),
                "min": values.min(),
                "max": values.max(),
                "median": values.median(),
                "p90": values.quantile(0.90),
                "p95": values.quantile(0.95),
                "p99": values.quantile(0.99),
            }
        )
    return pd.DataFrame(rows)


def validate(query: pd.DataFrame, dco: pd.DataFrame, metadata: dict) -> dict[str, float | int | bool]:
    require_columns(query, QUERY_REQUIRED, "query_stats.csv")
    require_columns(dco, DCO_REQUIRED, "dco_trace.csv")

    edge_accounting = query["n_edge_scan"] == query["n_duplicate"] + query["n_unique_neighbor"]
    distance_accounting = query["n_unique_neighbor"] == query["n_dist"]
    if not edge_accounting.all():
        raise AssertionError("N_edge_scan != N_duplicate + N_unique_neighbor")
    if not distance_accounting.all():
        raise AssertionError("N_unique_neighbor != N_dist")

    if int(metadata["dco_sample_modulus"]) == 1:
        records_per_query = dco.groupby("query_id").size().reindex(query["query_id"], fill_value=0).to_numpy()
        if not np.array_equal(records_per_query, query["n_dist"].to_numpy()):
            raise AssertionError("Full DCO record count does not match N_dist")

    valid = dco["threshold_valid_before"].astype(bool)
    margin_error = np.abs(
        dco.loc[valid, "absolute_margin"]
        - (dco.loc[valid, "dist_qd"] - dco.loc[valid, "threshold_before"])
    )
    max_margin_error = float(margin_error.max()) if len(margin_error) else 0.0
    if max_margin_error > 1e-7:
        raise AssertionError(f"Margin consistency failed: {max_margin_error}")

    geometry = dco[dco["geometry_valid"].astype(bool)]
    reconstructed = (
        geometry["dist_qc"]
        + geometry["edge_length_cd"] ** 2
        - 2.0 * geometry["edge_dot_qcd"]
    )
    relative_geometry_error = np.abs(reconstructed - geometry["dist_qd"]) / np.maximum(
        1.0, np.abs(geometry["dist_qd"])
    )
    max_geometry_error = float(relative_geometry_error.max()) if len(relative_geometry_error) else 0.0
    if max_geometry_error > 5e-4:
        raise AssertionError(f"Geometry identity failed: {max_geometry_error}")

    return {
        "query_count": int(len(query)),
        "dco_record_count": int(len(dco)),
        "edge_accounting_ok": True,
        "distance_accounting_ok": True,
        "max_margin_error": max_margin_error,
        "max_relative_geometry_error": max_geometry_error,
    }


def add_candidate_class(dco: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select(
            [
                dco["in_final_topk"].astype(bool),
                dco["expanded_later"].astype(bool),
                dco["inserted_candidate_queue"].astype(bool),
            ],
            ["FINAL_TOPK", "EXPANDED_NOT_FINAL", "INSERTED_NOT_EXPANDED"],
            default="IMMEDIATE_REJECT",
        ),
        index=dco.index,
    )


def make_plots(
    query: pd.DataFrame,
    dco: pd.DataFrame,
    edge: pd.DataFrame,
    plots: Path,
    exact_distance_ns: float,
    lookup_ns: float,
) -> dict[str, float]:
    plots.mkdir(parents=True, exist_ok=True)
    save_cdf(query["n_dist"].to_numpy(), "Exact distance workload", "N_dist per query", plots / "n_dist_cdf.png")

    valid = dco[dco["threshold_valid_before"].astype(bool)].copy()
    save_cdf(valid["relative_margin"].to_numpy(), "Normalized decision margins", "relative margin", plots / "relative_margin_cdf.png")

    bins = np.linspace(0.0, 1.0, 11)
    progress_bin = pd.cut(dco["search_progress_fraction"], bins=bins, include_lowest=True)
    neutral = dco.assign(progress_bin=progress_bin).groupby("progress_bin", observed=True)["is_state_neutral"].mean()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(np.arange(len(neutral)), neutral.to_numpy(), marker="o")
    ax.set(title="State-neutral ratio over search progress", xlabel="progress decile", ylabel="neutral ratio", ylim=(0, 1))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "neutral_over_progress.png", dpi=160)
    plt.close(fig)

    threshold = valid.assign(progress_bin=pd.cut(valid["search_progress_fraction"], bins=bins, include_lowest=True))
    threshold_curve = threshold.groupby("progress_bin", observed=True)["threshold_before"].median()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(np.arange(len(threshold_curve)), threshold_curve.to_numpy(), marker="o")
    ax.set(title="Threshold evolution", xlabel="progress decile", ylabel="median squared-L2 threshold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "threshold_over_progress.png", dpi=160)
    plt.close(fig)

    valid["triangle_prunable"] = valid["triangle_lower_bound"] ** 2 > valid["threshold_before"]
    triangle_rates = valid.groupby(valid["threshold_stability_indicator"].astype(bool))["triangle_prunable"].mean()
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    labels = ["before stable", "stable"]
    values = [float(triangle_rates.get(False, 0.0)), float(triangle_rates.get(True, 0.0))]
    ax.bar(labels, values)
    ax.set(title="Triangle-bound pruning opportunity", ylabel="potential pruning rate", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(plots / "triangle_bound_rate.png", dpi=160)
    plt.close(fig)

    scatter = valid[np.isfinite(valid["edge_cosine_qcd"]) & np.isfinite(valid["relative_margin"])].head(5000)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(scatter["edge_cosine_qcd"], scatter["relative_margin"], s=6, alpha=0.25)
    ax.set(title="Edge cosine versus normalized margin", xlabel="edge cosine", ylabel="relative margin")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots / "edge_cosine_vs_margin.png", dpi=160)
    plt.close(fig)

    t = np.linspace(max(1.0, float(query["n_dist"].min())), float(query["n_dist"].max()) * 1.25, 200)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for lut_us in (10.0, 50.0, 100.0):
        p_min = lookup_ns / exact_distance_ns + (lut_us * 1000.0) / (t * exact_distance_ns)
        ax.plot(t, p_min, label=f"LUT={lut_us:g} us")
    ax.set(title="Break-even pruning ratio", xlabel="N_dist", ylabel="required pruning ratio")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "break_even_vs_n_dist.png", dpi=160)
    plt.close(fig)

    direction_columns = [column for column in edge.columns if column.startswith("dir_")]
    directions = edge[direction_columns].to_numpy(dtype=float)
    components = max(1, min(32, directions.shape[0], directions.shape[1]))
    pca = PCA(n_components=components).fit(directions)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(np.arange(1, components + 1), cumulative, marker="o", markersize=3)
    ax.set(title="Edge-direction PCA", xlabel="components", ylabel="cumulative explained variance", ylim=(0, 1.02))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "edge_direction_pca.png", dpi=160)
    plt.close(fig)

    candidate_counts = dco["candidate_class"].value_counts()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    candidate_counts.plot.bar(ax=ax)
    ax.set(title="Candidate importance classes", xlabel="class", ylabel="DCO count")
    fig.tight_layout()
    fig.savefig(plots / "candidate_importance.png", dpi=160)
    plt.close(fig)

    cosine_margin = scatter[["edge_cosine_qcd", "relative_margin"]].dropna()
    if len(cosine_margin) >= 3:
        pearson = float(pearsonr(cosine_margin["edge_cosine_qcd"], cosine_margin["relative_margin"]).statistic)
        spearman = float(spearmanr(cosine_margin["edge_cosine_qcd"], cosine_margin["relative_margin"]).statistic)
    else:
        pearson = float("nan")
        spearman = float("nan")

    return {
        "triangle_pruning_rate": float(valid["triangle_prunable"].mean()),
        "edge_cosine_margin_pearson": pearson,
        "edge_cosine_margin_spearman": spearman,
        "pca_32_or_fewer_cumulative_variance": float(cumulative[-1]),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.input_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    query = pd.read_csv(args.input_dir / "query_stats.csv")
    dco = pd.read_csv(args.input_dir / "dco_trace.csv")
    edge = pd.read_csv(args.input_dir / "edge_directions.csv")

    validation = validate(query, dco, metadata)
    dco["candidate_class"] = add_candidate_class(dco)
    summary = summarize_queries(query)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    query.to_parquet(args.output_dir / "query_stats.parquet", index=False)
    dco.to_parquet(args.output_dir / "dco_trace.parquet", index=False)

    plot_metrics = make_plots(
        query,
        dco,
        edge,
        args.output_dir / "plots",
        args.exact_distance_ns,
        args.lookup_ns,
    )

    valid = dco[dco["threshold_valid_before"].astype(bool)]
    negative_or_neutral = valid[valid["is_state_neutral"].astype(bool)]
    metrics = {
        **validation,
        **plot_metrics,
        "mean_n_dist": float(query["n_dist"].mean()),
        "mean_recall_at_k": float(query["recall_at_k"].mean()),
        "strict_neutral_ratio": float(query["n_strict_state_neutral"].sum() / query["n_dist"].sum()),
        "valid_margin_count": int(len(valid)),
        "neutral_margin_gt_1pct": float((negative_or_neutral["relative_margin"] > 0.01).mean()) if len(negative_or_neutral) else 0.0,
        "neutral_margin_gt_5pct": float((negative_or_neutral["relative_margin"] > 0.05).mean()) if len(negative_or_neutral) else 0.0,
        "neutral_margin_gt_10pct": float((negative_or_neutral["relative_margin"] > 0.10).mean()) if len(negative_or_neutral) else 0.0,
    }
    with (args.output_dir / "validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "metrics": metrics}, handle, indent=2, allow_nan=True)

    report = [
        "# Baseline Trace Pipeline Validation",
        "",
        "The synthetic high-dimensional pipeline completed successfully.",
        "",
        f"- Queries: {metrics['query_count']}",
        f"- DCO records: {metrics['dco_record_count']}",
        f"- Mean N_dist: {metrics['mean_n_dist']:.2f}",
        f"- Mean recall@k: {metrics['mean_recall_at_k']:.4f}",
        f"- Strict neutral ratio: {metrics['strict_neutral_ratio']:.4f}",
        f"- Maximum relative geometry error: {metrics['max_relative_geometry_error']:.3e}",
        "",
        "This is a correctness validation run, not a performance result.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("analysis_pipeline_ok")
    print(json.dumps(metrics, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
