#!/usr/bin/env python3
"""Analyze aggregated real-data trace Parquet files without changing raw shards."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hnsw-real-trace-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def save_cdf(values: pd.Series, xlabel: str, output: Path) -> None:
    data = values.astype(float).replace([np.inf, -np.inf], np.nan).dropna().sort_values().to_numpy()
    if not len(data):
        raise RuntimeError(f"No finite values for {xlabel}")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(data, np.arange(1, len(data) + 1) / len(data))
    ax.set(xlabel=xlabel, ylabel="CDF")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def bootstrap_mean_ci(values: pd.Series, seed: int = 42, repeats: int = 2000) -> list[float]:
    data = values.astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if not len(data):
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=float)
    for start in range(0, repeats, 100):
        count = min(100, repeats - start)
        indices = rng.integers(0, len(data), size=(count, len(data)))
        means[start : start + count] = data[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots = args.output_dir / "plots"
    plots.mkdir(exist_ok=True)
    query = pd.read_parquet(args.input_dir / "query_stats.parquet")
    edge = pd.read_parquet(args.input_dir / "edge_directions.parquet")
    with (args.input_dir / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    edge_ok = query["n_edge_scan"] == query["n_duplicate"] + query["n_unique_neighbor"]
    distance_ok = query["n_unique_neighbor"] == query["n_dist"]
    if not edge_ok.all() or not distance_ok.all():
        raise RuntimeError("Aggregated query accounting failed")

    dco_file = pq.ParquetFile(args.input_dir / "dco_trace.parquet")
    dco_count = 0
    fully_neutral_count = 0
    valid_margin_count = 0
    valid_neutral_count = 0
    valid_neutral_margin_gt = {0.01: 0, 0.05: 0, 0.10: 0}
    sampled_inserted_candidate_count = 0
    sampled_threshold_changed_count = 0
    sampled_expanded_later_count = 0
    sampled_final_topk_count = 0
    per_query_sampled: dict[int, int] = {}
    per_query_fully_neutral: dict[int, int] = {}
    progress_total = np.zeros(10, dtype=np.int64)
    progress_valid = np.zeros(10, dtype=np.int64)
    progress_rejected = np.zeros(10, dtype=np.int64)
    progress_fully_neutral = np.zeros(10, dtype=np.int64)
    progress_inserted = np.zeros(10, dtype=np.int64)
    progress_threshold_changed = np.zeros(10, dtype=np.int64)
    samples: list[pd.DataFrame] = []
    sample_rows = 0
    sample_cap = 250_000
    for batch_index, batch in enumerate(dco_file.iter_batches(batch_size=200_000)):
        frame = batch.to_pandas()
        dco_count += len(frame)
        neutral_mask = frame["is_state_neutral"].astype(bool)
        valid_mask = frame["threshold_valid_before"].astype(bool)
        valid_neutral_mask = neutral_mask & valid_mask
        fully_neutral_count += int(neutral_mask.sum())
        valid_margin_count += int(valid_mask.sum())
        valid_neutral_count += int(valid_neutral_mask.sum())
        valid_neutral_margin = frame.loc[valid_neutral_mask, "relative_margin"].astype(float)
        for threshold in valid_neutral_margin_gt:
            valid_neutral_margin_gt[threshold] += int((valid_neutral_margin > threshold).sum())
        sampled_inserted_candidate_count += int(frame["inserted_candidate_queue"].astype(bool).sum())
        sampled_threshold_changed_count += int(frame["threshold_changed"].astype(bool).sum())
        sampled_expanded_later_count += int(frame["expanded_later"].astype(bool).sum())
        sampled_final_topk_count += int(frame["in_final_topk"].astype(bool).sum())
        query_ids = frame["query_id"].astype(int)
        for query_id, count in query_ids.value_counts().items():
            per_query_sampled[int(query_id)] = per_query_sampled.get(int(query_id), 0) + int(count)
        for query_id, count in frame.loc[neutral_mask, "query_id"].astype(int).value_counts().items():
            per_query_fully_neutral[int(query_id)] = per_query_fully_neutral.get(int(query_id), 0) + int(count)
        progress = np.minimum((frame["search_progress_fraction"].astype(float).to_numpy() * 10).astype(int), 9)
        progress = np.maximum(progress, 0)
        np.add.at(progress_total, progress, 1)
        np.add.at(progress_valid, progress[valid_mask.to_numpy()], 1)
        rejected_mask = valid_mask & frame["is_negative_at_evaluation"].astype(bool)
        np.add.at(progress_rejected, progress[rejected_mask.to_numpy()], 1)
        np.add.at(progress_fully_neutral, progress[neutral_mask.to_numpy()], 1)
        inserted_mask = frame["inserted_candidate_queue"].astype(bool)
        np.add.at(progress_inserted, progress[inserted_mask.to_numpy()], 1)
        changed_mask = frame["threshold_changed"].astype(bool)
        np.add.at(progress_threshold_changed, progress[changed_mask.to_numpy()], 1)
        take = min(5000, len(frame))
        if take:
            samples.append(frame.sample(n=take, random_state=batch_index))
            sample_rows += take
        if sample_rows > sample_cap * 2:
            merged = pd.concat(samples, ignore_index=True).sample(n=sample_cap, random_state=42)
            samples = [merged]
            sample_rows = len(merged)
    if dco_count == 0:
        raise RuntimeError("Aggregated DCO trace is empty")
    dco_sample = pd.concat(samples, ignore_index=True)
    if len(dco_sample) > sample_cap:
        dco_sample = dco_sample.sample(n=sample_cap, random_state=42)
    valid_neutral_sample = dco_sample[
        dco_sample["threshold_valid_before"].astype(bool)
        & dco_sample["is_state_neutral"].astype(bool)
    ]
    exact_dco_count = int(query["n_dist"].sum())
    rejected_count = int(query["n_strict_state_neutral"].sum())
    if exact_dco_count <= 0:
        raise RuntimeError("Aggregated query summaries contain no distance computations")
    per_query_neutral_ratios = pd.Series(
        [
            per_query_fully_neutral.get(int(query_id), 0) / per_query_sampled[int(query_id)]
            for query_id in sorted(per_query_sampled)
        ],
        dtype=float,
    )
    recall_ci = bootstrap_mean_ci(query["recall_at_k"], seed=42)
    n_dist_ci = bootstrap_mean_ci(query["n_dist"], seed=43)
    neutral_ci = bootstrap_mean_ci(per_query_neutral_ratios, seed=44)
    metrics = {
        "metric_schema_version": 2,
        "dataset": metadata["dataset"],
        "queries": len(query),
        "dco_records": dco_count,
        "exact_dco_count": exact_dco_count,
        "mean_recall_at_k": float(query["recall_at_k"].mean()),
        "mean_recall_at_k_bootstrap_95ci": recall_ci,
        "mean_recall_at_k_ci_low": recall_ci[0],
        "mean_recall_at_k_ci_high": recall_ci[1],
        "mean_n_dist": float(query["n_dist"].mean()),
        "mean_n_dist_bootstrap_95ci": n_dist_ci,
        "mean_n_dist_ci_low": n_dist_ci[0],
        "mean_n_dist_ci_high": n_dist_ci[1],
        "median_n_dist": float(query["n_dist"].median()),
        "p95_n_dist": float(query["n_dist"].quantile(0.95)),
        "p99_n_dist": float(query["n_dist"].quantile(0.99)),
        "rejected_at_evaluation_count": rejected_count,
        "rejected_at_evaluation_ratio": float(rejected_count / exact_dco_count),
        "sampled_fully_state_neutral_count": fully_neutral_count,
        "sampled_fully_state_neutral_ratio": float(fully_neutral_count / dco_count),
        "mean_per_query_sampled_fully_state_neutral_ratio": float(per_query_neutral_ratios.mean()),
        "mean_per_query_sampled_fully_state_neutral_ratio_bootstrap_95ci": neutral_ci,
        "mean_per_query_sampled_fully_state_neutral_ratio_ci_low": neutral_ci[0],
        "mean_per_query_sampled_fully_state_neutral_ratio_ci_high": neutral_ci[1],
        "sampled_threshold_valid_count": valid_margin_count,
        "sampled_threshold_valid_ratio": float(valid_margin_count / dco_count),
        "sampled_inserted_candidate_count": sampled_inserted_candidate_count,
        "sampled_inserted_candidate_ratio": float(sampled_inserted_candidate_count / dco_count),
        "sampled_threshold_changed_count": sampled_threshold_changed_count,
        "sampled_threshold_changed_ratio": float(sampled_threshold_changed_count / dco_count),
        "sampled_expanded_later_count": sampled_expanded_later_count,
        "sampled_final_topk_count": sampled_final_topk_count,
        "valid_margin_count": valid_margin_count,
        "valid_neutral_count": valid_neutral_count,
        "valid_neutral_margin_gt_1pct_count": valid_neutral_margin_gt[0.01],
        "valid_neutral_margin_gt_5pct_count": valid_neutral_margin_gt[0.05],
        "valid_neutral_margin_gt_10pct_count": valid_neutral_margin_gt[0.10],
        "valid_neutral_margin_gt_1pct": (
            float(valid_neutral_margin_gt[0.01] / valid_neutral_count) if valid_neutral_count else 0.0
        ),
        "valid_neutral_margin_gt_5pct": (
            float(valid_neutral_margin_gt[0.05] / valid_neutral_count) if valid_neutral_count else 0.0
        ),
        "valid_neutral_margin_gt_10pct": (
            float(valid_neutral_margin_gt[0.10] / valid_neutral_count) if valid_neutral_count else 0.0
        ),
    }
    # Backward-compatible aliases. New reports must use the explicit names above.
    metrics["strict_neutral_ratio"] = metrics["rejected_at_evaluation_ratio"]
    metrics["sampled_neutral_ratio"] = metrics["sampled_fully_state_neutral_ratio"]
    metrics["neutral_margin_gt_1pct"] = metrics["valid_neutral_margin_gt_1pct"]
    metrics["neutral_margin_gt_5pct"] = metrics["valid_neutral_margin_gt_5pct"]
    metrics["neutral_margin_gt_10pct"] = metrics["valid_neutral_margin_gt_10pct"]
    save_cdf(query["n_dist"], "N_dist per query", plots / "n_dist_cdf.png")
    if len(valid_neutral_sample):
        save_cdf(
            valid_neutral_sample["relative_margin"],
            "valid fully-neutral relative margin (bounded sample)",
            plots / "relative_margin_cdf.png",
        )

    progress_frame = pd.DataFrame(
        {
            "progress_decile": np.arange(10),
            "sampled_dco_count": progress_total,
            "threshold_valid_ratio": np.divide(
                progress_valid, progress_total, out=np.zeros(10, dtype=float), where=progress_total > 0
            ),
            "rejected_at_evaluation_ratio": np.divide(
                progress_rejected, progress_total, out=np.zeros(10, dtype=float), where=progress_total > 0
            ),
            "fully_state_neutral_ratio": np.divide(
                progress_fully_neutral, progress_total, out=np.zeros(10, dtype=float), where=progress_total > 0
            ),
            "candidate_insertion_ratio": np.divide(
                progress_inserted, progress_total, out=np.zeros(10, dtype=float), where=progress_total > 0
            ),
            "threshold_change_ratio": np.divide(
                progress_threshold_changed, progress_total, out=np.zeros(10, dtype=float), where=progress_total > 0
            ),
        }
    )
    sampled_progress = np.minimum(
        (dco_sample["search_progress_fraction"].astype(float).to_numpy() * 10).astype(int), 9
    )
    dco_sample = dco_sample.assign(progress_decile=np.maximum(sampled_progress, 0))
    margin_sample = dco_sample[
        dco_sample["threshold_valid_before"].astype(bool)
        & dco_sample["is_state_neutral"].astype(bool)
    ]
    median_margin = margin_sample.groupby("progress_decile")["relative_margin"].median()
    progress_frame["bounded_sample_median_valid_neutral_margin"] = progress_frame["progress_decile"].map(median_margin)
    progress_frame.to_csv(args.output_dir / "search_progress.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for column, label in (
        ("rejected_at_evaluation_ratio", "rejected at evaluation"),
        ("fully_state_neutral_ratio", "fully state-neutral"),
        ("threshold_valid_ratio", "threshold valid"),
        ("candidate_insertion_ratio", "candidate insertion"),
    ):
        ax.plot(progress_frame["progress_decile"], progress_frame[column], marker="o", label=label)
    ax.set(xlabel="search progress decile", ylabel="sampled DCO ratio", ylim=(0, 1.02))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "search_progress_ratios.png", dpi=160)
    plt.close(fig)

    direction_columns = [column for column in edge.columns if column.startswith("dir_")]
    if direction_columns and len(edge) >= 2:
        values = edge[direction_columns].to_numpy(dtype=float)
        components = min(32, values.shape[0], values.shape[1])
        cumulative = np.cumsum(PCA(n_components=components).fit(values).explained_variance_ratio_)
        metrics["edge_direction_sample_count"] = int(values.shape[0])
        metrics["edge_direction_dimension"] = int(values.shape[1])
        metrics["edge_pca_components"] = int(components)
        metrics["edge_pca_cumulative_variance"] = float(cumulative[-1])
        metrics["edge_pca_cumulative_curve"] = [float(value) for value in cumulative]
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(np.arange(1, components + 1), cumulative, marker="o", markersize=3)
        ax.set(xlabel="components", ylabel="cumulative explained variance", ylim=(0, 1.02))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots / "edge_direction_pca.png", dpi=160)
        plt.close(fig)

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Real-Dataset HNSW Trace Report",
        "",
        f"- Dataset: {metrics['dataset']}",
        f"- Queries: {metrics['queries']}",
        f"- Sampled DCO rows: {metrics['dco_records']}",
        f"- Mean Recall@K: {metrics['mean_recall_at_k']:.6f}",
        f"- Mean N_dist: {metrics['mean_n_dist']:.2f}",
        f"- Rejected-at-evaluation ratio: {metrics['rejected_at_evaluation_ratio']:.6f} "
        f"({metrics['rejected_at_evaluation_count']}/{metrics['exact_dco_count']})",
        f"- Sampled fully-state-neutral ratio: {metrics['sampled_fully_state_neutral_ratio']:.6f} "
        f"({metrics['sampled_fully_state_neutral_count']}/{metrics['dco_records']})",
        f"- Valid fully-neutral margins: {metrics['valid_neutral_count']}",
        f"- Valid fully-neutral margin >5%: {metrics['valid_neutral_margin_gt_5pct']:.6f} "
        f"({metrics['valid_neutral_margin_gt_5pct_count']}/{metrics['valid_neutral_count']})",
        "",
        "These are trace-distribution results. Performance conclusions require the separate trace-disabled benchmark.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("analyze_real_trace_ok")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
