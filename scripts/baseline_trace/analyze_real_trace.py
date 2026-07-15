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
    neutral_count = 0
    valid_margin_count = 0
    neutral_margin_gt = {0.01: 0, 0.05: 0, 0.10: 0}
    samples: list[pd.DataFrame] = []
    sample_rows = 0
    sample_cap = 250_000
    for batch_index, batch in enumerate(dco_file.iter_batches(batch_size=200_000)):
        frame = batch.to_pandas()
        dco_count += len(frame)
        neutral_mask = frame["is_state_neutral"].astype(bool)
        neutral_count += int(neutral_mask.sum())
        valid_margin_count += int(frame["threshold_valid_before"].astype(bool).sum())
        neutral_margin = frame.loc[neutral_mask, "relative_margin"].astype(float)
        for threshold in neutral_margin_gt:
            neutral_margin_gt[threshold] += int((neutral_margin > threshold).sum())
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
    valid_sample = dco_sample[dco_sample["threshold_valid_before"].astype(bool)]
    metrics = {
        "dataset": metadata["dataset"],
        "queries": len(query),
        "dco_records": dco_count,
        "mean_recall_at_k": float(query["recall_at_k"].mean()),
        "mean_n_dist": float(query["n_dist"].mean()),
        "strict_neutral_ratio": float(query["n_strict_state_neutral"].sum() / query["n_dist"].sum()),
        "sampled_neutral_ratio": float(neutral_count / dco_count),
        "valid_margin_count": valid_margin_count,
        "neutral_margin_gt_1pct": float(neutral_margin_gt[0.01] / neutral_count) if neutral_count else 0.0,
        "neutral_margin_gt_5pct": float(neutral_margin_gt[0.05] / neutral_count) if neutral_count else 0.0,
        "neutral_margin_gt_10pct": float(neutral_margin_gt[0.10] / neutral_count) if neutral_count else 0.0,
    }
    save_cdf(query["n_dist"], "N_dist per query", plots / "n_dist_cdf.png")
    if len(valid_sample):
        save_cdf(valid_sample["relative_margin"], "relative margin (bounded sample)", plots / "relative_margin_cdf.png")

    direction_columns = [column for column in edge.columns if column.startswith("dir_")]
    if direction_columns and len(edge) >= 2:
        values = edge[direction_columns].to_numpy(dtype=float)
        components = min(32, values.shape[0], values.shape[1])
        cumulative = np.cumsum(PCA(n_components=components).fit(values).explained_variance_ratio_)
        metrics["edge_pca_cumulative_variance"] = float(cumulative[-1])
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
        f"- Strict neutral ratio: {metrics['strict_neutral_ratio']:.6f}",
        "",
        "These are trace-distribution results. Performance conclusions require the separate trace-disabled benchmark.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("analyze_real_trace_ok")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
