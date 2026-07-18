#!/usr/bin/env python3
"""Compare completed provenance-rich real-dataset trace summaries."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hnsw-cross-dataset-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PLOT_METRICS = {
    "mean_recall_at_k": "Mean Recall@K",
    "mean_n_dist": "Mean distance computations",
    "rejected_at_evaluation_ratio": "Rejected-at-evaluation ratio",
    "sampled_fully_state_neutral_ratio": "Fully-state-neutral ratio",
    "valid_neutral_margin_gt_5pct": "Valid neutral margin >5%",
    "edge_pca_cumulative_variance": "PCA cumulative explained variance",
    "qps_mean": "Trace-disabled queries per second",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Required file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if len(args.run_root) < 2:
        raise RuntimeError("At least two --run-root values are required")

    rows: list[dict] = []
    provenance: list[dict] = []
    dataset_names: set[str] = set()
    for root_arg in args.run_root:
        run_root = root_arg.resolve()
        summary = load_json(run_root / "summary" / "summary.json")
        if int(summary.get("metric_schema_version", 0)) < 2:
            raise RuntimeError(
                f"{run_root} uses legacy metrics; regenerate its analysis and summary with schema version 2"
            )
        dataset = str(summary["dataset"])
        if dataset in dataset_names:
            raise RuntimeError(f"Duplicate dataset in comparison: {dataset}")
        dataset_names.add(dataset)
        dimension = int(summary["dimension"])
        provenance.append({key: value for key, value in summary.items() if key != "results"})
        for record in summary["results"]:
            row = dict(record)
            row.update(
                {
                    "run_id": summary["run_id"],
                    "dataset": dataset,
                    "dimension": dimension,
                    "n_base": int(summary["n_base"]),
                    "n_query": int(summary["n_query"]),
                    "experiment_commit": summary["experiment_commit"],
                    "analysis_commit": summary["analysis_commit"],
                    "metric_schema_version": int(summary["metric_schema_version"]),
                }
            )
            rows.append(row)

    frame = pd.DataFrame(rows).sort_values(["dataset", "ef_search"]).reset_index(drop=True)
    required = {"dataset", "dimension", "ef_search", *PLOT_METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Cross-dataset input is missing columns: {', '.join(missing)}")

    ef_sets = {
        dataset: set(group["ef_search"].astype(int))
        for dataset, group in frame.groupby("dataset", sort=True)
    }
    if len({tuple(sorted(values)) for values in ef_sets.values()}) != 1:
        raise RuntimeError(f"Datasets do not share one efSearch matrix: {ef_sets}")

    output = args.output_dir.resolve()
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "summary.csv", index=False)
    payload = {
        "metric_schema_version": 2,
        "datasets": sorted(dataset_names),
        "provenance": provenance,
        "results": json.loads(frame.to_json(orient="records")),
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for metric, ylabel in PLOT_METRICS.items():
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for dataset, group in frame.groupby("dataset", sort=True):
            label = f"{dataset} ({int(group['dimension'].iloc[0])}D)"
            ax.plot(group["ef_search"], group[metric], marker="o", label=label)
        ax.set(xlabel="efSearch", ylabel=ylabel)
        ax.set_xticks(sorted(frame["ef_search"].unique()))
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / f"{metric}.png", dpi=160)
        plt.close(fig)

    columns = [
        "dataset", "dimension", "ef_search", "mean_recall_at_k", "mean_n_dist",
        "rejected_at_evaluation_ratio", "sampled_fully_state_neutral_ratio",
        "valid_neutral_count", "valid_neutral_margin_gt_5pct",
        "edge_pca_cumulative_variance", "qps_mean",
    ]
    display = frame[columns].copy()
    for column in columns[3:]:
        if column != "valid_neutral_count":
            display[column] = display[column].map(lambda value: f"{float(value):.6g}")
    report = [
        "# SIFT1M and GIST1M Baseline-Trace Comparison",
        "",
        "## Core results",
        "",
        markdown_table(display),
        "",
        "## Metric definitions",
        "",
        "- `rejected_at_evaluation_ratio` is the exact all-DCO fraction not inserted into the candidate queue.",
        "- `sampled_fully_state_neutral_ratio` uses deterministic sampled DCOs and the complete state-neutral definition.",
        "- Margin fractions use only sampled, fully-state-neutral DCOs whose threshold was valid before evaluation.",
        "",
        "## Interpretation boundary",
        "",
        "This comparison tests whether observational opportunity signals generalize across datasets. "
        "It does not measure quantization error, establish a safe representation, prove Recall preservation, "
        "or demonstrate an end-to-end speedup.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"cross_dataset_summary_ok output_dir={output}")


if __name__ == "__main__":
    main()
