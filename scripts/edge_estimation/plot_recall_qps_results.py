from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.edge_estimation.contracts import (  # noqa: E402
    atomic_output_dir, file_entry, sha256_file)


METHODS = ("hnsw", "pq8", "pq_qjl", "opq")
DISPLAY = {
    "hnsw": "HNSW",
    "pq8": "PQ8",
    "pq_qjl": "PQ8 + QJL",
    "opq": "OPQ",
}
COLORS = {
    "hnsw": "#4D4D4D",
    "pq8": "#0072B2",
    "pq_qjl": "#009E73",
    "opq": "#D55E00",
}
MARKERS = {"hnsw": "o", "pq8": "s", "pq_qjl": "^", "opq": "D"}
THRESHOLDS = (0.90, 0.95, 0.97, 0.98, 0.985, 0.99)
REQUIRED_COLUMNS = (
    "case_id", "method", "beta", "ef_search", "recall_at_k", "qps",
    "latency_p50_ns", "latency_p95_ns", "latency_p99_ns",
    "batch_prepare_ns_per_query", "attempted_estimates",
    "pruned_estimates", "fallback_estimates",
)


class RecallQpsPlotError(ValueError):
    pass


def _configure_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def _read_points(inputs: Sequence[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        missing = [column for column in REQUIRED_COLUMNS if column not in frame]
        if missing:
            raise RecallQpsPlotError(f"{path}: missing columns {missing}")
        frame = frame.loc[:, REQUIRED_COLUMNS].copy()
        frame.insert(0, "run", path.parent.name)
        frame.insert(1, "source_file", str(path.resolve()))
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if set(data["method"]) != set(METHODS):
        raise RecallQpsPlotError("inputs must contain HNSW, PQ8, PQ8+QJL and OPQ")
    if data[["run", "case_id"]].duplicated().any():
        raise RecallQpsPlotError("duplicate run/case_id rows detected")
    numeric = [
        "beta", "ef_search", "recall_at_k", "qps", "latency_p50_ns",
        "latency_p95_ns", "latency_p99_ns", "batch_prepare_ns_per_query",
        "attempted_estimates", "pruned_estimates", "fallback_estimates",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[["ef_search", "recall_at_k", "qps"]].isna().any().any():
        raise RecallQpsPlotError("ef_search, recall_at_k and qps must be numeric")
    data["method_display"] = data["method"].map(DISPLAY)
    data["recall_percent"] = data["recall_at_k"] * 100.0
    for percentile in (50, 95, 99):
        data[f"latency_p{percentile}_ms"] = (
            data[f"latency_p{percentile}_ns"] / 1_000_000.0)
    data["prune_rate"] = data["pruned_estimates"] / data["attempted_estimates"]
    data["fallback_rate"] = data["fallback_estimates"] / data["attempted_estimates"]
    data["pareto"] = False
    for method in METHODS:
        indices = data.index[data["method"] == method]
        for index in indices:
            row = data.loc[index]
            peers = data.loc[indices]
            dominates = (
                (peers["recall_at_k"] >= row["recall_at_k"])
                & (peers["qps"] >= row["qps"])
                & ((peers["recall_at_k"] > row["recall_at_k"])
                   | (peers["qps"] > row["qps"]))
            )
            data.loc[index, "pareto"] = not bool(dominates.any())
    method_order = {method: i for i, method in enumerate(METHODS)}
    data["method_order"] = data["method"].map(method_order)
    return data.sort_values(
        ["method_order", "beta", "ef_search"], na_position="first").reset_index(drop=True)


def _threshold_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        selected: dict[str, pd.Series] = {}
        for method in METHODS:
            eligible = data[(data["method"] == method)
                            & (data["recall_at_k"] >= threshold)]
            if not eligible.empty:
                selected[method] = eligible.sort_values(
                    ["qps", "recall_at_k"], ascending=[False, False]).iloc[0]
        baseline_qps = (float(selected["hnsw"]["qps"])
                        if "hnsw" in selected else None)
        for method in METHODS:
            row = selected.get(method)
            if row is None:
                rows.append({
                    "recall_threshold": threshold,
                    "method": method,
                    "method_display": DISPLAY[method],
                    "status": "NO_POINT",
                    "case_id": "",
                    "recall_at_k": np.nan,
                    "qps": np.nan,
                    "qps_uplift_vs_hnsw": np.nan,
                    "ef_search": np.nan,
                    "beta": np.nan,
                })
                continue
            qps = float(row["qps"])
            uplift = ((qps / baseline_qps) - 1.0
                      if baseline_qps is not None else np.nan)
            rows.append({
                "recall_threshold": threshold,
                "method": method,
                "method_display": DISPLAY[method],
                "status": "OK",
                "case_id": str(row["case_id"]),
                "recall_at_k": float(row["recall_at_k"]),
                "qps": qps,
                "qps_uplift_vs_hnsw": uplift,
                "ef_search": int(row["ef_search"]),
                "beta": (float(row["beta"])
                         if pd.notna(row["beta"]) else np.nan),
            })
    return pd.DataFrame(rows)


def _method_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        subset = data[data["method"] == method]
        rows.append({
            "method": method,
            "method_display": DISPLAY[method],
            "point_count": len(subset),
            "pareto_point_count": int(subset["pareto"].sum()),
            "recall_min": float(subset["recall_at_k"].min()),
            "recall_max": float(subset["recall_at_k"].max()),
            "qps_min": float(subset["qps"].min()),
            "qps_max": float(subset["qps"].max()),
            "beta_values": ", ".join(
                f"{value:g}" for value in sorted(subset["beta"].dropna().unique())),
            "ef_min": int(subset["ef_search"].min()),
            "ef_max": int(subset["ef_search"].max()),
        })
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, directory: Path, stem: str) -> list[Path]:
    outputs = []
    for suffix in ("png", "pdf", "svg"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    plt.close(fig)
    return outputs


def _plot_frontier(data: pd.DataFrame, directory: Path) -> list[Path]:
    fig, axis = plt.subplots(figsize=(8.3, 5.0))
    for method in METHODS:
        raw = data[data["method"] == method]
        frontier = raw[raw["pareto"]].sort_values("recall_percent")
        axis.scatter(raw["recall_percent"], raw["qps"], s=20,
                     color=COLORS[method], marker=MARKERS[method], alpha=0.22,
                     edgecolors="none")
        axis.plot(frontier["recall_percent"], frontier["qps"],
                  color=COLORS[method], marker=MARKERS[method], markersize=4.5,
                  label=DISPLAY[method])
    for threshold in (95, 97, 98):
        axis.axvline(threshold, color="#8A8A8A", linestyle="--",
                     linewidth=0.8, alpha=0.65)
    axis.set_xlim(90, 98.55)
    axis.set_ylim(bottom=100)
    axis.set_xlabel("Recall@10 (%)")
    axis.set_ylabel("Queries per second (QPS)")
    axis.set_title("End-to-end Recall–QPS frontier", fontweight="bold")
    axis.legend(frameon=False, ncol=2)
    axis.text(0.01, 0.02,
              "Faint markers: all grid points; solid lines: non-dominated frontier",
              transform=axis.transAxes, fontsize=8, color="#555555")
    fig.tight_layout()
    return _save_figure(fig, directory, "fig01_recall_qps_frontier")


def _plot_beta_curves(data: pd.DataFrame, directory: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True, sharey=True)
    baseline = data[data["method"] == "hnsw"].sort_values("recall_percent")
    for axis, method in zip(axes, ("pq8", "pq_qjl", "opq")):
        axis.plot(baseline["recall_percent"], baseline["qps"], color=COLORS["hnsw"],
                  marker=MARKERS["hnsw"], markersize=3.5, alpha=0.75,
                  label="HNSW")
        subset = data[data["method"] == method]
        betas = sorted(float(value) for value in subset["beta"].dropna().unique())
        palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(betas)))
        for beta, color in zip(betas, palette):
            curve = subset[np.isclose(subset["beta"], beta)].sort_values("recall_percent")
            axis.plot(curve["recall_percent"], curve["qps"], marker=MARKERS[method],
                      markersize=3.6, color=color, label=f"beta={beta:g}")
        axis.set_title(DISPLAY[method])
        axis.set_xlabel("Recall@10 (%)")
        axis.set_xlim(90, 98.55)
        axis.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("QPS")
    fig.suptitle("Parameter-grid curves by beta", fontweight="bold")
    fig.tight_layout()
    return _save_figure(fig, directory, "fig02_beta_curves")


def _plot_threshold_qps(summary: pd.DataFrame, directory: Path) -> list[Path]:
    selected_thresholds = (0.90, 0.95, 0.97, 0.98)
    fig, axis = plt.subplots(figsize=(8.3, 4.4))
    positions = np.arange(len(selected_thresholds))
    width = 0.19
    for offset, method in enumerate(METHODS):
        values = []
        for threshold in selected_thresholds:
            row = summary[(summary["recall_threshold"] == threshold)
                          & (summary["method"] == method)].iloc[0]
            values.append(float(row["qps"]) if row["status"] == "OK" else np.nan)
        bars = axis.bar(positions + (offset - 1.5) * width, values, width,
                        color=COLORS[method], label=DISPLAY[method])
        axis.bar_label(bars, fmt="%.0f", padding=2, fontsize=7)
    axis.set_xticks(positions, [f">= {threshold * 100:g}%" for threshold in selected_thresholds])
    axis.set_xlabel("Minimum Recall@10")
    axis.set_ylabel("Best observed eligible QPS")
    axis.set_title("Best grid point above each recall threshold", fontweight="bold")
    axis.legend(frameon=False, ncol=4, loc="upper right")
    axis.set_ylim(bottom=0)
    fig.tight_layout()
    return _save_figure(fig, directory, "fig03_threshold_qps")


def _plot_uplift(summary: pd.DataFrame, directory: Path) -> list[Path]:
    selected_thresholds = (0.90, 0.95, 0.97, 0.98)
    methods = ("pq8", "pq_qjl", "opq")
    fig, axis = plt.subplots(figsize=(8.3, 4.4))
    positions = np.arange(len(selected_thresholds))
    width = 0.24
    for offset, method in enumerate(methods):
        values = []
        for threshold in selected_thresholds:
            row = summary[(summary["recall_threshold"] == threshold)
                          & (summary["method"] == method)].iloc[0]
            values.append(float(row["qps_uplift_vs_hnsw"]) * 100.0)
        bars = axis.bar(positions + (offset - 1) * width, values, width,
                        color=COLORS[method], label=DISPLAY[method])
        axis.bar_label(bars, fmt="%+.1f%%", padding=2, fontsize=7)
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set_xticks(positions, [f">= {threshold * 100:g}%" for threshold in selected_thresholds])
    axis.set_xlabel("Minimum Recall@10")
    axis.set_ylabel("QPS uplift vs threshold-qualified HNSW (%)")
    axis.set_title("Observed QPS advantage over HNSW", fontweight="bold")
    axis.legend(frameon=False, ncol=3)
    fig.tight_layout()
    return _save_figure(fig, directory, "fig04_qps_uplift_vs_hnsw")


README = """# Recall–QPS comparison report

This report combines the successful v1 grid (70 points) and high-recall v2
supplement (21 points). The failed one-second pilot run is intentionally not an
input because it produced no valid `points.csv`.

## Figures

- `fig01_recall_qps_frontier`: all observations and per-method non-dominated frontiers.
- `fig02_beta_curves`: detailed beta/efSearch curves for PQ8, PQ8+QJL and OPQ.
- `fig03_threshold_qps`: best observed QPS among points meeting each recall threshold.
- `fig04_qps_uplift_vs_hnsw`: threshold-qualified uplift relative to HNSW.

Each figure is provided as PNG, PDF and SVG. Tables include all 91 points,
Pareto-front points, per-method coverage, and threshold summaries.

Important: this is a one-repeat pilot grid. Threshold-qualified uplift compares
the best observed point at or above each threshold; it is not interpolated exact
matched-recall QPS and should not be presented as a confidence-bounded estimate.
"""


def generate(inputs: Sequence[Path], output: Path) -> None:
    data = _read_points(inputs)
    threshold = _threshold_summary(data)
    method = _method_summary(data)
    pareto = data[data["pareto"]].copy()
    export_columns = [
        "run", "source_file", "case_id", "method", "method_display", "beta",
        "ef_search", "recall_at_k", "recall_percent", "qps",
        "latency_p50_ns", "latency_p95_ns", "latency_p99_ns",
        "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
        "batch_prepare_ns_per_query", "attempted_estimates", "pruned_estimates",
        "fallback_estimates", "prune_rate", "fallback_rate", "pareto",
    ]
    _configure_style()
    with atomic_output_dir(output) as partial:
        figure_dir = partial / "figures"
        table_dir = partial / "tables"
        figure_dir.mkdir()
        table_dir.mkdir()
        outputs: list[Path] = []
        outputs.extend(_plot_frontier(data, figure_dir))
        outputs.extend(_plot_beta_curves(data, figure_dir))
        outputs.extend(_plot_threshold_qps(threshold, figure_dir))
        outputs.extend(_plot_uplift(threshold, figure_dir))
        tables = {
            "all_points.csv": data.loc[:, export_columns],
            "pareto_front.csv": pareto.loc[:, export_columns],
            "method_summary.csv": method,
            "threshold_summary.csv": threshold,
        }
        for name, frame in tables.items():
            path = table_dir / name
            frame.to_csv(path, index=False, float_format="%.9g")
            outputs.append(path)
        workbook_payload = partial / "workbook_payload.json"
        workbook_payload.write_text(json.dumps({
            "schema_version": 1,
            "all_points": data.loc[:, export_columns].astype(object).where(
                pd.notna(data.loc[:, export_columns]), None).to_dict("records"),
            "pareto_front": pareto.loc[:, export_columns].astype(object).where(
                pd.notna(pareto.loc[:, export_columns]), None).to_dict("records"),
            "method_summary": method.astype(object).where(
                pd.notna(method), None).to_dict("records"),
            "threshold_summary": threshold.astype(object).where(
                pd.notna(threshold), None).to_dict("records"),
            "inputs": [str(path.resolve()) for path in inputs],
        }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        outputs.append(workbook_payload)
        readme = partial / "README.md"
        readme.write_text(README, encoding="utf-8", newline="\n")
        outputs.append(readme)
        manifest = partial / "manifest.json"
        manifest.write_text(json.dumps({
            "schema_version": 1,
            "stage": "recall-qps-figures-and-tables",
            "inputs": [{"path": str(path.resolve()), "size": path.stat().st_size,
                        "sha256": sha256_file(path)} for path in inputs],
            "outputs": [file_entry(path) | {
                "path": str(path.relative_to(partial)).replace("\\", "/")}
                for path in outputs],
            "point_count": len(data),
            "method_count": len(METHODS),
            "thresholds": list(THRESHOLDS),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        outputs.append(manifest)
        complete = partial / "complete.json"
        complete.write_text(json.dumps({
            "valid": True,
            "point_count": len(data),
            "figure_count": 12,
            "table_count": 4,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "out": str(output.resolve()),
                      "point_count": len(data)}, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Recall-QPS figures and detailed tables")
    parser.add_argument("--input", action="append", required=True, type=Path,
                        help="Successful points.csv; repeat for multiple runs")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    generate([path.resolve() for path in args.input], args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
