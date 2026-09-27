from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

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


METHODS = ("pq8", "pq4", "opq", "prq", "jq", "rabitq")
PROFILES = ("balanced", "strict")
COLORS = {
    "pq8": "#0072B2",
    "pq4": "#56B4E9",
    "opq": "#D55E00",
    "prq": "#009E73",
    "jq": "#CC79A7",
    "rabitq": "#E69F00",
}
MARKERS = {"pq8": "o", "pq4": "o", "opq": "s", "prq": "o",
           "jq": "s", "rabitq": "o"}
DISPLAY = {"pq8": "PQ8", "pq4": "PQ4", "opq": "OPQ", "prq": "PRQ",
           "jq": "JQ", "rabitq": "RaBitQ"}
REQUIRED_INPUTS = (
    "quality_sweep.csv", "operating_points.csv", "timing_raw.csv",
    "timing_summary.csv", "batch_scaling.csv", "batch_parity.csv")


class PlotError(ValueError):
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
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def _read_inputs(data_dir: Path) -> dict[str, pd.DataFrame]:
    missing = [name for name in REQUIRED_INPUTS
               if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("missing comparison inputs: " + ", ".join(missing))
    frames = {Path(name).stem: pd.read_csv(data_dir / name)
              for name in REQUIRED_INPUTS}
    operating = frames["operating_points"]
    timing = frames["timing_summary"]
    quality = frames["quality_sweep"]
    if set(operating["method"]) != set(METHODS):
        raise PlotError("operating points do not contain all six methods")
    if set(operating["profile"]) != set(PROFILES):
        raise PlotError("balanced and strict operating points are required")
    if set(timing["method"]) != set(METHODS):
        raise PlotError("timing summary does not contain all six methods")
    if set(quality["method"]) != set(METHODS):
        raise PlotError("quality sweep does not contain all six methods")
    parity = frames["batch_parity"]
    valid = parity["valid"].astype(str).str.lower().eq("true")
    if not bool(valid.all()):
        raise PlotError("batch parity contains a failing row")
    return frames


def _save_figure(fig: plt.Figure, directory: Path, stem: str) -> list[Path]:
    outputs = []
    for suffix in ("png", "pdf", "svg"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    plt.close(fig)
    return outputs


def _plot_quality_cost(frames: Mapping[str, pd.DataFrame],
                       directory: Path) -> list[Path]:
    merged = frames["operating_points"].merge(
        frames["timing_summary"], on="method", validate="many_to_one")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    offsets = {
        ("balanced", "pq8"): (5, 5), ("balanced", "pq4"): (5, -12),
        ("balanced", "opq"): (5, 6), ("balanced", "prq"): (5, 5),
        ("balanced", "jq"): (5, -12), ("balanced", "rabitq"): (-45, 6),
        ("strict", "pq8"): (5, -12), ("strict", "pq4"): (5, -12),
        ("strict", "opq"): (5, 6), ("strict", "prq"): (5, 5),
        ("strict", "jq"): (5, 5), ("strict", "rabitq"): (-45, 6),
    }
    for axis, profile in zip(axes, PROFILES):
        subset = merged[merged["profile"] == profile]
        for _, row in subset.iterrows():
            method = str(row["method"])
            x = float(row["ns_per_eligible_edge"])
            y = float(row["correct_prune_rate"]) * 100.0
            scale = x / float(row["median_elapsed_ms"])
            low = float(row["block_min_ms"]) * scale
            high = float(row["block_max_ms"]) * scale
            axis.errorbar(
                x, y, xerr=np.array([[x - low], [high - x]]),
                fmt=MARKERS[method], color=COLORS[method], markersize=6,
                markeredgecolor="black", markeredgewidth=0.35,
                capsize=2.5, elinewidth=0.8, zorder=3)
            dx, dy = offsets[(profile, method)]
            axis.annotate(DISPLAY[method], (x, y), xytext=(dx, dy),
                          textcoords="offset points", fontsize=8)
        axis.set_xscale("log")
        axis.set_title(profile.capitalize())
        axis.set_xlabel("Estimator cost (ns / eligible edge, log scale)")
        axis.set_ylim(0, 72)
        axis.grid(True, which="both", axis="both")
    axes[0].set_ylabel("Correct-prune rate (%)")
    shape_legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#888888",
               markeredgecolor="black", markersize=6, label="Scalar path"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#888888",
               markeredgecolor="black", markersize=6, label="Batch-128 path"),
    ]
    axes[1].legend(handles=shape_legend, loc="lower right", frameon=False)
    fig.suptitle("Quality–cost Pareto comparison", fontweight="bold")
    fig.text(0.5, -0.01,
             "Upper-left is better; horizontal whiskers show the range of block medians.",
             ha="center", fontsize=8)
    fig.tight_layout()
    return _save_figure(fig, directory, "fig01_quality_cost_pareto")


def _plot_selected_risk(frames: Mapping[str, pd.DataFrame],
                        directory: Path) -> list[Path]:
    data = frames["operating_points"].copy()
    metrics = (
        ("global_false_prune_rate", "Global false-prune rate (%)",
         "max_global_false_prune_rate"),
        ("p95_query_false_prune_rate", "p95 query false-prune rate (%)",
         "max_p95_query_false_prune_rate"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.8), sharey=True)
    order = list(reversed(METHODS))
    y = np.arange(len(order))
    for column, profile in enumerate(PROFILES):
        subset = data[data["profile"] == profile].set_index("method")
        for row_index, (metric, label, gate_name) in enumerate(metrics):
            axis = axes[row_index, column]
            values = np.array([float(subset.loc[method, metric]) * 100.0
                               for method in order])
            for yi, method, value in zip(y, order, values):
                axis.scatter(value, yi, s=34, marker=MARKERS[method],
                             color=COLORS[method], edgecolor="black",
                             linewidth=0.35, zorder=3)
            gate = float(subset.iloc[0][gate_name]) * 100.0
            axis.axvline(gate, color="#555555", linestyle="--", linewidth=1.0,
                         label=f"Gate = {gate:g}%")
            axis.set_xscale("symlog", linthresh=0.001, linscale=0.7)
            axis.set_xlim(left=0)
            axis.set_yticks(y, [DISPLAY[method] for method in order])
            axis.set_xlabel(label)
            axis.set_title(profile.capitalize() if row_index == 0 else "")
            axis.legend(loc="lower right", frameon=False)
            axis.grid(True, which="both", axis="x")
    fig.suptitle("Selected operating points satisfy false-prune gates",
                 fontweight="bold")
    fig.tight_layout()
    return _save_figure(fig, directory, "fig02_selected_false_prune")


def _plot_batch_scaling(frames: Mapping[str, pd.DataFrame],
                        directory: Path) -> list[Path]:
    data = frames["batch_scaling"].sort_values(["method", "batch_size"])
    sizes = sorted(int(value) for value in data["batch_size"].unique())
    positions = np.arange(len(sizes))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    for method in ("opq", "jq"):
        subset = data[data["method"] == method].set_index("batch_size").loc[sizes]
        axes[0].plot(positions, subset["median_elapsed_ms"], marker=MARKERS[method],
                     color=COLORS[method], label=DISPLAY[method])
        axes[1].plot(positions, subset["speedup_vs_batch1"],
                     marker=MARKERS[method], color=COLORS[method],
                     label=DISPLAY[method])
    for axis in axes:
        axis.set_xticks(positions, [str(value) for value in sizes])
        axis.set_xlabel("Query batch size")
        if 128 in sizes:
            axis.axvline(sizes.index(128), color="#555555", linestyle="--",
                         linewidth=1.0, label="Selected: 128")
        axis.legend(frameon=False)
    axes[0].set_ylabel("Median trace time (ms)")
    axes[0].set_title("Absolute component cost")
    axes[1].set_ylabel("Speedup over batch 1 (×)")
    axes[1].set_title("Amortization benefit")
    fig.suptitle("Batched query preparation saturates near batch 128",
                 fontweight="bold")
    fig.tight_layout()
    return _save_figure(fig, directory, "fig03_batch_scaling")


def _plot_alpha_risk(frames: Mapping[str, pd.DataFrame],
                     directory: Path) -> list[Path]:
    quality = frames["quality_sweep"]
    operating = frames["operating_points"]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.2), sharey=True)
    for axis, method in zip(axes.flat, METHODS):
        subset = quality[quality["method"] == method].sort_values("alpha")
        x = subset["p95_query_false_prune_rate"].astype(float) * 100.0
        y = subset["correct_prune_rate"].astype(float) * 100.0
        axis.plot(x, y, color=COLORS[method], marker=MARKERS[method],
                  markersize=3.5)
        chosen = operating[operating["method"] == method]
        selected_points: dict[tuple[float, float], list[str]] = {}
        for _, row in chosen.iterrows():
            px = float(row["p95_query_false_prune_rate"]) * 100.0
            py = float(row["correct_prune_rate"]) * 100.0
            label = "B" if row["profile"] == "balanced" else "S"
            selected_points.setdefault((px, py), []).append(label)
        for (px, py), labels in selected_points.items():
            axis.scatter(px, py, s=52, marker=MARKERS[method],
                         facecolor=COLORS[method], edgecolor="black",
                         linewidth=0.7, zorder=4)
            axis.annotate("/".join(labels), (px, py), xytext=(4, 3),
                          textcoords="offset points", fontsize=8,
                          fontweight="bold")
        axis.axvline(0.25, color="#666666", linestyle=":", linewidth=0.9)
        axis.axvline(1.0, color="#666666", linestyle="--", linewidth=0.9)
        axis.set_xscale("symlog", linthresh=0.001, linscale=0.7)
        axis.set_xlim(left=0)
        axis.set_title(DISPLAY[method])
        axis.set_xlabel("p95 query false-prune rate (%)")
        axis.set_ylabel("Correct-prune rate (%)")
        axis.grid(True, which="both")
    handles = [
        Line2D([0], [0], color="#666666", linestyle=":",
               label="Strict p95 gate: 0.25%"),
        Line2D([0], [0], color="#666666", linestyle="--",
               label="Balanced p95 gate: 1%"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Alpha-dependent quality–risk frontiers", fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    return _save_figure(fig, directory, "figS1_alpha_quality_risk")


def _plot_cost_memory(frames: Mapping[str, pd.DataFrame],
                      directory: Path) -> list[Path]:
    data = frames["timing_summary"].copy().sort_values("ns_per_eligible_edge")
    methods = list(data["method"])
    labels = [DISPLAY[method] for method in methods]
    colors = [COLORS[method] for method in methods]
    y = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    costs = data["ns_per_eligible_edge"].astype(float).to_numpy()
    cost_floor = float(costs.min()) * 0.75
    for yi, value, color in zip(y, costs, colors):
        axes[0].hlines(yi, cost_floor, value, color=color, linewidth=2.0,
                       alpha=0.9)
        axes[0].scatter(value, yi, color=color, edgecolor="black",
                        linewidth=0.35, s=38, zorder=3)
    axes[0].set_xscale("log")
    axes[0].set_xlim(cost_floor, float(costs.max()) * 1.25)
    axes[0].set_xlabel("Estimator cost (ns / eligible edge, log scale)")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[1].barh(y, data["backend_mib"], color=colors, alpha=0.9)
    axes[1].set_xlabel("Backend memory (MiB)")
    for axis, values, formatter in (
        (axes[0], costs, lambda value: f"{value:.1f}"),
        (axes[1], data["backend_mib"], lambda value: f"{value:.0f}"),
    ):
        for yi, value in zip(y, values):
            axis.annotate(formatter(float(value)), (float(value), yi),
                          xytext=(4, 0), textcoords="offset points",
                          va="center", fontsize=8)
        axis.grid(True, axis="x")
    fig.suptitle("Component cost and backend memory", fontweight="bold")
    fig.tight_layout()
    return _save_figure(fig, directory, "figS2_cost_memory")


def _percent(value: object, digits: int = 3) -> str:
    return f"{float(value) * 100.0:.{digits}f}"


def _write_rows(path: Path, fields: Sequence[str],
                rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _latex_table(headers: Sequence[str], rows: Sequence[Sequence[str]],
                 caption: str, label: str) -> str:
    alignment = "ll" + "r" * (len(headers) - 2)
    escaped_headers = [value.replace("%", r"\%") for value in headers]
    lines = [
        r"\begin{table}[t]", r"\centering",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{alignment}}}", r"\toprule",
        " & ".join(escaped_headers) + r" \\", r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _write_tables(frames: Mapping[str, pd.DataFrame],
                  directory: Path) -> list[Path]:
    outputs: list[Path] = []
    operating = frames["operating_points"].copy()
    operating["method_order"] = operating["method"].map(
        {method: index for index, method in enumerate(METHODS)})
    operating["profile_order"] = operating["profile"].map(
        {profile: index for index, profile in enumerate(PROFILES)})
    operating = operating.sort_values(["method_order", "profile_order"])
    fields1 = ("method", "profile", "alpha", "correct_prune_percent",
               "global_false_prune_percent", "p95_query_false_prune_percent",
               "prune_precision_percent", "prunable_coverage_percent")
    rows1 = [{
        "method": DISPLAY[str(row["method"])],
        "profile": str(row["profile"]).capitalize(),
        "alpha": f"{float(row['alpha']):.2f}",
        "correct_prune_percent": _percent(row["correct_prune_rate"], 2),
        "global_false_prune_percent": _percent(
            row["global_false_prune_rate"], 4),
        "p95_query_false_prune_percent": _percent(
            row["p95_query_false_prune_rate"], 4),
        "prune_precision_percent": _percent(row["prune_precision"], 3),
        "prunable_coverage_percent": _percent(row["prunable_coverage"], 2),
    } for _, row in operating.iterrows()]
    headers1 = ("Method", "Profile", "Alpha", "Correct prune (%)",
                "Global FP (%)", "p95 query FP (%)", "Precision (%)",
                "Prunable coverage (%)")
    display1 = [[str(row[field]) for field in fields1] for row in rows1]
    for suffix, content in (
        ("csv", None),
        ("md", _markdown_table(headers1, display1)),
        ("tex", _latex_table(
            headers1, display1,
            "Selected balanced and strict quality operating points.",
            "tab:uq-operating-points")),
    ):
        path = directory / f"table1_operating_points.{suffix}"
        if suffix == "csv":
            _write_rows(path, fields1, rows1)
        else:
            path.write_text(str(content), encoding="utf-8", newline="\n")
        outputs.append(path)

    timing = frames["timing_summary"].copy()
    timing["method_order"] = timing["method"].map(
        {method: index for index, method in enumerate(METHODS)})
    timing = timing.sort_values("method_order")
    fields2 = ("method", "execution", "median_trace_ms", "ns_per_edge",
               "speed_vs_pq8", "block_cv_percent", "backend_mib")
    rows2 = [{
        "method": DISPLAY[str(row["method"])],
        "execution": (f"Batch-{int(row['query_batch_size'])}"
                      if row["execution_mode"] == "batch" else "Scalar"),
        "median_trace_ms": f"{float(row['median_elapsed_ms']):.3f}",
        "ns_per_edge": f"{float(row['ns_per_eligible_edge']):.3f}",
        "speed_vs_pq8": f"{float(row['paired_speed_ratio_vs_reference']):.3f}",
        "block_cv_percent": f"{float(row['block_cv_percent']):.2f}",
        "backend_mib": f"{float(row['backend_mib']):.2f}",
    } for _, row in timing.iterrows()]
    headers2 = ("Method", "Execution", "Median trace (ms)", "ns / edge",
                "Speed vs PQ8 (x)", "Block CV (%)", "Backend (MiB)")
    display2 = [[str(row[field]) for field in fields2] for row in rows2]
    for suffix, content in (
        ("csv", None),
        ("md", _markdown_table(headers2, display2)),
        ("tex", _latex_table(
            headers2, display2,
            "Same-node randomized paired component timing and memory.",
            "tab:uq-component-cost")),
    ):
        path = directory / f"table2_component_cost.{suffix}"
        if suffix == "csv":
            _write_rows(path, fields2, rows2)
        else:
            path.write_text(str(content), encoding="utf-8", newline="\n")
        outputs.append(path)

    batch = frames["batch_scaling"].copy().sort_values(["batch_size", "method"])
    fields3 = ("batch_size", "method", "median_trace_ms", "ns_per_query",
               "speedup_vs_batch1")
    rows3 = [{
        "batch_size": str(int(row["batch_size"])),
        "method": DISPLAY[str(row["method"])],
        "median_trace_ms": f"{float(row['median_elapsed_ms']):.3f}",
        "ns_per_query": f"{float(row['ns_per_query']):.3f}",
        "speedup_vs_batch1": f"{float(row['speedup_vs_batch1']):.3f}",
    } for _, row in batch.iterrows()]
    headers3 = ("Batch", "Method", "Median trace (ms)", "ns / query",
                "Speedup vs batch 1 (x)")
    display3 = [[str(row[field]) for field in fields3] for row in rows3]
    for suffix, content in (
        ("csv", None),
        ("md", _markdown_table(headers3, display3)),
        ("tex", _latex_table(
            headers3, display3,
            "OPQ and JQ batched query-preparation scaling.",
            "tab:uq-batch-scaling")),
    ):
        path = directory / f"table3_batch_scaling.{suffix}"
        if suffix == "csv":
            _write_rows(path, fields3, rows3)
        else:
            path.write_text(str(content), encoding="utf-8", newline="\n")
        outputs.append(path)
    return outputs


README = """# Quantizer comparison figures and tables

## Main figures

- `fig01_quality_cost_pareto`: balanced/strict quality–cost Pareto comparison.
- `fig02_selected_false_prune`: selected-point false-prune risks and gates.
- `fig03_batch_scaling`: OPQ/JQ batch amortization and the batch-128 choice.

## Supplementary figures

- `figS1_alpha_quality_risk`: full alpha-dependent p95-risk frontiers.
- `figS2_cost_memory`: normalized component cost and backend memory.

Each figure is emitted as PNG (preview), PDF and SVG (publication/vector).
Tables are emitted as CSV, Markdown and LaTeX/booktabs.

Important scope: correct-prune rate is TP divided by all valid decisions and is
not Recall@K. Timings measure estimator replay on the frozen trace and are not
end-to-end HNSW latency or QPS. OPQ/JQ use batch 128 in the final comparison;
the other methods use their scalar paths.
"""


def generate(data_dir: Path, output: Path) -> None:
    frames = _read_inputs(data_dir)
    _configure_style()
    with atomic_output_dir(output) as partial:
        figure_dir = partial / "figures"
        table_dir = partial / "tables"
        figure_dir.mkdir()
        table_dir.mkdir()
        outputs: list[Path] = []
        for function in (
            _plot_quality_cost,
            _plot_selected_risk,
            _plot_batch_scaling,
            _plot_alpha_risk,
            _plot_cost_memory,
        ):
            outputs.extend(function(frames, figure_dir))
        outputs.extend(_write_tables(frames, table_dir))
        readme = partial / "README.md"
        readme.write_text(README, encoding="utf-8", newline="\n")
        outputs.append(readme)
        manifest = partial / "manifest.json"
        _write_manifest(manifest, data_dir, outputs)
        outputs.append(manifest)
        complete = partial / "complete.json"
        complete.write_text(json.dumps({
            "schema_version": 1,
            "stage": "comparison-figures-and-tables",
            "outputs": [file_entry(path) | {
                "path": str(path.relative_to(partial)).replace("\\", "/")}
                for path in outputs],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "out": str(output.resolve()),
        "figure_files": 15,
        "table_files": 9,
    }, sort_keys=True))


def _write_manifest(path: Path, data_dir: Path, outputs: Sequence[Path]) -> None:
    inputs = []
    for name in REQUIRED_INPUTS:
        source = data_dir / name
        inputs.append({"path": name, "size": source.stat().st_size,
                       "sha256": sha256_file(source)})
    value = {
        "schema_version": 1,
        "stage": "comparison-figures-and-tables",
        "inputs": inputs,
        "outputs": [{
            "path": str(output.relative_to(path.parent)).replace("\\", "/"),
            "size": output.stat().st_size,
            "sha256": sha256_file(output),
        } for output in outputs],
        "plot_contract": {
            "methods": list(METHODS),
            "profiles": list(PROFILES),
            "formats": ["png", "pdf", "svg"],
            "rates_in_figures": "percent",
            "timing_scope": "component replay, not end-to-end QPS",
        },
    }
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate publication figures and tables from frozen data")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    generate(args.data_dir.resolve(), args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
