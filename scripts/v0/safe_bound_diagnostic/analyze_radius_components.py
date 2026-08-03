#!/usr/bin/env python3
"""Validate schema-v2 V0 shadow records and analyze radius components."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA_VERSION = 2
REQUIRED_COLUMNS = (
    "schema_version",
    "run_id",
    "query_id",
    "current_node_id",
    "candidate_id",
    "graph_layer",
    "bound_status",
    "ef_search",
    "current_squared_distance",
    "threshold",
    "edge_length",
    "direction_error",
    "anchor_projection",
    "anchor_projection_lower",
    "query_direction_inner_product_upper",
    "residual_direction_inner_product_upper",
    "length_squared_lower",
    "cross_term_upper",
    "base_plus_length_lower",
    "approximate_squared_distance",
    "current_distance_root_upper",
    "direction_error_radius",
    "stored_numeric_padding",
    "operational_l2_padding",
    "rounding_closure_padding",
    "error_radius",
    "lower_bound",
    "current_lb",
    "shadow_exact_squared_distance",
    "would_prune",
    "current_would_prune",
    "oracle_would_prune",
    "lower_bound_valid",
    "lower_bound_violation",
    "false_prune",
)

NUMERIC_COLUMNS = tuple(
    name
    for name in REQUIRED_COLUMNS
    if name not in {
        "run_id", "would_prune", "current_would_prune", "oracle_would_prune",
        "lower_bound_valid", "lower_bound_violation", "false_prune"
    }
)

BOOL_COLUMNS = (
    "would_prune",
    "current_would_prune",
    "oracle_would_prune",
    "lower_bound_valid",
    "lower_bound_violation",
    "false_prune",
)

COMPONENTS = (
    "direction_error_radius",
    "stored_numeric_padding",
    "operational_l2_padding",
    "rounding_closure_padding",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-query-count", type=int)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--relative-tolerance", type=float, default=5e-13)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-12)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_records(run_dir: Path) -> Path:
    plain = run_dir / "shadow_records.csv"
    compressed = run_dir / "shadow_records.csv.gz"
    if plain.exists():
        return plain
    if compressed.exists():
        return compressed
    raise FileNotFoundError("shadow_records.csv[.gz] is missing")


def normalize_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    numeric = pd.to_numeric(series, errors="raise")
    if not numeric.isin([0, 1]).all():
        raise ValueError(f"invalid boolean values in {series.name}")
    return numeric.astype(bool)


def prepare_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS) - set(chunk.columns))
    if missing:
        raise ValueError(f"schema-v2 columns are missing: {missing}")
    result = chunk.copy()
    for name in NUMERIC_COLUMNS:
        result[name] = pd.to_numeric(result[name], errors="raise")
    for name in BOOL_COLUMNS:
        result[name] = normalize_bool(result[name])
    return result


def derive(chunk: pd.DataFrame) -> pd.DataFrame:
    result = chunk.copy()
    result["margin"] = result["shadow_exact_squared_distance"] - result["threshold"]
    result["raw_margin"] = result["approximate_squared_distance"] - result["threshold"]
    result["current_prune_gap"] = result["lower_bound"] - result["threshold"]
    result["safety_slack"] = result["shadow_exact_squared_distance"] - result["lower_bound"]
    denominator = np.maximum(np.abs(result["margin"].to_numpy(float)), 1e-15)
    result["radius_margin_ratio"] = result["error_radius"].to_numpy(float) / denominator
    for name in COMPONENTS:
        result[f"share_{name}"] = np.divide(
            result[name].to_numpy(float),
            np.maximum(result["error_radius"].to_numpy(float), 1e-300),
        )
    result["oracle_prunable"] = result["shadow_exact_squared_distance"] > result["threshold"]
    result["raw_prunable"] = result["approximate_squared_distance"] > result["threshold"]
    result["current_prunable"] = result["lower_bound"] > result["threshold"]
    return result


def quantiles(series: pd.Series) -> dict[str, float]:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {key: math.nan for key in ("mean", "p50", "p90", "p95", "p99", "max")}
    return {
        "mean": float(clean.mean()),
        "p50": float(clean.quantile(0.50)),
        "p90": float(clean.quantile(0.90)),
        "p95": float(clean.quantile(0.95)),
        "p99": float(clean.quantile(0.99)),
        "max": float(clean.max()),
    }


def write_parquet(frames: Iterable[pd.DataFrame], path: Path) -> None:
    writer: pq.ParquetWriter | None = None
    try:
        for frame in frames:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def make_figures(data: pd.DataFrame, output_dir: Path) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    shares = [float(data[f"share_{name}"].median()) for name in COMPONENTS]
    labels = ["direction", "stored", "operational", "rounding"]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(labels, shares, color=["#2166ac", "#67a9cf", "#ef8a62", "#b2182b"])
    ax.set_ylabel("Median share of total radius")
    ax.set_ylim(0.0, max(1.0, max(shares) * 1.1 if shares else 1.0))
    ax.set_title("V0 conservative radius components")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "01_radius_component_median_share.png", dpi=180)
    plt.close(fig)

    sampled = data
    if len(sampled) > 100_000:
        sampled = sampled.sample(100_000, random_state=20260803)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.hexbin(
        sampled["raw_margin"],
        sampled["error_radius"],
        gridsize=70,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    finite = sampled[["raw_margin", "error_radius"]].replace([np.inf, -np.inf], np.nan).dropna()
    if not finite.empty:
        low = float(min(finite["raw_margin"].min(), 0.0))
        high = float(max(finite["raw_margin"].max(), finite["error_radius"].max(), 0.0))
        ax.plot([low, high], [low, high], color="#d73027", linewidth=1.3, label="raw_margin = radius")
        ax.legend(loc="upper left")
    ax.set_xlabel("Raw margin: approximate distance - threshold")
    ax.set_ylabel("Total error radius")
    ax.set_title("Raw pruning margin versus conservative radius")
    fig.tight_layout()
    fig.savefig(figures / "02_raw_margin_vs_radius.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = read_json(args.run_dir / "summary.json")
    metadata = read_json(args.run_dir / "metadata.json")
    records_path = find_records(args.run_dir)

    failures: list[str] = []
    if int(metadata.get("shadow_schema_version", -1)) != SCHEMA_VERSION:
        failures.append("metadata shadow_schema_version is not 2")
    if int(summary.get("shadow_schema_version", -1)) != SCHEMA_VERSION:
        failures.append("summary shadow_schema_version is not 2")
    if args.expected_query_count is not None and int(summary.get("query_count", -1)) != args.expected_query_count:
        failures.append("query_count does not match --expected-query-count")
    for name in ("mismatch_queries", "lower_bound_violation", "false_prune", "exact_distance_saved"):
        if int(summary.get(name, -1)) != 0:
            failures.append(f"summary {name} is not zero")
    bound_evaluated_summary = int(summary.get("bound_evaluated", 0))
    raw_prunable_summary = int(summary.get("raw_prunable", -1))
    oracle_prunable_summary = int(summary.get("oracle_prunable", -1))
    if not 0 <= raw_prunable_summary <= bound_evaluated_summary:
        failures.append("summary raw_prunable is outside [0, bound_evaluated]")
    if not 0 <= oracle_prunable_summary <= bound_evaluated_summary:
        failures.append("summary oracle_prunable is outside [0, bound_evaluated]")

    frames: list[pd.DataFrame] = []
    rows = 0
    closure_mismatches = 0
    lower_mismatches = 0
    decision_mismatches = 0
    current_alias_mismatches = 0
    oracle_decision_mismatches = 0
    invalid_numeric_rows = 0
    schema_versions: set[int] = set()
    run_ids: set[str] = set()

    for raw in pd.read_csv(records_path, chunksize=args.chunksize):
        chunk = prepare_chunk(raw)
        schema_versions.update(int(value) for value in chunk["schema_version"].unique())
        run_ids.update(str(value) for value in chunk["run_id"].unique())
        valid = chunk[chunk["lower_bound_valid"]].copy()
        if valid.empty:
            continue
        rows += len(valid)
        finite = np.isfinite(valid[list(NUMERIC_COLUMNS)].to_numpy(float)).all(axis=1)
        invalid_numeric_rows += int((~finite).sum())

        component_sum = sum(valid[name].to_numpy(float) for name in COMPONENTS)
        closure_mismatches += int((~np.isclose(
            valid["error_radius"].to_numpy(float),
            component_sum,
            rtol=args.relative_tolerance,
            atol=args.absolute_tolerance,
        )).sum())

        expected_lower = np.maximum(
            0.0,
            np.nextafter(
                valid["approximate_squared_distance"].to_numpy(float)
                - valid["error_radius"].to_numpy(float),
                -np.inf,
            ),
        )
        lower_mismatches += int((~np.isclose(
            valid["lower_bound"].to_numpy(float),
            expected_lower,
            rtol=args.relative_tolerance,
            atol=args.absolute_tolerance,
        )).sum())
        expected_decision = valid["lower_bound"] > valid["threshold"]
        decision_mismatches += int((expected_decision != valid["would_prune"]).sum())
        current_alias_mismatches += int((~np.isclose(
            valid["lower_bound"].to_numpy(float),
            valid["current_lb"].to_numpy(float),
            rtol=0.0,
            atol=0.0,
        )).sum())
        current_alias_mismatches += int(
            (valid["would_prune"] != valid["current_would_prune"]).sum()
        )
        expected_oracle = valid["shadow_exact_squared_distance"] > valid["threshold"]
        oracle_decision_mismatches += int(
            (expected_oracle != valid["oracle_would_prune"]).sum()
        )
        frames.append(derive(valid))

    if schema_versions != {SCHEMA_VERSION}:
        failures.append(f"record schema versions are {sorted(schema_versions)}")
    if len(run_ids) != 1:
        failures.append(f"expected one run_id, found {sorted(run_ids)}")
    if rows == 0:
        failures.append("no valid sampled bound records")
    if invalid_numeric_rows:
        failures.append(f"non-finite numeric rows={invalid_numeric_rows}")
    if closure_mismatches:
        failures.append(f"radius component closure mismatches={closure_mismatches}")
    if lower_mismatches:
        failures.append(f"lower-bound reconstruction mismatches={lower_mismatches}")
    if decision_mismatches:
        failures.append(f"would-prune mismatches={decision_mismatches}")
    if current_alias_mismatches:
        failures.append(f"current method alias mismatches={current_alias_mismatches}")
    if oracle_decision_mismatches:
        failures.append(f"oracle decision mismatches={oracle_decision_mismatches}")

    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not data.empty:
        write_parquet((frame for frame in frames), args.output_dir / "candidate_metrics.parquet")
        component_rows = []
        for name in (*COMPONENTS, "error_radius", "margin", "raw_margin", "radius_margin_ratio"):
            row = {"metric": name, **quantiles(data[name])}
            if name in COMPONENTS:
                row["mean_share"] = float(data[f"share_{name}"].mean())
                row["median_share"] = float(data[f"share_{name}"].median())
            component_rows.append(row)
        pd.DataFrame(component_rows).to_csv(
            args.output_dir / "radius_component_summary.csv", index=False
        )
        by_query = data.groupby("query_id", sort=True).agg(
            sampled_rows=("query_id", "size"),
            oracle_coverage=("oracle_prunable", "mean"),
            raw_coverage=("raw_prunable", "mean"),
            current_coverage=("current_prunable", "mean"),
            median_error_radius=("error_radius", "median"),
            median_radius_margin_ratio=("radius_margin_ratio", "median"),
        )
        by_query.to_csv(args.output_dir / "radius_component_by_query.csv")
        make_figures(data, args.output_dir)

    sampled_oracle = float(data["oracle_prunable"].mean()) if not data.empty else math.nan
    sampled_raw = float(data["raw_prunable"].mean()) if not data.empty else math.nan
    sampled_current = float(data["current_prunable"].mean()) if not data.empty else math.nan
    bound_evaluated = int(summary.get("bound_evaluated", 0))
    full_current = (
        int(summary.get("bound_pruned", 0)) / bound_evaluated
        if bound_evaluated
        else math.nan
    )
    full_raw = raw_prunable_summary / bound_evaluated if bound_evaluated else math.nan
    full_oracle = oracle_prunable_summary / bound_evaluated if bound_evaluated else math.nan
    quality = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "schema_version": SCHEMA_VERSION,
        "run_ids": sorted(run_ids),
        "analysis_valid_rows": rows,
        "invalid_numeric_rows": invalid_numeric_rows,
        "radius_component_closure_mismatches": closure_mismatches,
        "lower_bound_reconstruction_mismatches": lower_mismatches,
        "would_prune_mismatches": decision_mismatches,
        "current_method_alias_mismatches": current_alias_mismatches,
        "oracle_decision_mismatches": oracle_decision_mismatches,
        "sampled_oracle_coverage": sampled_oracle,
        "sampled_raw_coverage": sampled_raw,
        "sampled_current_coverage": sampled_current,
        "full_oracle_coverage": full_oracle,
        "full_raw_coverage": full_raw,
        "full_current_coverage": full_current,
    }
    with (args.output_dir / "data_quality.json").open("w", encoding="utf-8") as handle:
        json.dump(quality, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    dominant = "n/a"
    if not data.empty:
        medians = {name: float(data[f"share_{name}"].median()) for name in COMPONENTS}
        dominant = max(medians, key=medians.get)
    report = f"""# V0 radius component diagnostic

- Gate: **{quality['status'].upper()}**
- Valid sampled rows: `{rows}`
- Full current safe coverage: `{full_current:.8%}`
- Full oracle/raw/current coverage: `{full_oracle:.8%}` / `{full_raw:.8%}` / `{full_current:.8%}`
- Sampled oracle/raw/current coverage: `{sampled_oracle:.8%}` / `{sampled_raw:.8%}` / `{sampled_current:.8%}`
- Median-share dominant component: `{dominant}`
- Component closure mismatches: `{closure_mismatches}`
- Lower-bound reconstruction mismatches: `{lower_mismatches}`
- Would-prune mismatches: `{decision_mismatches}`
- Current alias mismatches: `{current_alias_mismatches}`
- Oracle decision mismatches: `{oracle_decision_mismatches}`

This stage diagnoses the current radius only. It does not authorize an empirical radius scale or real pruning change.
"""
    (args.output_dir / "analysis_summary.md").write_text(report, encoding="utf-8")

    print(json.dumps(quality, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
