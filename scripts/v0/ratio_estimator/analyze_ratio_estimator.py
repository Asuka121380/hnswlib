#!/usr/bin/env python3
"""Phase-1 diagnostic for the V0 ratio-corrected PQ estimator.

This program is deliberately offline-only.  It audits the exporter schema,
freezes a query-level split, reconstructs the raw and ratio-corrected
projection/distance estimators, checks exact-distance closure, and emits the
tables and figures required by the Phase-1 plan.  It never changes HNSW
control flow or the V0 sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hnsw-ratio-estimator-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ratio_estimator_core as estimator_core


ANALYZER_VERSION = "v0_ratio_estimator_phase1_v1"
SPLIT_MANIFEST_VERSION = 1
SCHEMA_AUDIT_VERSION = 1
DEFAULT_KAPPA_MIN_CANDIDATES = (0.1, 0.2, 0.3, 0.4, 0.5)
QUANTILES = (0.0, 0.01, 0.10, 0.50, 0.90, 0.95, 0.99, 0.999, 1.0)

REQUIRED_DIRECT_COLUMNS = {
    "query_id",
    "current_node_id",
    "candidate_id",
    "threshold",
    "edge_length",
    "reconstruction_norm",
    "x_norm",
    "x_dot_r",
    "x_dot_true_direction",
    "actual_direction_error",
    "direction_error",
    "current_lb",
    "diagnostic_valid",
}

# The existing cap exporter uses exact_squared_distance.  The plan uses the
# newer canonical name.  This is an explicit, audited alias rather than a
# silent default.
CANONICAL_ALIASES = {
    "shadow_exact_squared_distance": (
        "geometric_squared_distance",
        "shadow_exact_squared_distance",
        "exact_squared_distance",
    )
}

NUMERIC_COLUMNS = (
    "threshold",
    "edge_length",
    "reconstruction_norm",
    "x_norm",
    "x_dot_r",
    "x_dot_true_direction",
    "actual_direction_error",
    "direction_error",
    "current_lb",
    "shadow_exact_squared_distance",
)

IDENTIFIER_COLUMNS = ("query_id", "current_node_id", "candidate_id")


class SchemaError(ValueError):
    """Raised when the input cannot be interpreted without guessing."""


@dataclass(frozen=True)
class AnalysisConfig:
    mode: str = "formal"
    seed: int = 42
    train_fraction: float = 0.60
    calibration_fraction: float = 0.20
    test_fraction: float = 0.20
    closure_absolute_tolerance: float = 1e-10
    closure_relative_tolerance: float = 1e-9
    projection_tolerance: float = 1e-9
    kappa_min_candidates: tuple[float, ...] = DEFAULT_KAPPA_MIN_CANDIDATES
    selected_kappa_min: float | None = None
    minimum_eligible_fraction: float = 0.99
    skip_plots: bool = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: object) -> object:
    """Recursively replace non-finite scalars with JSON null."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def stable_query_sort_key(value: str) -> tuple[int, int | str, str]:
    text = str(value)
    try:
        return (0, int(text), text)
    except ValueError:
        return (1, text, text)


def allocate_split_sizes(query_count: int) -> tuple[int, int, int]:
    """Return deterministic 60/20/20 counts while keeping train non-empty."""
    if query_count <= 0:
        raise ValueError("cannot split an empty query set")
    train = int(math.floor(query_count * 0.60))
    calibration = int(math.floor(query_count * 0.20))
    if train == 0:
        train = 1
    test = query_count - train - calibration
    return train, calibration, test


def create_query_split(query_ids: Iterable[str], seed: int = 42) -> dict[str, list[str]]:
    ordered = sorted({str(value) for value in query_ids}, key=stable_query_sort_key)
    if not ordered:
        raise ValueError("input contains no query IDs")
    shuffled = ordered.copy()
    random.Random(seed).shuffle(shuffled)
    train_count, calibration_count, _ = allocate_split_sizes(len(shuffled))
    train_end = train_count
    calibration_end = train_end + calibration_count
    return {
        "train": shuffled[:train_end],
        "calibration": shuffled[train_end:calibration_end],
        "test": shuffled[calibration_end:],
    }


def validate_split(split: Mapping[str, Sequence[str]], query_ids: Iterable[str]) -> None:
    expected = {str(value) for value in query_ids}
    sets = {name: {str(value) for value in split.get(name, [])} for name in ("train", "calibration", "test")}
    if sets["train"] & sets["calibration"] or sets["train"] & sets["test"] or sets["calibration"] & sets["test"]:
        raise ValueError("query-level split leakage detected")
    assigned = sets["train"] | sets["calibration"] | sets["test"]
    if assigned != expected:
        missing = sorted(expected - assigned, key=stable_query_sort_key)
        extra = sorted(assigned - expected, key=stable_query_sort_key)
        raise ValueError(f"split/query mismatch: missing={missing}, extra={extra}")


def normalize_bool(series: pd.Series, name: str) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    mapped = text.map({"1": True, "0": False, "true": True, "false": False})
    if mapped.isna().any():
        examples = sorted(set(text[mapped.isna()].dropna().head(5).tolist()))
        raise SchemaError(f"invalid boolean values in {name}: {examples}")
    return mapped.astype(bool)


def audit_and_normalize_schema(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    columns = list(raw.columns)
    missing = sorted(REQUIRED_DIRECT_COLUMNS - set(columns))
    alias_resolution: dict[str, str] = {}
    coexisting_alias_differences: dict[str, dict[str, float | int]] = {}
    normalized = raw.copy()

    for canonical, aliases in CANONICAL_ALIASES.items():
        present = [name for name in aliases if name in columns]
        if not present:
            missing.append(canonical)
            continue
        selected = present[0]
        alias_resolution[canonical] = selected
        if len(present) > 1:
            left = pd.to_numeric(raw[present[0]], errors="coerce")
            for other_name in present[1:]:
                right = pd.to_numeric(raw[other_name], errors="coerce")
                finite = np.isfinite(left.to_numpy(float)) & np.isfinite(right.to_numpy(float))
                difference = np.abs(left.to_numpy(float) - right.to_numpy(float))
                coexisting_alias_differences[f"{present[0]}_vs_{other_name}"] = {
                    "finite_comparison_count": int(finite.sum()),
                    "different_count": int((finite & (difference != 0.0)).sum()),
                    "max_abs_difference": float(difference[finite].max()) if finite.any() else math.nan,
                }
        normalized[canonical] = raw[selected]

    audit: dict[str, object] = {
        "format": "v0_ratio_estimator_schema_audit",
        "format_version": SCHEMA_AUDIT_VERSION,
        "status": "pass" if not missing else "fail",
        "input_columns": columns,
        "required_direct_columns": sorted(REQUIRED_DIRECT_COLUMNS),
        "canonical_aliases": {key: list(value) for key, value in CANONICAL_ALIASES.items()},
        "alias_resolution": alias_resolution,
        "missing_columns": sorted(set(missing)),
        "coexisting_source_differences": json_safe(coexisting_alias_differences),
    }
    if missing:
        details = []
        if missing:
            details.append("missing columns: " + ", ".join(sorted(set(missing))))
        raise SchemaError("; ".join(details))
    return normalized, audit


def numeric_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    result = frame.copy()
    conversion_failure = pd.Series(False, index=result.index)
    for name in NUMERIC_COLUMNS:
        converted = pd.to_numeric(result[name], errors="coerce")
        conversion_failure |= converted.isna() | ~np.isfinite(converted.to_numpy(dtype=float))
        result[name] = converted.astype(float)
    for name in IDENTIFIER_COLUMNS:
        result[name] = result[name].astype("string")
    result["diagnostic_valid"] = normalize_bool(result["diagnostic_valid"], "diagnostic_valid")
    return result, conversion_failure


def append_reason(reason: pd.Series, mask: pd.Series | np.ndarray, label: str) -> None:
    mask_series = pd.Series(mask, index=reason.index).fillna(False).astype(bool)
    empty = reason.eq("") & mask_series
    reason.loc[empty] = label
    reason.loc[~empty & mask_series] = reason.loc[~empty & mask_series] + ";" + label


def derive_estimators(frame: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    return estimator_core.derive_estimators(frame, config)


def choose_kappa_min(data: pd.DataFrame, candidates: Sequence[float], minimum_fraction: float) -> tuple[float, pd.DataFrame]:
    train = data[(data["split"] == "train") & data["analysis_valid"]]
    if train.empty:
        raise ValueError("no analysis-valid train records are available")
    rows = []
    for candidate in candidates:
        eligible = train["kappa_actual"] >= candidate
        subset = train[eligible]
        rows.append({
            "kappa_min": float(candidate),
            "analysis_valid_records": int(len(train)),
            "eligible_records": int(eligible.sum()),
            "eligible_fraction": float(eligible.mean()),
            "ratio_clipped_distance_bias": float(subset["distance_error_ratio_clipped"].mean()) if len(subset) else math.nan,
            "ratio_clipped_distance_mae": float(subset["distance_error_ratio_clipped"].abs().mean()) if len(subset) else math.nan,
            "ratio_clipped_distance_p99_abs_error": float(subset["distance_error_ratio_clipped"].abs().quantile(0.99)) if len(subset) else math.nan,
        })
    scan = pd.DataFrame(rows)
    supported = scan[scan["eligible_fraction"] >= minimum_fraction]
    if not supported.empty:
        selected = float(supported.sort_values("kappa_min").iloc[-1]["kappa_min"])
    else:
        selected = float(scan.sort_values(["eligible_fraction", "kappa_min"], ascending=[False, True]).iloc[0]["kappa_min"])
    return selected, scan


def metric_row(data: pd.DataFrame, split: str, estimator: str, clipping: str) -> dict[str, object]:
    subset = data[(data["split"] == split) & data["estimator_eligible"]]
    rho_error = subset[f"rho_error_{estimator}_{clipping}"].replace([np.inf, -np.inf], np.nan).dropna()
    distance_error = subset[f"distance_error_{estimator}_{clipping}"].replace([np.inf, -np.inf], np.nan).dropna()
    row: dict[str, object] = {
        "split": split,
        "estimator": estimator,
        "clipping": clipping,
        "count": int(len(distance_error)),
        "invalid_count_all_input": int((~data["analysis_valid"]).sum()),
        "fallback_count_split": int(((data["split"] == split) & ~data["estimator_eligible"]).sum()),
    }
    for label, values in (("rho", rho_error), ("distance", distance_error)):
        row[f"{label}_mean_error"] = float(values.mean()) if len(values) else math.nan
        row[f"{label}_median_error"] = float(values.median()) if len(values) else math.nan
        row[f"{label}_mae"] = float(values.abs().mean()) if len(values) else math.nan
        row[f"{label}_rmse"] = float(np.sqrt(np.mean(np.square(values)))) if len(values) else math.nan
        row[f"{label}_std"] = float(values.std(ddof=1)) if len(values) > 1 else math.nan
        for probability in QUANTILES:
            key = f"p{probability * 100:g}".replace(".", "_")
            row[f"{label}_{key}"] = float(values.quantile(probability)) if len(values) else math.nan
    out_of_range = subset[f"{estimator}_rho_out_of_range"]
    row["rho_out_of_range_fraction"] = float(out_of_range.mean()) if len(out_of_range) else math.nan
    return row


def summarize_estimators(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    visible_splits = [name for name in ("train", "calibration", "test") if (data["split"] == name).any()]
    for split in visible_splits:
        for estimator in ("raw", "ratio", "ratio_meta"):
            for clipping in ("unclipped", "clipped"):
                rows.append(metric_row(data, split, estimator, clipping))
    return pd.DataFrame(rows)


def query_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for query_id, group in data.groupby("query_id", sort=False):
        eligible = group[group["estimator_eligible"]]
        for estimator in ("raw", "ratio", "ratio_meta"):
            for clipping in ("unclipped", "clipped"):
                rho = eligible[f"rho_error_{estimator}_{clipping}"]
                distance = eligible[f"distance_error_{estimator}_{clipping}"]
                rows.append({
                    "query_id": str(query_id),
                    "split": str(group["split"].iloc[0]),
                    "estimator": estimator,
                    "clipping": clipping,
                    "records": int(len(group)),
                    "eligible_records": int(len(eligible)),
                    "fallback_fraction": float(1.0 - len(eligible) / len(group)),
                    "rho_bias": float(rho.mean()) if len(rho) else math.nan,
                    "rho_mae": float(rho.abs().mean()) if len(rho) else math.nan,
                    "distance_bias": float(distance.mean()) if len(distance) else math.nan,
                    "distance_mae": float(distance.abs().mean()) if len(distance) else math.nan,
                    "distance_p99_abs_error": float(distance.abs().quantile(0.99)) if len(distance) else math.nan,
                })
    return pd.DataFrame(rows)


def error_summary_for_group(group: pd.DataFrame, label: object) -> list[dict[str, object]]:
    rows = []
    for estimator in ("raw", "ratio", "ratio_meta"):
        for clipping in ("unclipped", "clipped"):
            values = group[f"distance_error_{estimator}_{clipping}"]
            rows.append({
                "bin": str(label),
                "estimator": estimator,
                "clipping": clipping,
                "count": int(len(values)),
                "distance_bias": float(values.mean()) if len(values) else math.nan,
                "distance_mae": float(values.abs().mean()) if len(values) else math.nan,
                "distance_rmse": float(np.sqrt(np.mean(np.square(values)))) if len(values) else math.nan,
                "distance_p95_abs_error": float(values.abs().quantile(0.95)) if len(values) else math.nan,
                "distance_p99_abs_error": float(values.abs().quantile(0.99)) if len(values) else math.nan,
            })
    return rows


def conditional_deciles(data: pd.DataFrame, column: str) -> pd.DataFrame:
    train = data[(data["split"] == "train") & data["estimator_eligible"]].copy()
    if train.empty:
        return pd.DataFrame()
    try:
        train["condition_bin"] = pd.qcut(train[column], q=10, duplicates="drop")
    except ValueError:
        train["condition_bin"] = "all"
    rows: list[dict[str, object]] = []
    for label, group in train.groupby("condition_bin", observed=True, sort=True):
        group_rows = error_summary_for_group(group, label)
        for row in group_rows:
            row["condition"] = column
            row["bin_min"] = float(group[column].min())
            row["bin_max"] = float(group[column].max())
        rows.extend(group_rows)
    return pd.DataFrame(rows)


def conditional_margin(data: pd.DataFrame) -> pd.DataFrame:
    train = data[(data["split"] == "train") & data["estimator_eligible"]]
    rows: list[dict[str, object]] = []
    definitions = [
        ("all", np.ones(len(train), dtype=bool)),
        ("oracle_margin_positive", train["oracle_margin"] > 0.0),
        ("oracle_margin_non_positive", train["oracle_margin"] <= 0.0),
    ]
    definitions.extend(
        (f"relative_abs_margin_le_{limit:g}", train["relative_abs_margin"] <= limit)
        for limit in (1e-4, 1e-3, 1e-2, 1e-1)
    )
    for label, mask in definitions:
        group = train.loc[mask]
        group_rows = error_summary_for_group(group, label)
        for row in group_rows:
            row["condition"] = "oracle_margin"
            row["margin_min"] = float(group["oracle_margin"].min()) if len(group) else math.nan
            row["margin_max"] = float(group["oracle_margin"].max()) if len(group) else math.nan
        rows.extend(group_rows)
    return pd.DataFrame(rows)


def make_figures(data: pd.DataFrame, output_dir: Path) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    train = data[(data["split"] == "train") & data["estimator_eligible"]].copy()
    if train.empty:
        return

    def save_hist(raw_values: pd.Series, ratio_values: pd.Series, title: str, xlabel: str, name: str) -> None:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        combined = pd.concat([raw_values, ratio_values]).replace([np.inf, -np.inf], np.nan).dropna()
        if not combined.empty:
            low, high = combined.quantile([0.005, 0.995])
            if low == high:
                low, high = float(low) - 1.0, float(high) + 1.0
            bins = np.linspace(float(low), float(high), 100)
            ax.hist(raw_values.clip(low, high), bins=bins, alpha=0.55, label="raw", density=True)
            ax.hist(ratio_values.clip(low, high), bins=bins, alpha=0.55, label="ratio", density=True)
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / name, dpi=180)
        plt.close(fig)

    save_hist(
        train["rho_error_raw_clipped"], train["rho_error_ratio_clipped"],
        "Projection error (train, clipped)", "estimated rho - true rho", "rho_error_histogram.png",
    )
    save_hist(
        train["distance_error_raw_clipped"], train["distance_error_ratio_clipped"],
        "Distance error (train, clipped)", "estimated D - true D", "distance_error_histogram.png",
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    labels = ["raw rho", "ratio rho", "raw D", "ratio D"]
    values = [
        train["rho_error_raw_clipped"].abs().mean(),
        train["rho_error_ratio_clipped"].abs().mean(),
        train["distance_error_raw_clipped"].abs().mean(),
        train["distance_error_ratio_clipped"].abs().mean(),
    ]
    ax.bar(labels, values, color=["#6baed6", "#2171b5", "#fdae6b", "#e6550d"])
    ax.set_ylabel("Mean absolute error")
    ax.set_title("Raw versus ratio estimator MAE")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figures / "raw_vs_ratio_mae.png", dpi=180)
    plt.close(fig)

    kappa_table = conditional_deciles(data, "kappa_actual")
    ratio_kappa = kappa_table[(kappa_table.get("estimator") == "ratio") & (kappa_table.get("clipping") == "clipped")]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    if not ratio_kappa.empty:
        centers = (ratio_kappa["bin_min"] + ratio_kappa["bin_max"]) / 2.0
        ax.plot(centers, ratio_kappa["distance_bias"], marker="o")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("kappa (decile center)")
    ax.set_ylabel("Mean distance error")
    ax.set_title("Ratio estimator bias versus kappa")
    fig.tight_layout()
    fig.savefig(figures / "bias_vs_kappa.png", dpi=180)
    plt.close(fig)

    sample = train if len(train) <= 100_000 else train.sample(100_000, random_state=42)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.hexbin(sample["oracle_margin"], sample["distance_error_ratio_clipped"], gridsize=65, bins="log", mincnt=1, cmap="viridis")
    ax.axhline(0.0, color="white", linewidth=1.0)
    ax.axvline(0.0, color="white", linewidth=1.0)
    ax.set_xlabel("Oracle margin D - threshold")
    ax.set_ylabel("Ratio distance error")
    ax.set_title("Ratio bias versus pruning margin")
    fig.tight_layout()
    fig.savefig(figures / "bias_vs_margin.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.hexbin(sample["shadow_exact_squared_distance"], sample["distance_hat_ratio_clipped"], gridsize=65, bins="log", mincnt=1, cmap="magma")
    low = float(min(sample["shadow_exact_squared_distance"].min(), sample["distance_hat_ratio_clipped"].min()))
    high = float(max(sample["shadow_exact_squared_distance"].max(), sample["distance_hat_ratio_clipped"].max()))
    ax.plot([low, high], [low, high], color="white", linewidth=1.0)
    ax.set_xlabel("True squared distance")
    ax.set_ylabel("Ratio-estimated squared distance")
    ax.set_title("Predicted versus true distance")
    fig.tight_layout()
    fig.savefig(figures / "predicted_vs_true_distance.png", dpi=180)
    plt.close(fig)


def build_decision(data: pd.DataFrame, selected_kappa_min: float, config: AnalysisConfig) -> tuple[str, list[str], dict[str, float | int | None]]:
    train_valid = data[(data["split"] == "train") & data["analysis_valid"]]
    train = data[(data["split"] == "train") & data["estimator_eligible"]]
    closure_failures = int((~data["distance_closure_valid"] & data["diagnostic_valid"]).sum())
    eligible_fraction = float(len(train) / len(train_valid)) if len(train_valid) else 0.0
    raw_bias = float(train["distance_error_raw_clipped"].mean()) if len(train) else math.nan
    ratio_bias = float(train["distance_error_ratio_clipped"].mean()) if len(train) else math.nan
    raw_mae = float(train["distance_error_raw_clipped"].abs().mean()) if len(train) else math.nan
    ratio_mae = float(train["distance_error_ratio_clipped"].abs().mean()) if len(train) else math.nan
    near = train[train["relative_abs_margin"] <= 1e-2]
    near_raw_mae = float(near["distance_error_raw_clipped"].abs().mean()) if len(near) else math.nan
    near_ratio_mae = float(near["distance_error_ratio_clipped"].abs().mean()) if len(near) else math.nan
    near_raw_p99 = float(near["distance_error_raw_clipped"].abs().quantile(0.99)) if len(near) else math.nan
    near_ratio_p99 = float(near["distance_error_ratio_clipped"].abs().quantile(0.99)) if len(near) else math.nan
    kappa_meta_max_error = float(train_valid["kappa_meta_abs_error"].max()) if len(train_valid) else math.nan
    diagnostics: dict[str, float | int | None] = {
        "selected_kappa_min": selected_kappa_min,
        "train_analysis_valid_records": int(len(train_valid)),
        "train_estimator_eligible_records": int(len(train)),
        "eligible_fraction": eligible_fraction,
        "raw_clipped_distance_bias": raw_bias,
        "ratio_clipped_distance_bias": ratio_bias,
        "raw_clipped_distance_mae": raw_mae,
        "ratio_clipped_distance_mae": ratio_mae,
        "near_threshold_records_rel_1e_2": int(len(near)),
        "near_raw_mae": near_raw_mae,
        "near_ratio_mae": near_ratio_mae,
        "near_raw_p99_abs_error": near_raw_p99,
        "near_ratio_p99_abs_error": near_ratio_p99,
        "kappa_meta_max_abs_error": kappa_meta_max_error,
        "distance_closure_failures": closure_failures,
    }
    reasons: list[str] = []
    if closure_failures:
        reasons.append("exact_distance_closure_failure")
    if not len(train):
        reasons.append("no_train_estimator_records")
    if eligible_fraction < config.minimum_eligible_fraction:
        reasons.append("eligible_fraction_below_gate")
    if len(train) and not abs(ratio_bias) < abs(raw_bias):
        reasons.append("ratio_did_not_reduce_absolute_bias")
    if len(near) and near_ratio_mae > near_raw_mae and near_ratio_p99 > near_raw_p99:
        reasons.append("ratio_worsened_near_threshold_mae_and_tail")
    if math.isfinite(kappa_meta_max_error) and kappa_meta_max_error > 1e-9:
        reasons.append("metadata_kappa_mismatch")

    if config.mode == "smoke":
        return "SMOKE_ONLY", reasons, diagnostics
    return ("GO_TO_PHASE2" if not reasons else "NO_GO_RATIO_ESTIMATOR"), reasons, diagnostics


def write_report(output_dir: Path, summary: Mapping[str, object]) -> None:
    decision = str(summary["decision"])
    quality = str(summary["status"])
    counts = summary["counts"]
    gate = summary["gate_diagnostics"]
    reasons = summary.get("decision_reasons", [])
    reason_text = "none" if not reasons else ", ".join(f"`{reason}`" for reason in reasons)
    report = f"""# Phase 1 ratio-corrected PQ estimator diagnostic

- Run status: **{quality}**
- Decision: **{decision}**
- Mode: `{summary['mode']}`
- Input records / queries: `{counts['records']}` / `{counts['queries']}`
- Analysis-valid records: `{counts['analysis_valid_records']}`
- Exact-distance closure failures: `{counts['distance_closure_failures']}`
- Selected `kappa_min`: `{gate['selected_kappa_min']}`
- Train eligible fraction: `{gate['eligible_fraction']:.8%}`
- Raw/ratio clipped distance bias: `{gate['raw_clipped_distance_bias']}` / `{gate['ratio_clipped_distance_bias']}`
- Raw/ratio clipped distance MAE: `{gate['raw_clipped_distance_mae']}` / `{gate['ratio_clipped_distance_mae']}`
- Gate reasons: {reason_text}

The query split is frozen in `query_split_manifest.json`.  Kappa selection and
all Phase-1 gate metrics use the train split only; calibration and test query
IDs are not used to tune the estimator.  `SMOKE_ONLY` is never permission to
enter Phase 2.  Formal mode fails closed when any exact-distance closure check
violates the specified mixed tolerance.
"""
    (output_dir / "PHASE1_ESTIMATOR_DIAGNOSTIC_REPORT.md").write_text(report, encoding="utf-8")


def analyze(input_path: Path, output_dir: Path, config: AnalysisConfig) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_path.resolve()
    input_sha = sha256_file(input_path)
    raw = pd.read_csv(input_path, dtype="string", keep_default_na=False)

    try:
        normalized, schema_audit = audit_and_normalize_schema(raw)
    except SchemaError as exc:
        # Re-run enough of the audit to preserve a useful fail-closed artifact.
        missing = sorted(REQUIRED_DIRECT_COLUMNS - set(raw.columns))
        alias_missing = [canonical for canonical, aliases in CANONICAL_ALIASES.items() if not any(alias in raw.columns for alias in aliases)]
        schema_audit = {
            "format": "v0_ratio_estimator_schema_audit",
            "format_version": SCHEMA_AUDIT_VERSION,
            "status": "fail",
            "input_columns": list(raw.columns),
            "missing_columns": sorted(set(missing + alias_missing)),
            "error": str(exc),
        }
        (output_dir / "schema_audit.json").write_text(json.dumps(schema_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise

    (output_dir / "schema_audit.json").write_text(json.dumps(schema_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    data = derive_estimators(normalized, config)

    split = create_query_split(data["query_id"].tolist(), config.seed)
    validate_split(split, data["query_id"].tolist())
    split_lookup = {query_id: name for name, ids in split.items() for query_id in ids}
    data["split"] = data["query_id"].map(split_lookup)
    if data["split"].isna().any():
        raise RuntimeError("internal split assignment failure")

    split_manifest = {
        "format": "v0_ratio_estimator_query_split",
        "format_version": SPLIT_MANIFEST_VERSION,
        "seed": config.seed,
        "split_unit": "query_id",
        "fractions": {"train": config.train_fraction, "calibration": config.calibration_fraction, "test": config.test_fraction},
        "input_path": str(input_path),
        "input_sha256": input_sha,
        "analyzer_version": ANALYZER_VERSION,
        "estimator_formula_version": estimator_core.ESTIMATOR_FORMULA_VERSION,
        "query_count": int(data["query_id"].nunique()),
        "query_ids": split,
    }
    (output_dir / "query_split_manifest.json").write_text(json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if config.selected_kappa_min is None:
        selected_kappa_min, kappa_scan = choose_kappa_min(data, config.kappa_min_candidates, config.minimum_eligible_fraction)
    else:
        selected_kappa_min = float(config.selected_kappa_min)
        _, kappa_scan = choose_kappa_min(data, config.kappa_min_candidates, config.minimum_eligible_fraction)
    data["ratio_fallback"] = ~data["analysis_valid"] | (data["kappa_actual"] < selected_kappa_min)
    data["estimator_eligible"] = ~data["ratio_fallback"]
    kappa_scan["selected"] = np.isclose(kappa_scan["kappa_min"], selected_kappa_min, rtol=0.0, atol=0.0)
    kappa_scan.to_csv(output_dir / "kappa_min_scan.csv", index=False)

    invalid_columns = [*IDENTIFIER_COLUMNS, "split", "diagnostic_valid", "analysis_valid", "invalid_reason", "closure_abs_error", "closure_tolerance", "kappa_actual", "kappa_meta"]
    data.loc[~data["analysis_valid"], invalid_columns].to_csv(output_dir / "invalid_records.csv", index=False)

    estimator_summary = summarize_estimators(data)
    estimator_summary.to_csv(output_dir / "estimator_record_summary.csv", index=False)
    query_summary(data).to_csv(output_dir / "estimator_query_summary.csv", index=False)
    conditional_deciles(data, "kappa_actual").to_csv(output_dir / "conditional_bias_by_kappa.csv", index=False)
    conditional_deciles(data, "rho_true").to_csv(output_dir / "conditional_bias_by_rho.csv", index=False)
    conditional_deciles(data, "edge_length").to_csv(output_dir / "conditional_bias_by_edge_length.csv", index=False)
    conditional_deciles(data, "x_norm").to_csv(output_dir / "conditional_bias_by_x_norm.csv", index=False)
    conditional_margin(data).to_csv(output_dir / "conditional_bias_by_margin.csv", index=False)

    if not config.skip_plots:
        make_figures(data, output_dir)

    decision, reasons, gate = build_decision(data, selected_kappa_min, config)
    closure_failures = int((~data["distance_closure_valid"] & data["diagnostic_valid"]).sum())
    status = "PASS" if (config.mode == "smoke" or closure_failures == 0) else "FAIL_CLOSED"
    summary: dict[str, object] = {
        "format": "v0_ratio_estimator_phase1_summary",
        "format_version": 1,
        "analyzer_version": ANALYZER_VERSION,
        "estimator_formula_version": estimator_core.ESTIMATOR_FORMULA_VERSION,
        "status": status,
        "mode": config.mode,
        "decision": decision,
        "decision_reasons": reasons,
        "counts": {
            "records": int(len(data)),
            "queries": int(data["query_id"].nunique()),
            "diagnostic_valid_records": int(data["diagnostic_valid"].sum()),
            "analysis_valid_records": int(data["analysis_valid"].sum()),
            "invalid_records": int((~data["analysis_valid"]).sum()),
            "distance_closure_failures": closure_failures,
            "projection_closure_failures": int((~data["projection_closure_valid"] & data["diagnostic_valid"]).sum()),
            "estimator_eligible_records": int(data["estimator_eligible"].sum()),
            "ratio_fallback_records": int(data["ratio_fallback"].sum()),
        },
        "tolerances": {
            "closure_absolute": config.closure_absolute_tolerance,
            "closure_relative": config.closure_relative_tolerance,
            "projection": config.projection_tolerance,
        },
        "gate_diagnostics": gate,
        "artifacts": {
            "schema_audit": "schema_audit.json",
            "query_split_manifest": "query_split_manifest.json",
            "record_summary": "estimator_record_summary.csv",
            "query_summary": "estimator_query_summary.csv",
            "invalid_records": "invalid_records.csv",
            "kappa_scan": "kappa_min_scan.csv",
            "report": "PHASE1_ESTIMATOR_DIAGNOSTIC_REPORT.md",
        },
    }
    summary = json_safe(summary)
    assert isinstance(summary, dict)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    run_manifest = {
        "format": "v0_ratio_estimator_phase1_run_manifest",
        "format_version": 1,
        "analyzer_version": ANALYZER_VERSION,
        "estimator_formula_version": estimator_core.ESTIMATOR_FORMULA_VERSION,
        "analyzer_path": str(Path(__file__).resolve()),
        "analyzer_sha256": sha256_file(Path(__file__).resolve()),
        "input_path": str(input_path),
        "input_sha256": input_sha,
        "output_dir": str(output_dir.resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "matplotlib_version": matplotlib.__version__,
        "config": {
            "mode": config.mode,
            "seed": config.seed,
            "kappa_min_candidates": list(config.kappa_min_candidates),
            "selected_kappa_min": selected_kappa_min,
            "minimum_eligible_fraction": config.minimum_eligible_fraction,
            "skip_plots": config.skip_plots,
            "closure_absolute_tolerance": config.closure_absolute_tolerance,
            "closure_relative_tolerance": config.closure_relative_tolerance,
        },
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(output_dir, summary)
    return summary


def parse_kappa_candidates(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("kappa candidates must be comma-separated floats") from exc
    if not values or any(not math.isfinite(item) or item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("kappa candidates must be finite and positive")
    return tuple(sorted(set(values)))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kappa-min-candidates", type=parse_kappa_candidates, default=DEFAULT_KAPPA_MIN_CANDIDATES)
    parser.add_argument("--selected-kappa-min", type=float)
    parser.add_argument("--minimum-eligible-fraction", type=float, default=0.99)
    parser.add_argument("--closure-absolute-tolerance", type=float, default=1e-10)
    parser.add_argument("--closure-relative-tolerance", type=float, default=1e-9)
    parser.add_argument("--projection-tolerance", type=float, default=1e-9)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = AnalysisConfig(
        mode=args.mode,
        seed=args.seed,
        closure_absolute_tolerance=args.closure_absolute_tolerance,
        closure_relative_tolerance=args.closure_relative_tolerance,
        projection_tolerance=args.projection_tolerance,
        kappa_min_candidates=args.kappa_min_candidates,
        selected_kappa_min=args.selected_kappa_min,
        minimum_eligible_fraction=args.minimum_eligible_fraction,
        skip_plots=args.skip_plots,
    )
    try:
        summary = analyze(args.input, args.output_dir, config)
    except (OSError, ValueError, SchemaError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
