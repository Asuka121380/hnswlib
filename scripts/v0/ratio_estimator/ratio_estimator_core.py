#!/usr/bin/env python3
"""Shared schema and estimator implementation for V0 ratio diagnostics.

Phase 1, calibration, offline simulation, and the future C++ parity harness
must all use the formula version declared here.  Keeping the estimator in one
module prevents silent formula, clipping, and validity-policy drift.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np
import pandas as pd


ESTIMATOR_FORMULA_VERSION = "v0_ratio_corrected_pq_distance_v1"
DEPLOYMENT_ESTIMATOR = "ratio_meta"
DEPLOYMENT_CLIPPING = "clipped"
FALLBACK_POLICY = "probabilistic_estimator_unavailable_use_current_lb"

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


class EstimatorConfig(Protocol):
    closure_absolute_tolerance: float
    closure_relative_tolerance: float
    projection_tolerance: float


class SchemaError(ValueError):
    """Raised when an input cannot be interpreted without guessing."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def canonical_json(value: object) -> str:
    return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


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
    coexisting_alias_differences: dict[str, dict[str, float | int | None]] = {}
    normalized = raw.copy()

    for canonical, aliases in CANONICAL_ALIASES.items():
        present = [name for name in aliases if name in columns]
        if not present:
            missing.append(canonical)
            continue
        selected = present[0]
        alias_resolution[canonical] = selected
        if len(present) > 1:
            left = pd.to_numeric(raw[present[0]], errors="coerce").to_numpy(float)
            for other_name in present[1:]:
                right = pd.to_numeric(raw[other_name], errors="coerce").to_numpy(float)
                finite = np.isfinite(left) & np.isfinite(right)
                difference = np.abs(left - right)
                coexisting_alias_differences[f"{present[0]}_vs_{other_name}"] = {
                    "finite_comparison_count": int(finite.sum()),
                    "different_count": int((finite & (difference != 0.0)).sum()),
                    "max_abs_difference": float(difference[finite].max()) if finite.any() else None,
                }
        normalized[canonical] = raw[selected]

    audit: dict[str, object] = {
        "format": "v0_ratio_estimator_schema_audit",
        "format_version": 1,
        "status": "pass" if not missing else "fail",
        "input_columns": columns,
        "required_direct_columns": sorted(REQUIRED_DIRECT_COLUMNS),
        "canonical_aliases": {key: list(value) for key, value in CANONICAL_ALIASES.items()},
        "alias_resolution": alias_resolution,
        "missing_columns": sorted(set(missing)),
        "coexisting_source_differences": coexisting_alias_differences,
    }
    if missing:
        raise SchemaError("missing columns: " + ", ".join(sorted(set(missing))))
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


def derive_estimators(frame: pd.DataFrame, config: EstimatorConfig) -> pd.DataFrame:
    """Compute every estimator column and the shared fail-closed validity mask."""
    result, conversion_failure = numeric_frame(frame)
    reason = pd.Series("", index=result.index, dtype="string")
    append_reason(reason, ~result["diagnostic_valid"], "diagnostic_valid_false")
    append_reason(reason, conversion_failure, "non_finite_or_non_numeric_input")
    append_reason(reason, result["x_norm"] <= 0.0, "non_positive_x_norm")
    append_reason(reason, result["reconstruction_norm"] <= 0.0, "non_positive_reconstruction_norm")

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        n = result["x_norm"]
        ell = result["edge_length"]
        s = result["reconstruction_norm"]
        x_dot_r = result["x_dot_r"]
        x_dot_u = result["x_dot_true_direction"]
        e_actual = result["actual_direction_error"]
        e_meta = result["direction_error"]

        result["rho_true"] = x_dot_u / n
        result["rho_hat_raw_unclipped"] = x_dot_r / (n * s)
        result["kappa_actual"] = (1.0 + s * s - e_actual * e_actual) / (2.0 * s)
        result["kappa_meta"] = (1.0 + s * s - e_meta * e_meta) / (2.0 * s)
        result["rho_hat_ratio_unclipped"] = result["rho_hat_raw_unclipped"] / result["kappa_actual"]
        result["rho_hat_ratio_meta_unclipped"] = result["rho_hat_raw_unclipped"] / result["kappa_meta"]

        for prefix in ("raw", "ratio", "ratio_meta"):
            source = f"rho_hat_{prefix}_unclipped"
            result[f"rho_hat_{prefix}_clipped"] = result[source].clip(-1.0, 1.0)

        result["distance_reconstructed"] = n * n + ell * ell - 2.0 * ell * x_dot_u
        for prefix in ("raw", "ratio", "ratio_meta"):
            for clipping in ("unclipped", "clipped"):
                rho_name = f"rho_hat_{prefix}_{clipping}"
                result[f"distance_hat_{prefix}_{clipping}"] = n * n + ell * ell - 2.0 * n * ell * result[rho_name]

    result["closure_abs_error"] = (
        result["distance_reconstructed"] - result["shadow_exact_squared_distance"]
    ).abs()
    result["closure_tolerance"] = (
        config.closure_absolute_tolerance
        + config.closure_relative_tolerance
        * np.maximum(
            result["distance_reconstructed"].abs(),
            result["shadow_exact_squared_distance"].abs(),
        )
    )
    result["distance_closure_valid"] = result["closure_abs_error"] <= result["closure_tolerance"]
    result["projection_closure_valid"] = result["rho_true"].abs() <= (1.0 + config.projection_tolerance)
    append_reason(reason, result["kappa_actual"] <= 0.0, "non_positive_kappa_actual")
    append_reason(reason, ~np.isfinite(result["kappa_actual"]), "non_finite_kappa_actual")
    append_reason(reason, ~result["distance_closure_valid"], "exact_distance_closure_failure")
    append_reason(reason, ~result["projection_closure_valid"], "true_projection_outside_unit_interval")

    derived_columns = [
        name for name in result.columns
        if name.startswith(("rho_", "kappa_", "distance_hat_", "distance_reconstructed"))
    ]
    derived_finite = np.isfinite(result[derived_columns].to_numpy(dtype=float)).all(axis=1)
    append_reason(reason, ~derived_finite, "non_finite_derived_value")

    result["invalid_reason"] = reason
    result["analysis_valid"] = reason.eq("")
    result["oracle_margin"] = result["shadow_exact_squared_distance"] - result["threshold"]
    result["relative_abs_margin"] = result["oracle_margin"].abs() / np.maximum(
        result["threshold"].abs(), 1e-12
    )
    result["kappa_meta_abs_error"] = (result["kappa_actual"] - result["kappa_meta"]).abs()

    for prefix in ("raw", "ratio", "ratio_meta"):
        for clipping in ("unclipped", "clipped"):
            rho_hat = result[f"rho_hat_{prefix}_{clipping}"]
            distance_hat = result[f"distance_hat_{prefix}_{clipping}"]
            result[f"rho_error_{prefix}_{clipping}"] = rho_hat - result["rho_true"]
            result[f"distance_error_{prefix}_{clipping}"] = (
                distance_hat - result["shadow_exact_squared_distance"]
            )
    result["raw_rho_out_of_range"] = result["rho_hat_raw_unclipped"].abs() > 1.0
    result["ratio_rho_out_of_range"] = result["rho_hat_ratio_unclipped"].abs() > 1.0
    result["ratio_meta_rho_out_of_range"] = result["rho_hat_ratio_meta_unclipped"].abs() > 1.0
    return result


def deployment_columns(estimator: str = DEPLOYMENT_ESTIMATOR, clipping: str = DEPLOYMENT_CLIPPING) -> tuple[str, str]:
    if estimator not in {"raw", "ratio", "ratio_meta"}:
        raise ValueError(f"unsupported estimator: {estimator}")
    if clipping not in {"clipped", "unclipped"}:
        raise ValueError(f"unsupported clipping policy: {clipping}")
    return f"distance_hat_{estimator}_{clipping}", f"distance_error_{estimator}_{clipping}"
