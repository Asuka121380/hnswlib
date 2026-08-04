#!/usr/bin/env python3
"""Shared split-conformal utilities for V0 probabilistic lower bounds."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import beta

import ratio_estimator_core as estimator_core


PHASE2_VERSION = "v0_ratio_probabilistic_bound_phase2_v1"
CALIBRATOR_FORMAT = "v0_ratio_probabilistic_calibrator"
CALIBRATOR_FORMAT_VERSION = 1
CALIBRATION_MANIFEST_FORMAT = "v0_ratio_probabilistic_calibration_manifest"
CALIBRATION_MANIFEST_VERSION = 1
DEFAULT_ALPHAS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)


@dataclass(frozen=True)
class RuntimeEstimatorConfig:
    closure_absolute_tolerance: float = 1e-10
    closure_relative_tolerance: float = 1e-9
    projection_tolerance: float = 1e-9


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(estimator_core.json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def parse_alphas(value: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        try:
            parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
        except ValueError as exc:
            raise ValueError("alphas must be comma-separated floats") from exc
    else:
        parsed = [float(item) for item in value]
    if not parsed or any(not math.isfinite(item) or not 0.0 < item < 1.0 for item in parsed):
        raise ValueError("every alpha must be finite and in (0, 1)")
    return tuple(sorted(set(parsed), reverse=True))


def alpha_label(alpha: float) -> str:
    text = f"{float(alpha):.0e}"
    mantissa, exponent = text.split("e")
    exponent_value = int(exponent)
    return f"{mantissa}e{exponent_value:+d}".replace("+", "")


def conformal_order_statistic(scores: Iterable[float], alpha: float) -> dict[str, object]:
    values = np.asarray(list(scores), dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("calibration scores must be a finite one-dimensional sequence")
    m = int(values.size)
    j = int(math.ceil((m + 1) * (1.0 - float(alpha))))
    supported = m > 0 and j <= m
    quantile = float(np.sort(values, kind="mergesort")[j - 1]) if supported else None
    return {
        "status": "supported" if supported else "unsupported_sample_size",
        "sample_count": m,
        "order_statistic_index_one_based": j,
        "minimum_required_sample_count": int(math.ceil(1.0 / float(alpha)) - 1),
        "quantile": quantile,
        "quantile_decimal": None if quantile is None else repr(quantile),
        "quantile_hex": None if quantile is None else float(quantile).hex(),
    }


def query_max_scores(
    eligible: pd.DataFrame,
    all_query_ids: Sequence[str],
    score_column: str,
) -> pd.Series:
    if score_column not in eligible.columns:
        raise ValueError(f"missing score column: {score_column}")
    grouped = eligible.groupby("query_id", sort=False)[score_column].max()
    ordered_ids = [str(value) for value in all_query_ids]
    # A query with no probabilistic evaluations has simultaneous score -inf;
    # it contributes no interval risk. In expected V0 data every query has
    # eligible records, but preserving it keeps query-level denominators honest.
    return grouped.reindex(ordered_ids, fill_value=-np.inf).astype(float)


def finite_query_scores(scores: pd.Series) -> np.ndarray:
    values = scores.to_numpy(dtype=np.float64)
    # -inf means no probabilistic record for that query. It is smaller than
    # every finite score and cannot influence an upper order statistic unless
    # the calibrator is already uninformative. Replace with the smallest finite
    # float to retain the query in the finite-sample count and JSON-safe flow.
    return np.where(np.isneginf(values), -np.finfo(np.float64).max, values)


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials")
    if trials == 0:
        return math.nan, math.nan
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if successes == 0 else float(beta.ppf(tail, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1.0 - tail, successes + 1, trials - successes))
    return lower, upper


def stable_score_sha256(frame: pd.DataFrame, score_column: str) -> str:
    columns = ["query_id", "current_node_id", "candidate_id", score_column]
    ordered = frame[columns].copy()
    for name in ("query_id", "current_node_id", "candidate_id"):
        ordered[name] = ordered[name].astype(str)
    ordered[score_column] = ordered[score_column].astype(float).map(float.hex)
    ordered = ordered.sort_values(["query_id", "current_node_id", "candidate_id", score_column], kind="mergesort")
    return estimator_core.sha256_text(ordered.to_csv(index=False, lineterminator="\n"))


def assign_frozen_split(data: pd.DataFrame, split_manifest: Mapping[str, object]) -> pd.DataFrame:
    query_ids_obj = split_manifest.get("query_ids")
    if not isinstance(query_ids_obj, Mapping):
        raise ValueError("split manifest has no query_ids object")
    split_sets: dict[str, set[str]] = {}
    lookup: dict[str, str] = {}
    for name in ("train", "calibration", "test"):
        raw_ids = query_ids_obj.get(name)
        if not isinstance(raw_ids, list):
            raise ValueError(f"split manifest has no {name} query list")
        ids = {str(value) for value in raw_ids}
        split_sets[name] = ids
        for query_id in ids:
            if query_id in lookup:
                raise ValueError(f"query split leakage for query_id={query_id}")
            lookup[query_id] = name
    result = data.copy()
    result["split"] = result["query_id"].astype(str).map(lookup)
    if result["split"].isna().any():
        missing = sorted(set(result.loc[result["split"].isna(), "query_id"].astype(str)))
        raise ValueError(f"input contains queries absent from frozen split: {missing[:10]}")
    return result


def load_phase1_contract(input_path: Path, phase1_dir: Path) -> dict[str, object]:
    input_path = input_path.resolve()
    phase1_dir = phase1_dir.resolve()
    split_path = phase1_dir / "query_split_manifest.json"
    run_path = phase1_dir / "run_manifest.json"
    summary_path = phase1_dir / "summary.json"
    for path in (input_path, split_path, run_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    input_sha = estimator_core.sha256_file(input_path)
    split = read_json(split_path)
    run = read_json(run_path)
    summary = read_json(summary_path)
    expected_hashes = {
        str(split.get("input_sha256", "")),
        str(run.get("input_sha256", "")),
    }
    if expected_hashes != {input_sha}:
        raise ValueError(
            f"Phase-1 input SHA mismatch: actual={input_sha}, expected={sorted(expected_hashes)}"
        )
    if summary.get("status") != "PASS" or summary.get("decision") != "GO_TO_PHASE2":
        raise ValueError("Phase 1 is not frozen as PASS / GO_TO_PHASE2")

    config_obj = run.get("config")
    if not isinstance(config_obj, Mapping):
        raise ValueError("Phase-1 run manifest has no config")
    selected_kappa = config_obj.get("selected_kappa_min")
    if selected_kappa is None:
        gate = summary.get("gate_diagnostics")
        selected_kappa = gate.get("selected_kappa_min") if isinstance(gate, Mapping) else None
    if selected_kappa is None or not math.isfinite(float(selected_kappa)):
        raise ValueError("Phase-1 selected kappa_min is missing")

    formula_version = run.get("estimator_formula_version")
    if formula_version is None:
        # Frozen Phase-1 v1 artifacts predate the explicit field. The analyzer
        # version uniquely identifies the formula now centralized in core.
        if run.get("analyzer_version") != "v0_ratio_estimator_phase1_v1":
            raise ValueError("cannot infer estimator formula version")
        formula_version = estimator_core.ESTIMATOR_FORMULA_VERSION
    if formula_version != estimator_core.ESTIMATOR_FORMULA_VERSION:
        raise ValueError(
            f"formula version mismatch: Phase1={formula_version}, code={estimator_core.ESTIMATOR_FORMULA_VERSION}"
        )

    tolerances = summary.get("tolerances")
    if not isinstance(tolerances, Mapping):
        raise ValueError("Phase-1 summary has no tolerances")
    runtime_config = RuntimeEstimatorConfig(
        closure_absolute_tolerance=float(tolerances.get("closure_absolute", 1e-10)),
        closure_relative_tolerance=float(tolerances.get("closure_relative", 1e-9)),
        projection_tolerance=float(tolerances.get("projection", 1e-9)),
    )
    return {
        "input_path": input_path,
        "input_sha256": input_sha,
        "phase1_dir": phase1_dir,
        "phase1_run_manifest": run,
        "phase1_summary": summary,
        "query_split_manifest": split,
        "query_split_manifest_sha256": estimator_core.sha256_file(split_path),
        "phase1_run_manifest_sha256": estimator_core.sha256_file(run_path),
        "phase1_summary_sha256": estimator_core.sha256_file(summary_path),
        "kappa_min": float(selected_kappa),
        "estimator_formula_version": str(formula_version),
        "runtime_config": runtime_config,
    }


def load_derived_input(
    contract: Mapping[str, object],
    required_split: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    input_path = contract["input_path"]
    if not isinstance(input_path, Path):
        raise TypeError("contract input_path must be Path")
    raw = pd.read_csv(input_path, dtype="string", keep_default_na=False)
    if required_split is not None:
        if required_split not in {"train", "calibration", "test"}:
            raise ValueError(f"unknown frozen split: {required_split}")
        split = contract["query_split_manifest"]
        query_ids_obj = split.get("query_ids") if isinstance(split, Mapping) else None
        if not isinstance(query_ids_obj, Mapping):
            raise ValueError("query split manifest has no query_ids mapping")
        required_ids_obj = query_ids_obj.get(required_split)
        if not isinstance(required_ids_obj, list):
            raise ValueError(f"query split manifest has no {required_split} query list")
        required_ids = {str(value) for value in required_ids_obj}
        if "query_id" not in raw.columns:
            raise ValueError("input schema is missing required column: query_id")
        # Discard sealed splits before schema normalization or estimator math.
        # The source CSV still has to be scanned to select rows, but calibration
        # values from train/test never enter the calibration computation.
        raw = raw[raw["query_id"].astype(str).isin(required_ids)].copy()
        if raw.empty:
            raise ValueError(f"frozen {required_split} split has no records")
    normalized, audit = estimator_core.audit_and_normalize_schema(raw)
    runtime_config = contract["runtime_config"]
    data = estimator_core.derive_estimators(normalized, runtime_config)  # type: ignore[arg-type]
    split = contract["query_split_manifest"]
    data = assign_frozen_split(data, split)  # type: ignore[arg-type]
    if required_split is not None and not data["split"].eq(required_split).all():
        raise AssertionError(f"records outside frozen {required_split} split were loaded")
    kappa_min = float(contract["kappa_min"])
    data["ratio_eligible"] = (
        data["analysis_valid"]
        & np.isfinite(data["kappa_meta"])
        & (data["kappa_meta"] >= kappa_min)
    )
    data["current_lb_available"] = np.isfinite(data["current_lb"]) & (data["current_lb"] >= 0.0)
    data["current_lb_safe"] = data["current_lb"].where(data["current_lb_available"], 0.0).clip(lower=0.0)
    fallback_reason = pd.Series("", index=data.index, dtype="string")
    fallback_reason.loc[~data["analysis_valid"]] = "analysis_invalid"
    fallback_reason.loc[data["analysis_valid"] & (data["kappa_meta"] < kappa_min)] = "kappa_below_min"
    fallback_reason.loc[~data["current_lb_available"]] = fallback_reason.loc[~data["current_lb_available"]].where(
        fallback_reason.loc[~data["current_lb_available"]].ne(""), "current_lb_unavailable"
    )
    data["ratio_fallback_reason"] = fallback_reason
    distance_column, error_column = estimator_core.deployment_columns()
    data["deployment_distance_hat"] = data[distance_column]
    data["deployment_score"] = data[error_column]
    return data, audit


def apply_calibrator_to_frame(data: pd.DataFrame, quantile: float) -> pd.DataFrame:
    if not math.isfinite(float(quantile)):
        raise ValueError("quantile must be finite")
    result = data.copy()
    ratio_lb = np.maximum(0.0, result["deployment_distance_hat"].to_numpy(float) - float(quantile))
    result["ratio_lb"] = ratio_lb
    result["probabilistic_lb"] = np.where(
        result["ratio_eligible"].to_numpy(bool),
        ratio_lb,
        result["current_lb_safe"].to_numpy(float),
    )
    result["used_current_lb_fallback"] = ~result["ratio_eligible"]
    result["interval_violation"] = (
        result["probabilistic_lb"] > result["shadow_exact_squared_distance"]
    )
    result["violation_magnitude"] = np.maximum(
        0.0,
        result["probabilistic_lb"] - result["shadow_exact_squared_distance"],
    )
    if (result["probabilistic_lb"] < 0.0).any():
        raise AssertionError("probabilistic lower bound became negative")
    return result


def validate_calibrator(calibrator: Mapping[str, object], contract: Mapping[str, object]) -> None:
    if calibrator.get("format") != CALIBRATOR_FORMAT or calibrator.get("format_version") != CALIBRATOR_FORMAT_VERSION:
        raise ValueError("unsupported calibrator format")
    if calibrator.get("input_sha256") != contract.get("input_sha256"):
        raise ValueError("calibrator/input SHA mismatch")
    if calibrator.get("query_split_manifest_sha256") != contract.get("query_split_manifest_sha256"):
        raise ValueError("calibrator/query split SHA mismatch")
    if calibrator.get("estimator_formula_version") != contract.get("estimator_formula_version"):
        raise ValueError("calibrator/formula version mismatch")
    if float(calibrator.get("kappa_min", math.nan)) != float(contract.get("kappa_min", math.nan)):
        raise ValueError("calibrator/kappa_min mismatch")
