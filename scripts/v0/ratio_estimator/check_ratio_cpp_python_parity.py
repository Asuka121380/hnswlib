#!/usr/bin/env python3
"""Recompute Phase-4 ratio shadow quantities and decisions in Python."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = (
    "kappa_meta",
    "rho_hat_raw",
    "rho_hat_ratio",
    "ratio_estimated_squared_distance",
    "ratio_lb",
    "ratio_effective_lb",
)


def check_parity(frame: pd.DataFrame, atol: float, rtol: float) -> dict:
    required = {
        "current_squared_distance", "threshold", "current_lb", "current_lb_valid",
        "edge_length", "direction_error", "reconstruction_norm",
        "x_dot_reconstruction", "kappa_meta", "ratio_eligible",
        "rho_hat_raw", "rho_hat_ratio", "ratio_estimated_squared_distance",
        "ratio_quantile", "ratio_lb", "ratio_effective_lb",
        "ratio_used_current_fallback", "ratio_would_prune",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing parity columns: {missing}")

    n = np.sqrt(frame["current_squared_distance"].to_numpy(float))
    ell = frame["edge_length"].to_numpy(float)
    err = frame["direction_error"].to_numpy(float)
    s = frame["reconstruction_norm"].to_numpy(float)
    dot = frame["x_dot_reconstruction"].to_numpy(float)
    q = frame["ratio_quantile"].to_numpy(float)
    current_lb = frame["current_lb"].to_numpy(float)
    current_valid = frame["current_lb_valid"].to_numpy(int).astype(bool)
    eligible = frame["ratio_eligible"].to_numpy(int).astype(bool)

    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = (1.0 + s * s - err * err) / (2.0 * s)
        raw = dot / (n * s)
        ratio_unclipped = raw / kappa
    ratio = np.clip(ratio_unclipped, -1.0, 1.0)
    distance = n * n + ell * ell - 2.0 * n * ell * ratio
    ratio_lb = np.maximum(0.0, distance - q)
    effective = np.where(eligible, ratio_lb, np.where(current_valid, current_lb, 0.0))
    used_fallback = (~eligible) & current_valid
    would_prune = (eligible | current_valid) & (
        effective > frame["threshold"].to_numpy(float))

    expected = {
        "kappa_meta": kappa,
        "rho_hat_raw": raw,
        "rho_hat_ratio": ratio,
        "ratio_estimated_squared_distance": distance,
        "ratio_lb": ratio_lb,
        "ratio_effective_lb": effective,
    }
    max_abs = {}
    mismatch_count = 0
    for name in NUMERIC_COLUMNS:
        actual = frame[name].to_numpy(float)
        finite = np.isfinite(actual) & np.isfinite(expected[name])
        close = np.zeros(len(frame), dtype=bool)
        close[finite] = np.isclose(
            actual[finite], expected[name][finite], atol=atol, rtol=rtol)
        # Ineligible records may intentionally carry zeroed estimator fields.
        if name != "ratio_effective_lb":
            close[~eligible] = True
        mismatch_count += int((~close).sum())
        if finite.any():
            max_abs[name] = float(np.max(np.abs(actual[finite] - expected[name][finite])))
        else:
            max_abs[name] = None

    boolean_checks = {
        "ratio_used_current_fallback": used_fallback,
        "ratio_would_prune": would_prune,
    }
    decision_mismatches = 0
    for name, expected_values in boolean_checks.items():
        actual = frame[name].to_numpy(int).astype(bool)
        decision_mismatches += int((actual != expected_values).sum())

    status = "PASS" if mismatch_count == 0 and decision_mismatches == 0 else "FAIL"
    return {
        "status": status,
        "record_count": int(len(frame)),
        "numeric_mismatch_count": mismatch_count,
        "decision_mismatch_count": decision_mismatches,
        "max_abs_difference": max_abs,
        "atol": atol,
        "rtol": rtol,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--rtol", type=float, default=1e-9)
    args = parser.parse_args()
    result = check_parity(pd.read_csv(args.input), args.atol, args.rtol)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
