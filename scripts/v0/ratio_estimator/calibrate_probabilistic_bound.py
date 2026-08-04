#!/usr/bin/env python3
"""Freeze record- and query-level split-conformal V0 calibrators.

This program consumes calibration queries only.  It never evaluates test-set
coverage and never makes a pruning decision.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import probabilistic_bound_core as bound_core
import ratio_estimator_core as estimator_core


def score_distribution(values: np.ndarray) -> dict[str, object]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else None,
        "min": float(finite.min()),
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
        "p99_9": float(np.quantile(finite, 0.999)),
        "max": float(finite.max()),
    }


def build_calibrator(
    *,
    calibration_level: str,
    alpha: float,
    scores: np.ndarray,
    contract: Mapping[str, object],
    calibration_query_ids: Sequence[str],
    calibration_scores_sha256: str,
) -> dict[str, object]:
    if calibration_level not in {"record", "query"}:
        raise ValueError("calibration_level must be record or query")
    statistic = bound_core.conformal_order_statistic(scores, alpha)
    calibrator_id = f"{calibration_level}_alpha_{bound_core.alpha_label(alpha)}"
    return {
        "format": bound_core.CALIBRATOR_FORMAT,
        "format_version": bound_core.CALIBRATOR_FORMAT_VERSION,
        "phase2_version": bound_core.PHASE2_VERSION,
        "calibrator_id": calibrator_id,
        "status": statistic["status"],
        "calibration_level": calibration_level,
        "nominal_alpha": float(alpha),
        "sample_count": statistic["sample_count"],
        "order_statistic_index_one_based": statistic["order_statistic_index_one_based"],
        "minimum_required_sample_count": statistic["minimum_required_sample_count"],
        "quantile": statistic["quantile"],
        "quantile_decimal": statistic["quantile_decimal"],
        "quantile_hex": statistic["quantile_hex"],
        "score_definition": (
            "distance_hat_ratio_meta_clipped_minus_geometric_squared_distance"
            if calibration_level == "record"
            else "per_query_max_distance_hat_ratio_meta_clipped_minus_geometric_squared_distance"
        ),
        "estimator_formula_version": contract["estimator_formula_version"],
        "estimator_variant": estimator_core.DEPLOYMENT_ESTIMATOR,
        "clipping_policy": estimator_core.DEPLOYMENT_CLIPPING,
        "numeric_precision": "float64",
        "kappa_min": contract["kappa_min"],
        "eligibility_kappa": "kappa_meta",
        "fallback_policy": estimator_core.FALLBACK_POLICY,
        "input_sha256": contract["input_sha256"],
        "query_split_manifest_sha256": contract["query_split_manifest_sha256"],
        "phase1_run_manifest_sha256": contract["phase1_run_manifest_sha256"],
        "phase1_summary_sha256": contract["phase1_summary_sha256"],
        "calibration_scores_sha256": calibration_scores_sha256,
        "calibration_query_ids": [str(value) for value in calibration_query_ids],
        "test_queries_read": False,
    }


def calibrate(
    input_path: Path,
    phase1_dir: Path,
    output_dir: Path,
    alphas: Sequence[float] = bound_core.DEFAULT_ALPHAS,
) -> dict[str, object]:
    alphas = bound_core.parse_alphas(alphas)
    contract = bound_core.load_phase1_contract(input_path, phase1_dir)
    data, schema_audit = bound_core.load_derived_input(contract, required_split="calibration")
    split = contract["query_split_manifest"]
    query_ids_obj = split["query_ids"]  # type: ignore[index]
    calibration_query_ids = [str(value) for value in query_ids_obj["calibration"]]  # type: ignore[index]
    calibration = data.copy()
    eligible = calibration[calibration["ratio_eligible"]].copy()
    if calibration.empty:
        raise ValueError("frozen calibration split has no records")
    if eligible.empty:
        raise ValueError("frozen calibration split has no ratio-eligible records")

    record_scores = eligible["deployment_score"].to_numpy(dtype=np.float64)
    query_scores_series = bound_core.query_max_scores(
        eligible, calibration_query_ids, "deployment_score"
    )
    query_scores = bound_core.finite_query_scores(query_scores_series)
    record_score_sha = bound_core.stable_score_sha256(eligible, "deployment_score")
    query_score_payload = [
        {"query_id": query_id, "score_hex": float(score).hex()}
        for query_id, score in zip(calibration_query_ids, query_scores, strict=True)
    ]
    query_score_sha = estimator_core.sha256_text(estimator_core.canonical_json(query_score_payload))

    output_dir.mkdir(parents=True, exist_ok=True)
    calibrator_dir = output_dir / "calibrators"
    calibrator_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    calibrator_artifacts: list[dict[str, object]] = []
    calibrators: list[dict[str, object]] = []

    for level, scores, score_sha in (
        ("record", record_scores, record_score_sha),
        ("query", query_scores, query_score_sha),
    ):
        distribution = score_distribution(scores)
        for alpha in alphas:
            calibrator = build_calibrator(
                calibration_level=level,
                alpha=alpha,
                scores=scores,
                contract=contract,
                calibration_query_ids=calibration_query_ids,
                calibration_scores_sha256=score_sha,
            )
            calibrators.append(calibrator)
            filename = f"{calibrator['calibrator_id']}.json"
            path = calibrator_dir / filename
            bound_core.write_json(path, calibrator)
            artifact = {
                "calibrator_id": calibrator["calibrator_id"],
                "path": f"calibrators/{filename}",
                "sha256": estimator_core.sha256_file(path),
                "status": calibrator["status"],
            }
            calibrator_artifacts.append(artifact)
            rows.append({
                "calibrator_id": calibrator["calibrator_id"],
                "calibration_level": level,
                "nominal_alpha": float(alpha),
                "status": calibrator["status"],
                "sample_count": calibrator["sample_count"],
                "order_statistic_index_one_based": calibrator["order_statistic_index_one_based"],
                "minimum_required_sample_count": calibrator["minimum_required_sample_count"],
                "quantile": calibrator["quantile"],
                "score_mean": distribution.get("mean"),
                "score_p50": distribution.get("p50"),
                "score_p90": distribution.get("p90"),
                "score_p99": distribution.get("p99"),
                "score_max": distribution.get("max"),
            })

    pd.DataFrame(rows).to_csv(output_dir / "calibration_scores_summary.csv", index=False)
    fallback_count = int((~calibration["ratio_eligible"]).sum())
    invalid_count = int((~calibration["analysis_valid"]).sum())
    manifest: dict[str, object] = {
        "format": bound_core.CALIBRATION_MANIFEST_FORMAT,
        "format_version": bound_core.CALIBRATION_MANIFEST_VERSION,
        "phase2_version": bound_core.PHASE2_VERSION,
        "status": "frozen",
        "test_queries_read": False,
        "input_path": str(input_path.resolve()),
        "input_sha256": contract["input_sha256"],
        "phase1_dir": str(phase1_dir.resolve()),
        "query_split_manifest_sha256": contract["query_split_manifest_sha256"],
        "phase1_run_manifest_sha256": contract["phase1_run_manifest_sha256"],
        "phase1_summary_sha256": contract["phase1_summary_sha256"],
        "estimator_formula_version": contract["estimator_formula_version"],
        "estimator_variant": estimator_core.DEPLOYMENT_ESTIMATOR,
        "clipping_policy": estimator_core.DEPLOYMENT_CLIPPING,
        "kappa_min": contract["kappa_min"],
        "fallback_policy": estimator_core.FALLBACK_POLICY,
        "alphas": list(alphas),
        "calibration_query_count": len(calibration_query_ids),
        "calibration_query_ids": calibration_query_ids,
        "calibration_record_count": int(len(calibration)),
        "ratio_eligible_count": int(len(eligible)),
        "ratio_eligible_fraction": float(len(eligible) / len(calibration)),
        "current_lb_fallback_count": fallback_count,
        "current_lb_fallback_fraction": float(fallback_count / len(calibration)),
        "invalid_fail_closed_count": invalid_count,
        "invalid_fail_closed_fraction": float(invalid_count / len(calibration)),
        "record_calibration_scores_sha256": record_score_sha,
        "query_calibration_scores_sha256": query_score_sha,
        "schema_audit": schema_audit,
        "calibrators": calibrator_artifacts,
        "producer": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "calibration_script_sha256": estimator_core.sha256_file(Path(__file__).resolve()),
            "bound_core_sha256": estimator_core.sha256_file(Path(bound_core.__file__).resolve()),
            "estimator_core_sha256": estimator_core.sha256_file(Path(estimator_core.__file__).resolve()),
        },
    }
    bound_core.write_json(output_dir / "calibration_manifest.json", manifest)
    print(json.dumps(estimator_core.json_safe(manifest), indent=2, sort_keys=True, allow_nan=False))
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--phase1-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--alphas", default=",".join(str(value) for value in bound_core.DEFAULT_ALPHAS))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        calibrate(args.input, args.phase1_dir, args.output_dir, bound_core.parse_alphas(args.alphas))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
