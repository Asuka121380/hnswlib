#!/usr/bin/env python3
"""Synthetic tests for Phase-3 offline probabilistic-pruning simulation."""

from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import apply_probabilistic_bound as phase2_apply
import calibrate_probabilistic_bound as phase2_calibrate
import probabilistic_bound_core as bound_core
import ratio_estimator_core as estimator_core
import simulate_probabilistic_pruning as phase3


def make_row(
    query_id: int,
    score: float,
    *,
    threshold: float = 0.75,
    current_lb: float = 0.25,
    diagnostic_valid: bool = True,
) -> dict[str, str]:
    n = 1.0
    ell = 1.0
    x_dot_u = 0.5
    exact = n * n + ell * ell - 2.0 * ell * x_dot_u
    x_dot_r = x_dot_u - score / 2.0
    raw_hat = exact + score
    return {
        "cap_input_schema_version": "2",
        "run_id": "phase3-synthetic",
        "query_id": str(query_id),
        "current_node_id": str(1000 + query_id),
        "candidate_id": str(2000 + query_id),
        "graph_layer": "0",
        "bound_status": "0",
        "ef_search": "200",
        "current_squared_distance": "0.5",
        "threshold": repr(threshold),
        "edge_length": repr(ell),
        "reconstruction_norm": "1.0",
        "x_norm": repr(n),
        "x_dot_r": repr(x_dot_r),
        "x_dot_true_direction": repr(x_dot_u),
        "actual_direction_error": "0.0",
        "direction_error": "0.0",
        "anchor_projection": repr(x_dot_r),
        "current_lb": repr(current_lb),
        "exact_squared_distance": repr(exact),
        "geometric_squared_distance": repr(exact),
        "current_would_prune": "1" if current_lb > threshold else "0",
        "raw_would_prune": "1" if diagnostic_valid and raw_hat > threshold else "0",
        "oracle_would_prune": "1" if exact > threshold else "0",
        "diagnostic_valid": "1" if diagnostic_valid else "0",
    }


def write_fixture(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    rows = [make_row(query_id, score) for query_id, score in enumerate([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3])]
    rows.extend([
        make_row(6, 0.1),
        make_row(7, 0.2),
        make_row(8, 0.5, threshold=0.9),
        make_row(9, 0.5, threshold=1.0),
    ])
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)
    input_path = raw_dir / "cap_diagnostic_input.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    input_sha = estimator_core.sha256_file(input_path)
    split = {
        "train": [str(value) for value in range(6)],
        "calibration": ["6", "7"],
        "test": ["8", "9"],
    }
    phase1 = root / "phase1"
    phase1.mkdir()
    bound_core.write_json(phase1 / "query_split_manifest.json", {
        "format": "v0_ratio_estimator_query_split",
        "format_version": 1,
        "input_sha256": input_sha,
        "query_ids": split,
    })
    bound_core.write_json(phase1 / "run_manifest.json", {
        "format": "v0_ratio_estimator_phase1_run_manifest",
        "format_version": 1,
        "analyzer_version": "v0_ratio_estimator_phase1_v1",
        "estimator_formula_version": estimator_core.ESTIMATOR_FORMULA_VERSION,
        "input_sha256": input_sha,
        "config": {"selected_kappa_min": 0.5},
    })
    bound_core.write_json(phase1 / "summary.json", {
        "format": "v0_ratio_estimator_phase1_summary",
        "format_version": 1,
        "status": "PASS",
        "decision": "GO_TO_PHASE2",
        "tolerances": {
            "closure_absolute": 1e-10,
            "closure_relative": 1e-9,
            "projection": 1e-9,
        },
        "gate_diagnostics": {"selected_kappa_min": 0.5},
    })
    metadata_path = raw_dir / "metadata.json"
    bound_core.write_json(metadata_path, {
        "format": "hnswlib_v0_search_run",
        "format_version": 2,
        "run_id": "phase3-synthetic",
        "dataset": "synthetic",
        "mode": "shadow",
        "query_count": 1000,
        "ef_search": 200,
        "shadow_sample_modulus": 16,
        "shadow_sample_remainder": 0,
        "real_pruning_enabled": False,
        "bound_pruned_semantics": "would_prune_observe_only",
        "index_sha256": "a" * 64,
        "sidecar_sha256": "b" * 64,
    })
    query_metrics_path = raw_dir / "query_metrics.csv"
    pd.DataFrame([
        {
            "query_id": str(query_id),
            "baseline_recall_at_k": 1.0,
            "v0_recall_at_k": 1.0,
            "results_equal": 1,
            "baseline_latency_ns": 1000,
            "v0_latency_ns": 2000,
            "bound_evaluated": 16,
            "bound_pruned": 0,
            "raw_prunable": 1,
            "oracle_prunable": 1,
            "exact_fallback": 0,
            "exact_only_fallback": 0,
            "exact_distance_saved": 0,
            "lower_bound_violation": 0,
            "false_prune": 0,
        }
        for query_id in range(10)
    ]).to_csv(query_metrics_path, index=False)
    phase2 = root / "phase2"
    with contextlib.redirect_stdout(io.StringIO()):
        phase2_calibrate.calibrate(input_path, phase1, phase2, [0.5])
        phase2_manifest = phase2_apply.apply(input_path, phase1, phase2, phase2)
    if phase2_manifest["decision"] != "GO_TO_PHASE3":
        raise AssertionError(f"synthetic Phase 2 fixture failed: {phase2_manifest}")
    return input_path, phase1, phase2, metadata_path, query_metrics_path


def primitive_frame(
    *,
    distance_hat: float,
    threshold: float,
    exact: float,
    current_lb: float = 0.0,
    eligible: bool = True,
    valid: bool = True,
) -> pd.DataFrame:
    return pd.DataFrame({
        "query_id": ["q"],
        "current_node_id": ["c"],
        "candidate_id": ["x"],
        "threshold": [threshold],
        "shadow_exact_squared_distance": [exact],
        "deployment_distance_hat": [distance_hat],
        "ratio_eligible": [eligible],
        "analysis_valid": [valid],
        "kappa_meta": [1.0],
        "current_lb_safe": [current_lb],
        "current_prune": [current_lb > threshold],
        "oracle_prune": [exact > threshold],
        "oracle_margin": [exact - threshold],
        "relative_abs_margin": [abs(exact - threshold) / max(abs(threshold), 1e-12)],
        "near_margin": [False],
    })


class DecisionPrimitiveTests(unittest.TestCase):
    def test_equal_lower_bound_does_not_prune(self) -> None:
        data = primitive_frame(distance_hat=1.0, threshold=1.0, exact=2.0)
        result = phase3.apply_frozen_quantile(data, 0.0)
        self.assertFalse(bool(result.iloc[0]["simulation_prune"]))

    def test_strictly_greater_lower_bound_prunes(self) -> None:
        data = primitive_frame(distance_hat=1.0001, threshold=1.0, exact=2.0)
        result = phase3.apply_frozen_quantile(data, 0.0)
        self.assertTrue(bool(result.iloc[0]["simulation_prune"]))

    def test_false_prune_definition_uses_exact_threshold(self) -> None:
        data = primitive_frame(distance_hat=1.2, threshold=1.0, exact=1.0)
        result = phase3.apply_frozen_quantile(data, 0.0)
        self.assertTrue(bool(result.iloc[0]["simulation_false_prune"]))

    def test_fallback_uses_current_lower_bound(self) -> None:
        data = primitive_frame(distance_hat=100.0, threshold=0.7, exact=1.0, current_lb=0.8, eligible=False)
        result = phase3.apply_frozen_quantile(data, 0.0)
        self.assertEqual(float(result.iloc[0]["simulation_lb"]), 0.8)
        self.assertTrue(bool(result.iloc[0]["simulation_used_current_fallback"]))

    def test_invalid_record_cannot_use_probability_estimator(self) -> None:
        data = primitive_frame(distance_hat=100.0, threshold=1.0, exact=2.0, eligible=False, valid=False)
        result = phase3.apply_frozen_quantile(data, 0.0, kappa_min=0.4)
        self.assertFalse(bool(result.iloc[0]["simulation_ratio_eligible"]))
        self.assertFalse(bool(result.iloc[0]["simulation_prune"]))

    def test_zero_oracle_denominator_is_explicit_na(self) -> None:
        data = primitive_frame(distance_hat=0.5, threshold=1.0, exact=1.0)
        data["decision"] = False
        data["lb"] = 0.5
        summary = phase3.summarize_decisions(
            data,
            operating_point_id="test",
            strategy="test",
            strategy_role="test",
            lb_column="lb",
            decision_column="decision",
            formal_candidate=False,
        )
        self.assertTrue(math.isnan(float(summary["oracle_recovery_rate"])))

    def test_record_and_query_primary_risk_are_not_mixed(self) -> None:
        rows = []
        for index in range(100):
            row = primitive_frame(distance_hat=2.0 if index == 0 else 0.0, threshold=1.0, exact=1.0)
            row["query_id"] = "one-query"
            rows.append(row)
        data = pd.concat(rows, ignore_index=True)
        data["decision"] = data["deployment_distance_hat"] > data["threshold"]
        data["lb"] = data["deployment_distance_hat"]
        record = phase3.summarize_decisions(
            data,
            operating_point_id="record",
            strategy="test",
            strategy_role="test",
            lb_column="lb",
            decision_column="decision",
            formal_candidate=True,
            calibration_level="record",
            nominal_alpha=0.01,
            quantile=0.0,
        )
        query = phase3.summarize_decisions(
            data,
            operating_point_id="query",
            strategy="test",
            strategy_role="test",
            lb_column="lb",
            decision_column="decision",
            formal_candidate=True,
            calibration_level="query",
            nominal_alpha=0.01,
            quantile=0.0,
        )
        self.assertTrue(record["nominal_statistically_compatible"])
        self.assertFalse(query["nominal_statistically_compatible"])


class EndToEndTests(unittest.TestCase):
    def run_fixture(self, root: Path) -> dict[str, object]:
        input_path, phase1, phase2, metadata, query_metrics = write_fixture(root)
        output = root / "phase3"
        with contextlib.redirect_stdout(io.StringIO()):
            manifest = phase3.simulate(
                input_path,
                phase1,
                phase2,
                output,
                metadata_path=metadata,
                query_metrics_path=query_metrics,
            )
        return {"manifest": manifest, "output": output}

    def test_same_calibrator_and_input_are_deterministic(self) -> None:
        artifacts = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as temp:
                result = self.run_fixture(Path(temp))
                output = result["output"]
                selected = json.loads((output / "selected_operating_points.json").read_text(encoding="utf-8"))
                artifacts.append({
                    "summary": (output / "operating_point_summary.csv").read_bytes(),
                    "decision": selected["decision"],
                    "operating_points": selected["operating_points"],
                })
        self.assertEqual(artifacts[0], artifacts[1])

    def test_test_and_calibration_query_sets_are_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            input_path, phase1, _, _, _ = write_fixture(Path(temp))
            contract = bound_core.load_phase1_contract(input_path, phase1)
            split = contract["query_split_manifest"]["query_ids"]
            self.assertFalse(set(split["calibration"]) & set(split["test"]))

    def test_raw_point_estimate_is_diagnostic_not_formal(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = self.run_fixture(Path(temp))
            summary = pd.read_csv(result["output"] / "operating_point_summary.csv")
            raw = summary[summary["operating_point_id"].eq("raw_point_estimate")].iloc[0]
            self.assertFalse(bool(raw["formal_candidate"]))
            self.assertEqual(raw["strategy_role"], "diagnostic_upper_bound")

    def test_output_counts_fallback_and_freezes_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = self.run_fixture(Path(temp))
            manifest = result["manifest"]
            selected = json.loads((result["output"] / "selected_operating_points.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["calibration_test_disjoint"])
            self.assertLessEqual(selected["selection_count"], 3)
            self.assertFalse(selected["test_quantile_refit"])
            self.assertFalse(selected["sensitivity_used_for_selection"])
            self.assertTrue(selected["requires_fresh_phase4_evaluation_queries"])


if __name__ == "__main__":
    unittest.main()
