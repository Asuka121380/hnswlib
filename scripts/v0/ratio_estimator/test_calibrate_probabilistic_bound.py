#!/usr/bin/env python3
"""Synthetic tests for Phase-2 empirical calibration and held-out apply."""

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

import apply_probabilistic_bound as apply_script
import calibrate_probabilistic_bound as calibrate_script
import probabilistic_bound_core as bound_core
import ratio_estimator_core as estimator_core


def make_row(query_id: int, score: float, *, fallback: bool = False) -> dict[str, str]:
    n = 1.0
    ell = 1.0
    x_dot_u = 0.5
    true_distance = n * n + ell * ell - 2.0 * ell * x_dot_u
    x_dot_r = x_dot_u - score / 2.0
    reconstruction_norm = 1.0
    direction_error = 0.0 if not fallback else math.sqrt(2.0 - 2.0 * 0.4)
    return {
        "query_id": str(query_id),
        "current_node_id": str(1000 + query_id),
        "candidate_id": str(2000 + query_id),
        "threshold": "0.75",
        "edge_length": repr(ell),
        "reconstruction_norm": repr(reconstruction_norm),
        "x_norm": repr(n),
        "x_dot_r": repr(x_dot_r),
        "x_dot_true_direction": repr(x_dot_u),
        "actual_direction_error": repr(direction_error),
        "direction_error": repr(direction_error),
        "current_lb": "0.25",
        "exact_squared_distance": repr(true_distance),
        "geometric_squared_distance": repr(true_distance),
        "diagnostic_valid": "1",
    }


def write_phase1_fixture(
    root: Path,
    rows: list[dict[str, str]],
    *,
    split: dict[str, list[str]] | None = None,
) -> tuple[Path, Path]:
    input_path = root / "raw.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    input_sha = estimator_core.sha256_file(input_path)
    query_ids = sorted({row["query_id"] for row in rows}, key=int)
    if split is None:
        split = {
            "train": query_ids[:6],
            "calibration": query_ids[6:8],
            "test": query_ids[8:],
        }
    phase1 = root / "phase1"
    phase1.mkdir(parents=True)
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
    return input_path, phase1


class OrderStatisticTests(unittest.TestCase):
    def test_hand_computable_order_statistic(self) -> None:
        result = bound_core.conformal_order_statistic([4.0, 1.0, 3.0, 2.0], 0.4)
        self.assertEqual(result["order_statistic_index_one_based"], 3)
        self.assertEqual(result["quantile"], 3.0)
        self.assertEqual(result["quantile_hex"], float(3.0).hex())

    def test_unsupported_sample_size_is_explicit(self) -> None:
        result = bound_core.conformal_order_statistic([1.0, 2.0, 3.0], 0.1)
        self.assertEqual(result["status"], "unsupported_sample_size")
        self.assertIsNone(result["quantile"])
        self.assertGreater(result["order_statistic_index_one_based"], result["sample_count"])

    def test_query_max_score_with_multiple_records(self) -> None:
        frame = pd.DataFrame({
            "query_id": ["a", "a", "b"],
            "score": [0.1, 0.7, -0.2],
        })
        result = bound_core.query_max_scores(frame, ["a", "b"], "score")
        self.assertEqual(result.loc["a"], 0.7)
        self.assertEqual(result.loc["b"], -0.2)

    def test_clopper_pearson_known_endpoints(self) -> None:
        lower, upper = bound_core.clopper_pearson(0, 10)
        self.assertEqual(lower, 0.0)
        self.assertAlmostEqual(upper, 1.0 - 0.025 ** (1.0 / 10.0), places=12)
        lower_all, upper_all = bound_core.clopper_pearson(10, 10)
        self.assertAlmostEqual(lower_all, 0.025 ** (1.0 / 10.0), places=12)
        self.assertEqual(upper_all, 1.0)


class ApplyPrimitiveTests(unittest.TestCase):
    def test_fallback_is_counted_and_all_bounds_are_nonnegative(self) -> None:
        frame = pd.DataFrame({
            "deployment_distance_hat": [1.0, 2.0, -1.0],
            "ratio_eligible": [True, False, True],
            "current_lb_safe": [0.2, 0.4, 0.0],
            "shadow_exact_squared_distance": [0.8, 0.5, 0.0],
        })
        result = bound_core.apply_calibrator_to_frame(frame, 0.3)
        self.assertEqual(int(result["used_current_lb_fallback"].sum()), 1)
        self.assertTrue((result["probabilistic_lb"] >= 0.0).all())
        self.assertEqual(result.iloc[1]["probabilistic_lb"], 0.4)

    def test_calibrator_round_trip_preserves_float(self) -> None:
        calibrator = {
            "value": 0.12345678901234566,
            "hex": float(0.12345678901234566).hex(),
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "calibrator.json"
            bound_core.write_json(path, calibrator)
            loaded = bound_core.read_json(path)
            self.assertEqual(float(loaded["value"]).hex(), calibrator["hex"])


class EndToEndTests(unittest.TestCase):
    def fixture_rows(self, test_delta: float = 0.0) -> list[dict[str, str]]:
        scores = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.15, 0.35, 0.1 + test_delta, 0.25 + test_delta]
        return [make_row(query_id, score) for query_id, score in enumerate(scores)]

    def test_calibration_and_test_queries_are_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path, phase1 = write_phase1_fixture(root, self.fixture_rows())
            contract = bound_core.load_phase1_contract(input_path, phase1)
            data, _ = bound_core.load_derived_input(contract)
            calibration = set(data.loc[data["split"] == "calibration", "query_id"].astype(str))
            test = set(data.loc[data["split"] == "test", "query_id"].astype(str))
            self.assertFalse(calibration & test)

    def test_test_values_do_not_influence_quantile(self) -> None:
        quantiles = []
        score_hashes = []
        for delta in (0.0, 100.0):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                input_path, phase1 = write_phase1_fixture(root, self.fixture_rows(delta))
                output = root / "phase2"
                with contextlib.redirect_stdout(io.StringIO()):
                    manifest = calibrate_script.calibrate(input_path, phase1, output, [0.5])
                calibrator = bound_core.read_json(output / "calibrators" / "record_alpha_5e-1.json")
                quantiles.append(calibrator["quantile"])
                score_hashes.append(manifest["record_calibration_scores_sha256"])
        self.assertEqual(quantiles[0], quantiles[1])
        self.assertEqual(score_hashes[0], score_hashes[1])

    def test_input_order_does_not_change_calibration_scores(self) -> None:
        hashes = []
        quantiles = []
        rows = self.fixture_rows()
        for ordered in (rows, list(reversed(rows))):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                input_path, phase1 = write_phase1_fixture(root, ordered)
                output = root / "phase2"
                with contextlib.redirect_stdout(io.StringIO()):
                    manifest = calibrate_script.calibrate(input_path, phase1, output, [0.5])
                hashes.append(manifest["record_calibration_scores_sha256"])
                quantiles.append(bound_core.read_json(output / "calibrators" / "record_alpha_5e-1.json")["quantile"])
        self.assertEqual(hashes[0], hashes[1])
        self.assertEqual(quantiles[0], quantiles[1])

    def test_sha_mismatch_rejects_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path, phase1 = write_phase1_fixture(root, self.fixture_rows())
            output = root / "phase2"
            with contextlib.redirect_stdout(io.StringIO()):
                calibrate_script.calibrate(input_path, phase1, output, [0.5])
            input_path.write_text(input_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                bound_core.load_phase1_contract(input_path, phase1)

    def test_apply_emits_coverage_and_counts_fallback(self) -> None:
        rows = self.fixture_rows()
        rows[-1] = make_row(9, 0.25, fallback=True)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path, phase1 = write_phase1_fixture(root, rows)
            output = root / "phase2"
            with contextlib.redirect_stdout(io.StringIO()):
                calibrate_script.calibrate(input_path, phase1, output, [0.5])
                manifest = apply_script.apply(input_path, phase1, output, output)
            coverage = pd.read_csv(output / "test_coverage_summary.csv")
            supported = coverage[coverage["status"] == "supported"]
            self.assertFalse(supported.empty)
            self.assertGreater(int(supported["current_lb_fallback_count"].max()), 0)
            self.assertEqual(manifest["calibration_test_disjoint"], True)
            self.assertTrue((pd.read_csv(output / "violation_records.csv").get("probabilistic_lb", pd.Series(dtype=float)) >= 0).all())

    def test_calibrator_validation_rejects_input_sha_mismatch(self) -> None:
        contract = {
            "input_sha256": "a" * 64,
            "query_split_manifest_sha256": "b" * 64,
            "estimator_formula_version": estimator_core.ESTIMATOR_FORMULA_VERSION,
            "kappa_min": 0.5,
        }
        calibrator = {
            "format": bound_core.CALIBRATOR_FORMAT,
            "format_version": bound_core.CALIBRATOR_FORMAT_VERSION,
            "input_sha256": "c" * 64,
            "query_split_manifest_sha256": "b" * 64,
            "estimator_formula_version": estimator_core.ESTIMATOR_FORMULA_VERSION,
            "kappa_min": 0.5,
        }
        with self.assertRaisesRegex(ValueError, "input SHA mismatch"):
            bound_core.validate_calibrator(calibrator, contract)


if __name__ == "__main__":
    unittest.main()
