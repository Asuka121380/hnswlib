#!/usr/bin/env python3
"""Tests for the Phase-1 ratio estimator diagnostic."""

from __future__ import annotations

import csv
import contextlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).with_name("analyze_ratio_estimator.py")
SPEC = importlib.util.spec_from_file_location("analyze_ratio_estimator", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ratio = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ratio
SPEC.loader.exec_module(ratio)


def make_row(
    query_id: int = 0,
    *,
    n: float = 1.0,
    ell: float = 1.0,
    s: float = 1.0,
    x_dot_r: float = 1.0,
    x_dot_u: float = 1.0,
    actual_error: float = 0.0,
    meta_error: float | None = None,
    exact_offset: float = 0.0,
    diagnostic_valid: str = "1",
) -> dict[str, str]:
    exact = n * n + ell * ell - 2.0 * ell * x_dot_u + exact_offset
    return {
        "query_id": str(query_id),
        "current_node_id": str(1000 + query_id),
        "candidate_id": str(2000 + query_id),
        "threshold": "1.0",
        "edge_length": repr(ell),
        "reconstruction_norm": repr(s),
        "x_norm": repr(n),
        "x_dot_r": repr(x_dot_r),
        "x_dot_true_direction": repr(x_dot_u),
        "actual_direction_error": repr(actual_error),
        "direction_error": repr(actual_error if meta_error is None else meta_error),
        "current_lb": "0.0",
        "exact_squared_distance": repr(exact),
        "diagnostic_valid": diagnostic_valid,
    }


def to_derived(rows: list[dict[str, str]], **config_kwargs) -> pd.DataFrame:
    raw = pd.DataFrame(rows, dtype="string")
    normalized, _ = ratio.audit_and_normalize_schema(raw)
    return ratio.derive_estimators(normalized, ratio.AnalysisConfig(**config_kwargs))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class FormulaTests(unittest.TestCase):
    def test_hand_computable_two_dimensional_record(self) -> None:
        # u=(1,0), x=(1,0), r_hat=(0.8,0.6), ||r||=1.
        # kappa=0.8, raw rho=0.8, ratio rho=1, and true D=0.
        e = math.sqrt((1.0 - 0.8) ** 2 + 0.6**2)
        data = to_derived([make_row(x_dot_r=0.8, actual_error=e)])
        row = data.iloc[0]
        self.assertAlmostEqual(row["kappa_actual"], 0.8, places=14)
        self.assertAlmostEqual(row["rho_hat_raw_unclipped"], 0.8, places=14)
        self.assertAlmostEqual(row["rho_hat_ratio_unclipped"], 1.0, places=14)
        self.assertAlmostEqual(row["distance_hat_ratio_unclipped"], 0.0, places=14)

    def test_exact_reconstruction_has_unit_kappa_and_exact_estimators(self) -> None:
        data = to_derived([make_row()])
        row = data.iloc[0]
        self.assertEqual(row["kappa_actual"], 1.0)
        self.assertEqual(row["rho_true"], 1.0)
        self.assertEqual(row["rho_hat_raw_unclipped"], 1.0)
        self.assertEqual(row["rho_hat_ratio_unclipped"], 1.0)
        self.assertEqual(row["distance_hat_raw_unclipped"], 0.0)
        self.assertEqual(row["distance_hat_ratio_unclipped"], 0.0)

    def test_direction_attenuation_is_removed_by_ratio(self) -> None:
        kappa = 0.6
        tangent = math.sqrt(1.0 - kappa * kappa)
        error = math.sqrt((1.0 - kappa) ** 2 + tangent * tangent)
        data = to_derived([make_row(s=1.0, x_dot_r=kappa, actual_error=error)])
        row = data.iloc[0]
        self.assertAlmostEqual(row["rho_hat_raw_unclipped"], kappa)
        self.assertAlmostEqual(row["rho_hat_ratio_unclipped"], 1.0)

    def test_small_kappa_triggers_fallback(self) -> None:
        kappa = 0.05
        error = math.sqrt(2.0 - 2.0 * kappa)
        data = to_derived([make_row(x_dot_r=kappa, actual_error=error)])
        data["split"] = "train"
        data["ratio_fallback"] = ~data["analysis_valid"] | (data["kappa_actual"] < 0.1)
        data["estimator_eligible"] = ~data["ratio_fallback"]
        self.assertTrue(bool(data.iloc[0]["ratio_fallback"]))
        self.assertFalse(bool(data.iloc[0]["estimator_eligible"]))

    def test_non_finite_and_zero_norm_are_rejected(self) -> None:
        zero = make_row(n=0.0, x_dot_u=0.0, x_dot_r=0.0)
        non_finite = make_row(query_id=1)
        non_finite["x_dot_r"] = "nan"
        data = to_derived([zero, non_finite])
        self.assertFalse(data["analysis_valid"].any())
        self.assertIn("non_positive_x_norm", data.iloc[0]["invalid_reason"])
        self.assertIn("non_finite_or_non_numeric_input", data.iloc[1]["invalid_reason"])

    def test_clipped_and_unclipped_outputs_are_both_preserved(self) -> None:
        # kappa=.5 and raw rho=.75 produces ratio rho=1.5 before clipping.
        kappa = 0.5
        error = math.sqrt(2.0 - 2.0 * kappa)
        data = to_derived([make_row(x_dot_r=0.75, actual_error=error)])
        row = data.iloc[0]
        self.assertAlmostEqual(row["rho_hat_ratio_unclipped"], 1.5)
        self.assertEqual(row["rho_hat_ratio_clipped"], 1.0)
        self.assertIn("distance_hat_ratio_unclipped", data.columns)
        self.assertIn("distance_hat_ratio_clipped", data.columns)


class SplitAndSchemaTests(unittest.TestCase):
    def test_query_split_has_no_leakage(self) -> None:
        split = ratio.create_query_split([str(i) for i in range(100)], seed=42)
        ratio.validate_split(split, [str(i) for i in range(100)])
        self.assertEqual((len(split["train"]), len(split["calibration"]), len(split["test"])), (60, 20, 20))

    def test_input_order_does_not_change_split(self) -> None:
        ordered = [str(i) for i in range(31)]
        reversed_ids = list(reversed(ordered))
        self.assertEqual(ratio.create_query_split(ordered, 42), ratio.create_query_split(reversed_ids, 42))

    def test_missing_column_fails_closed(self) -> None:
        row = make_row()
        del row["x_dot_r"]
        with self.assertRaises(ratio.SchemaError):
            ratio.audit_and_normalize_schema(pd.DataFrame([row], dtype="string"))

    def test_exact_distance_alias_is_audited(self) -> None:
        normalized, audit = ratio.audit_and_normalize_schema(pd.DataFrame([make_row()], dtype="string"))
        self.assertIn("shadow_exact_squared_distance", normalized.columns)
        self.assertEqual(audit["alias_resolution"]["shadow_exact_squared_distance"], "exact_squared_distance")

    def test_geometric_distance_is_preferred_and_source_difference_is_audited(self) -> None:
        row = make_row(exact_offset=1e-4)
        row["geometric_squared_distance"] = "0.0"
        normalized, audit = ratio.audit_and_normalize_schema(pd.DataFrame([row], dtype="string"))
        self.assertEqual(float(normalized.iloc[0]["shadow_exact_squared_distance"]), 0.0)
        self.assertEqual(audit["alias_resolution"]["shadow_exact_squared_distance"], "geometric_squared_distance")
        differences = audit["coexisting_source_differences"]
        self.assertEqual(differences["geometric_squared_distance_vs_exact_squared_distance"]["different_count"], 1)


class EndToEndTests(unittest.TestCase):
    def test_formal_closure_failure_returns_fail_closed(self) -> None:
        rows = [make_row(query_id=i, exact_offset=1e-4 if i == 0 else 0.0) for i in range(10)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "input.csv"
            output_dir = root / "output"
            write_csv(input_path, rows)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                exit_code = ratio.main([
                    "--input", str(input_path),
                    "--output-dir", str(output_dir),
                    "--mode", "formal",
                    "--selected-kappa-min", "0.1",
                    "--skip-plots",
                ])
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertNotEqual(exit_code, 0)
            self.assertEqual(summary["status"], "FAIL_CLOSED")
            self.assertEqual(summary["decision"], "NO_GO_RATIO_ESTIMATOR")
            self.assertGreater(summary["counts"]["distance_closure_failures"], 0)
            self.assertTrue((output_dir / "invalid_records.csv").exists())

    def test_smoke_run_emits_required_artifacts(self) -> None:
        rows = []
        for query_id in range(10):
            kappa = 0.8
            error = math.sqrt(2.0 - 2.0 * kappa)
            rows.append(make_row(query_id=query_id, x_dot_r=kappa, actual_error=error))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "input.csv"
            output_dir = root / "output"
            write_csv(input_path, rows)
            summary = ratio.analyze(
                input_path,
                output_dir,
                ratio.AnalysisConfig(mode="smoke", selected_kappa_min=0.1, skip_plots=True),
            )
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(summary["decision"], "SMOKE_ONLY")
            for name in (
                "run_manifest.json",
                "query_split_manifest.json",
                "schema_audit.json",
                "estimator_record_summary.csv",
                "estimator_query_summary.csv",
                "conditional_bias_by_kappa.csv",
                "conditional_bias_by_rho.csv",
                "conditional_bias_by_margin.csv",
                "invalid_records.csv",
                "summary.json",
                "PHASE1_ESTIMATOR_DIAGNOSTIC_REPORT.md",
            ):
                self.assertTrue((output_dir / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
