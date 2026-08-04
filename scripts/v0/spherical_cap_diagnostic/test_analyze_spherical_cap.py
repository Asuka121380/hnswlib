#!/usr/bin/env python3

import csv
import json
import tempfile
import unittest
from decimal import Decimal, localcontext
from pathlib import Path

from analyze_spherical_cap import (
    DECIMAL_PRECISION,
    analyze,
    cap_geometry,
    closed_form_support,
    span_candidate_support,
)


class GeometryTest(unittest.TestCase):
    def test_geometry_classification(self):
        self.assertEqual(cap_geometry(Decimal("1"), Decimal("0"))[0], "singleton")
        self.assertEqual(cap_geometry(Decimal("1"), Decimal("2"))[0], "full_sphere")
        self.assertEqual(cap_geometry(Decimal("2"), Decimal("0.5"))[0], "empty")
        self.assertEqual(cap_geometry(Decimal("0"), Decimal("1"))[0], "s_zero")

    def test_closed_form_matches_independent_span_reduction(self):
        with localcontext() as context:
            context.prec = DECIMAL_PRECISION
            kind, kappa, valid = cap_geometry(Decimal("1"), Decimal("1"))
            self.assertTrue(valid)
            support, branch, t = closed_form_support(
                kind, kappa, Decimal("1"), Decimal("0"), Decimal("1"))
            independent = span_candidate_support(kind, kappa, Decimal("1"), t)
            self.assertEqual(branch, "boundary")
            self.assertLess(abs(support - independent), Decimal("1e-70"))


class AnalyzerTest(unittest.TestCase):
    FIELDS = [
        "cap_input_schema_version", "run_id", "query_id", "current_node_id",
        "candidate_id", "graph_layer", "bound_status", "ef_search",
        "current_squared_distance", "threshold", "edge_length",
        "direction_error", "anchor_projection", "current_lb",
        "exact_squared_distance", "current_would_prune", "oracle_would_prune",
        "raw_would_prune",
        "reconstruction_norm", "x_norm", "x_dot_r", "true_edge_norm",
        "x_dot_true_direction", "actual_direction_error", "certificate_slack",
        "diagnostic_valid",
    ]

    @staticmethod
    def row(**overrides):
        row = {
            "cap_input_schema_version": "1", "run_id": "synthetic",
            "query_id": "0", "current_node_id": "10", "candidate_id": "11",
            "graph_layer": "0", "bound_status": "0", "ef_search": "100",
            "current_squared_distance": "1", "threshold": "0.2",
            "edge_length": "1", "direction_error": "1",
            "anchor_projection": "0", "current_lb": "0",
            "exact_squared_distance": "1", "current_would_prune": "0",
            "raw_would_prune": "1", "oracle_would_prune": "1",
            "reconstruction_norm": "1",
            "x_norm": "1", "x_dot_r": "0", "true_edge_norm": "1",
            "x_dot_true_direction": "0.5", "actual_direction_error": "0.5",
            "certificate_slack": "0.5", "diagnostic_valid": "1",
        }
        row.update(overrides)
        return row

    def test_end_to_end_artifacts_and_gate(self):
        rows = [
            self.row(),
            self.row(query_id="1", candidate_id="12", x_dot_r="1",
                     x_dot_true_direction="1", exact_squared_distance="0",
                     threshold="0.2", oracle_would_prune="0"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.csv"
            with input_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            summary = analyze(
                input_path, root / "out", Decimal("1e-10"),
                Decimal("1e-30"), Decimal("1e-12"),
                min_extra_coverage=0.0, max_top_query_share=1.0,
                min_proper_fraction=0.0, min_boundary_fraction=0.0)
            self.assertEqual(summary["decision"], "go_candidate")
            self.assertEqual(summary["counts"]["cap_extra_prune"], 1)
            self.assertEqual(summary["counts"]["cap_lower_bound_violation"], 0)
            for name in ("summary.json", "cap_candidate_table.csv.gz",
                         "cap_geometry_summary.csv", "cap_coverage_by_query.csv",
                         "cap_data_quality.json"):
                self.assertTrue((root / "out" / name).is_file(), name)
            parsed = json.loads((root / "out" / "summary.json").read_text())
            self.assertEqual(parsed["decimal_precision"], DECIMAL_PRECISION)


if __name__ == "__main__":
    unittest.main()
