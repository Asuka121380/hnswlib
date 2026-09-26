from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.edge_estimation.quality_policy import (  # noqa: E402
    analyze_native_summary, select_operating_points, sweep_alphas,
    validate_quality_policy)


def policy() -> dict:
    return {
        "schema_version": 1,
        "protocol_id": "test-v2",
        "policy_kind": "scaled_threshold",
        "diagnostic_alphas": [0.9],
        "candidate_alphas": [1.0, 1.1, 1.2],
        "gates": {
            "balanced": {
                "max_global_false_prune_rate": 0.05,
                "max_p95_query_false_prune_rate": 0.1,
            },
            "strict": {
                "max_global_false_prune_rate": 0.01,
                "max_p95_query_false_prune_rate": 0.02,
            },
        },
        "fine_sweep": {
            "half_width": 0.05, "step": 0.01,
            "minimum_candidate_alpha": 1.0,
            "extension_alphas": [1.4, 1.6],
        },
        "selection": {"objective": "maximize_correct_prune_rate_subject_to_gate"},
    }


def native(alpha: float, *, tp: int, fp: int, fn: int, tn: int) -> dict:
    total = tp + fp + fn + tn
    first = total // 2
    second = total - first
    first_fp = min(fp, first)
    return {
        "alpha": alpha, "decision_count_s": total,
        "valid_estimate_count": total, "fallback_count": 0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "per_query": [
            {"query_id": 1, "counts": {
                "decision_count_s": first, "fp": first_fp}},
            {"query_id": 2, "counts": {
                "decision_count_s": second, "fp": fp - first_fp}},
        ],
    }


class QualityPolicyTest(unittest.TestCase):
    def test_repository_policy_is_valid_and_contains_legacy_anchors(self) -> None:
        value = json.loads((ROOT / "configs/edge_estimation/quality_policy_v2.json").read_text())
        validate_quality_policy(value)
        self.assertIn(1.45, value["candidate_alphas"])
        self.assertIn(1.54, value["candidate_alphas"])
        self.assertEqual(14, len(sweep_alphas(value)))

    def test_native_summary_derives_query_tail_rate(self) -> None:
        result = analyze_native_summary(native(1.0, tp=60, fp=2, fn=20, tn=18))
        self.assertEqual(0.02, result["metrics"]["global_false_prune_rate"])
        self.assertGreater(result["metrics"]["p95_query_false_prune_rate"], 0.03)

    def test_selection_uses_only_formal_candidates_and_emits_fine_grid(self) -> None:
        value = policy()
        analyses = [
            analyze_native_summary(native(0.9, tp=80, fp=10, fn=5, tn=5)),
            analyze_native_summary(native(1.0, tp=70, fp=6, fn=15, tn=9)),
            analyze_native_summary(native(1.1, tp=60, fp=2, fn=25, tn=13)),
            analyze_native_summary(native(1.2, tp=40, fp=0, fn=45, tn=15)),
        ]
        result = select_operating_points(analyses, value)
        self.assertEqual(1.1, result["selections"]["balanced"]["alpha"])
        self.assertEqual(1.2, result["selections"]["strict"]["alpha"])
        self.assertIn(1.15, result["selections"]["strict"]["fine_sweep_alphas"])
        self.assertNotEqual(0.9, result["selections"]["balanced"]["alpha"])

    def test_missing_safe_candidate_requests_extension(self) -> None:
        value = policy()
        analyses = [
            analyze_native_summary(native(alpha, tp=50, fp=10, fn=20, tn=20))
            for alpha in (1.0, 1.1, 1.2)
        ]
        result = select_operating_points(analyses, value)
        self.assertEqual("no_candidate_meets_gate",
                         result["selections"]["strict"]["status"])
        self.assertEqual([1.4, 1.6],
                         result["selections"]["strict"]["extension_alphas"])


if __name__ == "__main__":
    unittest.main()
