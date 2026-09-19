"""Contract tests for residual replay decisions and query aggregation."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] /
                       "scripts" / "v0" / "residual_estimator"))
from common import query_split  # noqa: E402
from replay_frontier import evaluate  # noqa: E402


class ReplayTest(unittest.TestCase):
    def test_split_has_no_leakage(self):
        split = query_split(100, [60, 20, 20], 42)
        groups = [set(group) for group in split.values()]
        self.assertEqual(set.union(*groups), set(range(100)))
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])

    def test_event_and_query_denominators(self):
        decision = np.array([True, False, True, False])
        far = np.array([True, True, False, False])
        ids = np.array([7, 7, 8, 8])
        row, queries = evaluate(decision, far, ids, "development",
                                "residual-direct", 1.0, 42, 64)
        self.assertEqual((row["N"], row["TP"], row["FP"], row["FN"]),
                         (4, 1, 1, 1))
        self.assertEqual(row["FP_per_N"], 0.25)
        self.assertEqual(row["FP_per_near"], 0.5)
        self.assertEqual(row["sampled_query_exposure"], 0.5)
        self.assertEqual([query["FP"] for query in queries], [0, 1])

    def test_strict_threshold_tie_is_retained(self):
        estimate = np.array([1.0, 1.0001])
        threshold = np.array([1.0, 1.0])
        np.testing.assert_array_equal(estimate > threshold, [False, True])


if __name__ == "__main__":
    unittest.main()
