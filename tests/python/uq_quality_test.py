import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.edge_estimation.quality import ScoredEvent, summarize  # noqa: E402


class QualityTest(unittest.TestCase):
    def test_fixed_denominator_tie_and_fallback(self):
        events = [
            ScoredEvent(1, True, 10.0, 12.0, 11.0, "valid"),  # TP
            ScoredEvent(1, True, 10.0, 10.0, 11.0, "valid"),  # FP, exact tie is non-far
            ScoredEvent(2, True, 10.0, 15.0, None, "zero_length"),  # FN fallback
            ScoredEvent(2, True, 0.0, 0.0, 0.0, "valid"),  # TN, tau=0 valid
            ScoredEvent(3, False, 0.0, 1.0, 2.0, "valid"),  # outside S
        ]
        result = summarize(events, 1.0)
        self.assertEqual(4, result["decision_count_s"])
        self.assertEqual({"tp": 1, "fp": 1, "fn": 1, "tn": 1},
                         result["confusion"])
        self.assertEqual(1, result["fallback_count"])
        self.assertEqual(0.25, result["metrics"]["global_false_prune_rate"])

    def test_zero_denominators_are_null(self):
        result = summarize([], 1.0)
        self.assertIsNone(result["metrics"]["correct_prune_rate"])
        self.assertIsNone(result["metrics"]["prune_precision"])


if __name__ == "__main__":
    unittest.main()
