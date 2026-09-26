import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.edge_estimation.timing import paired_schedule, summarize_paired  # noqa: E402


class TimingTest(unittest.TestCase):
    def test_schedule_is_balanced_and_deterministic(self):
        first = paired_schedule(["pq", "pq4", "qjl"], blocks=5, repeats=5, seed=7)
        second = paired_schedule(["pq", "pq4", "qjl"], blocks=5, repeats=5, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(75, len(first))

    def test_paired_summary_and_contract_rejection(self):
        records = []
        for block in range(2):
            for method, elapsed in (("pq", 100.0), ("pq4", 50.0)):
                for repeat in range(2):
                    records.append({
                        "dataset_id": "d", "build_id": "b",
                        "mode": "ordered_estimator", "thread_count": 1,
                        "isa_profile": "portable", "block": block,
                        "method": method, "elapsed_ns": elapsed + repeat,
                        "query_count": 10, "eligible_events": 100,
                    })
        result = summarize_paired(records, "pq")
        self.assertAlmostEqual(2.0, result["methods"]["pq4"]["paired_speed_ratio_vs_reference"], places=1)
        records[-1] = dict(records[-1], build_id="different")
        with self.assertRaises(ValueError):
            summarize_paired(records, "pq")


if __name__ == "__main__":
    unittest.main()
