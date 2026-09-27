from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.edge_estimation.run_final_horizontal import (  # noqa: E402
    METHODS, benchmark_command)


class FinalHorizontalTest(unittest.TestCase):
    def test_canonical_method_order_is_frozen(self) -> None:
        self.assertEqual(
            METHODS, ("pq8", "pq4", "opq", "prq", "jq", "rabitq"))

    def test_opq_and_jq_use_batch_command(self) -> None:
        for method in ("opq", "jq"):
            command = benchmark_command(
                Path("runner"), Path("events"), Path(method),
                Path("queries"), method, 128)
            self.assertEqual(command[1], "bench-batch-artifact")
            self.assertEqual(command[-2:], ["128", "1"])

    def test_other_methods_use_scalar_command(self) -> None:
        for method in ("pq8", "pq4", "prq", "rabitq"):
            command = benchmark_command(
                Path("runner"), Path("events"), Path(method),
                Path("queries"), method, 128)
            self.assertEqual(command[1], "bench-artifact")
            self.assertEqual(command[-1], "1")


if __name__ == "__main__":
    unittest.main()
