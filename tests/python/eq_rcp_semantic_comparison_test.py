#!/usr/bin/env python3

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "eq_rcp"
    / "compare_semantic_outputs.py"
)
SPEC = importlib.util.spec_from_file_location("compare_semantic_outputs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SemanticComparisonTest(unittest.TestCase):
    def test_normalizers_remove_only_nondeterministic_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "metadata.json"
            metadata.write_text(
                json.dumps({"schema_version": 1, "build_latency_ns": 9, "seed": 42}),
                encoding="utf-8",
            )
            normalized = json.loads(MODULE.normalized_metadata(metadata))
            self.assertEqual(normalized, {"schema_version": 1, "seed": 42})

            stats = root / "query_stats.csv"
            stats.write_text(
                "query_id,n_dist,baseline_query_latency_ns,trace_query_latency_ns,"
                "exact_distance_time_ns,recall_at_k\n0,12,1,2,3,1\n",
                encoding="utf-8",
            )
            text = MODULE.normalized_query_stats(stats).decode("utf-8")
            self.assertEqual(text, "query_id,n_dist,recall_at_k\n0,12,1\n")


if __name__ == "__main__":
    unittest.main()
