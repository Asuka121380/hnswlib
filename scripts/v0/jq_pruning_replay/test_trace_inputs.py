#!/usr/bin/env python3

from __future__ import annotations

import csv
import gzip
import tempfile
from pathlib import Path
import unittest

import numpy as np

from trace_inputs import FvecsMemmap, load_internal_to_label, load_shadow_records


def _write_fvecs(path: Path, values: np.ndarray) -> None:
    with path.open("wb") as handle:
        for row in np.asarray(values, dtype="<f4"):
            handle.write(np.asarray([row.size], dtype="<i4").tobytes())
            handle.write(row.tobytes())


class TraceInputsTest(unittest.TestCase):
    def test_fvecs_and_non_identity_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vectors = np.arange(24, dtype=np.float32).reshape(6, 4)
            path = root / "vectors.fvecs"
            _write_fvecs(path, vectors)
            reader = FvecsMemmap(path, expected_dimension=4)
            np.testing.assert_array_equal(reader.take(np.asarray([5, 1])), vectors[[5, 1]])
            reader.close()
            mapping_path = root / "mapping.npy"
            np.save(mapping_path, np.asarray([2, 5, 1, 4, 0, 3], dtype=np.uint64))
            mapping = load_internal_to_label(mapping_path, expected_count=6)
            self.assertFalse(np.array_equal(mapping, np.arange(6)))

    def test_load_old_shadow_schema(self) -> None:
        fields = [
            "query_id", "current_node_id", "candidate_id", "bound_status",
            "current_squared_distance", "threshold", "approximate_squared_distance",
            "error_radius", "lower_bound", "shadow_exact_squared_distance",
            "would_prune", "lower_bound_valid", "lower_bound_violation", "false_prune",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "shadow.csv.gz"
            with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "query_id": 0, "current_node_id": 3, "candidate_id": 4,
                    "bound_status": 0, "current_squared_distance": 1.0,
                    "threshold": 2.0, "approximate_squared_distance": 3.0,
                    "error_radius": 0.5, "lower_bound": 2.5,
                    "shadow_exact_squared_distance": 4.0, "would_prune": 1,
                    "lower_bound_valid": 1, "lower_bound_violation": 0,
                    "false_prune": 0,
                })
            records = load_shadow_records(path)
            self.assertEqual(records.count, 1)
            self.assertTrue(records.recorded_would_prune[0])


if __name__ == "__main__":
    unittest.main()
