#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "scripts" / "edge_transform_v1" / "gate_b_replay.py"


def write_fvecs(path: Path, vectors: np.ndarray) -> None:
    vectors = np.asarray(vectors, dtype=np.float32)
    records = np.empty((vectors.shape[0], vectors.shape[1] + 1), dtype=np.int32)
    records[:, 0] = vectors.shape[1]
    records[:, 1:] = vectors.view(np.int32)
    records.tofile(path)


class GateBReplayIntegrationTest(unittest.TestCase):
    def test_small_end_to_end_replay_has_no_lb_violation(self) -> None:
        rng = np.random.default_rng(20260810)
        dimension, base_count, query_count, record_count = 8, 256, 18, 720
        base = rng.normal(size=(base_count, dimension)).astype(np.float32)
        queries = rng.normal(size=(query_count, dimension)).astype(np.float32)
        query_ids = np.arange(record_count, dtype=np.int64) % query_count
        current_ids = rng.integers(0, base_count, record_count, dtype=np.int64)
        candidate_ids = rng.integers(0, base_count, record_count, dtype=np.int64)
        candidate_ids[candidate_ids == current_ids] = (candidate_ids[candidate_ids == current_ids] + 1) % base_count
        exact = np.sum((queries[query_ids] - base[candidate_ids]) ** 2, axis=1)
        current_exact = np.sum((queries[query_ids] - base[current_ids]) ** 2, axis=1)
        threshold = np.quantile(exact, 0.55) * np.ones(record_count)

        with tempfile.TemporaryDirectory(prefix="etv1-gate-b-") as temporary:
            root = Path(temporary)
            base_path, query_path = root / "base.fvecs", root / "query.fvecs"
            records_path, split_path, output = root / "records.parquet", root / "split.json", root / "output"
            mapping_path = root / "internal_to_label.npy"
            write_fvecs(base_path, base)
            write_fvecs(query_path, queries)
            np.save(mapping_path, np.arange(base_count, dtype=np.uint64), allow_pickle=False)
            pq.write_table(
                pa.table(
                    {
                        "query_id": query_ids,
                        "current_node_id": current_ids,
                        "candidate_id": candidate_ids,
                        "threshold": threshold,
                        "direction_error": np.full(record_count, 0.25),
                        "current_lb": np.zeros(record_count),
                        "shadow_exact_squared_distance": exact,
                        "approximate_squared_distance": current_exact,
                    }
                ),
                records_path,
            )
            split_path.write_text(
                json.dumps(
                    {
                        "query_ids": {
                            "train": list(range(0, 10)),
                            "validation": list(range(10, 14)),
                            "test": list(range(14, 18)),
                        }
                    }
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--records", str(records_path),
                    "--base-fvecs", str(base_path),
                    "--query-fvecs", str(query_path),
                    "--split-manifest", str(split_path),
                    "--internal-to-label", str(mapping_path),
                    "--output-dir", str(output),
                    "--m", "2",
                    "--ksub", "8",
                    "--train-records", "300",
                    "--opq-iterations", "1",
                    "--kmeans-iterations", "5",
                    "--batch-size", "128",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            report = json.loads((output / "gate_b_report.json").read_text(encoding="utf-8"))
            self.assertEqual(set(report["methods"]), {"identity_pq", "random_jq", "edge_opq", "etv1_weighted_opq"})
            self.assertIn(report["gate_b"]["decision"], {"GO_ETV1", "GO_EDGE_OPQ", "NO_GO_CPP"})
            for method in report["methods"].values():
                for split in method.values():
                    for pivot in split.values():
                        self.assertEqual(pivot["new_lb_violations_at_1e_8"], 0)


if __name__ == "__main__":
    unittest.main()
