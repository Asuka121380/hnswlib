"""Finite active matrix and paired summary regression tests."""

import sys
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] /
                       "scripts" / "v0" / "residual_estimator"))
from common import sha256, write_json  # noqa: E402
from capture_current_trace import main as capture_current_trace  # noqa: E402
from encode_sketch import edge_dtype, open_edges  # noqa: E402
from run_active_matrix import quality_cases  # noqa: E402
from summarize_active import summarize  # noqa: E402


class PipelineTest(unittest.TestCase):
    def test_current_shadow_loads_index_once_per_ef(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            contract = root / "contract.json"
            split = root / "split.json"
            output = root / "shadow"
            write_json(config, {"ef_search": [435, 500], "k": 10,
                                "development_queries": 3})
            write_json(split, {"development": [1, 4, 7]})
            write_json(contract, {"config": {"split_seed": 42},
                                  "split_path": str(split),
                                  "assets": {key: {"path": key} for key in
                                             ("dataset_config", "index", "sidecar")}})
            calls = []

            def fake_run(command, check):
                self.assertTrue(check)
                calls.append(command)
                target = Path(command[command.index("--output-dir") + 1])
                target.mkdir(parents=True)
                write_json(target / "summary.json", {"status": "valid"})

            argv = ["capture_current_trace.py", "--config", str(config),
                    "--contract", str(contract), "--runner", "runner",
                    "--output", str(output)]
            with patch("sys.argv", argv), patch(
                    "capture_current_trace.subprocess.run", side_effect=fake_run):
                capture_current_trace()
            self.assertEqual(len(calls), 2)
            ids = output / "development-query-ids.txt"
            self.assertEqual(ids.read_text().splitlines(), ["1", "4", "7"])
            for command in calls:
                self.assertEqual(command[command.index("--query-id-file") + 1],
                                 str(ids))

    def test_edge_geometry_layout_and_original_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "edges.bin"
            rows = np.zeros(1, dtype=edge_dtype(4, 2))
            rows["ordinal"][0] = 7
            rows["code"][0] = [13, 29]
            with path.open("wb") as output:
                output.write(b"V0RGE002" + struct.pack("<IQI", 4, 1, 2))
                output.write(rows.tobytes())
            loaded = open_edges(path, 4)
            self.assertEqual(int(loaded["ordinal"][0]), 7)
            self.assertEqual(loaded["code"][0].tolist(), [13, 29])
            del loaded

    def test_quality_matrix_is_finite(self):
        config = {"ef_search": [350, 500], "beta_values": [1.3, 1.45]}
        selection = {"selected_bits": [64], "choices": [
            {"bits": 64, "primary_theta": 1.08}]}
        cases = quality_cases(config, selection)
        self.assertEqual(len(cases), 10)
        self.assertEqual(sum(case["method"] == "residual-direct"
                             for case in cases), 2)

    def test_paired_summary_and_missing_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = []
            for block in range(3):
                for method, qps in (("baseline", 100 + block),
                                    ("approx-no-retry", 120 + block),
                                    ("residual-direct", 130 + block)):
                    result = root / f"{block}-{method}.json"
                    write_json(result, {"qps": qps,
                                        "latency_p95_ns": 1000,
                                        "latency_p99_ns": 1200})
                    runs.append({"block": block,
                                 "case": {"target_recall": 0.95,
                                          "method": method, "matched": True},
                                 "result": str(result), "sha256": sha256(result)})
            manifest = {"stage": "qps", "runs": runs}
            rows = summarize(manifest)
            residual = next(row for row in rows
                            if row["method"] == "residual-direct")
            self.assertGreater(residual["speedup_vs_pq"], 1)
            self.assertEqual(residual["blocks"], 3)
            self.assertEqual(len(residual["speedup_vs_pq_ci95"]), 2)
            manifest["runs"].pop()
            with self.assertRaises(ValueError):
                summarize(manifest)


if __name__ == "__main__":
    unittest.main()
