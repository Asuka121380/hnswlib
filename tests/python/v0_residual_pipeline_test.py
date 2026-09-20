"""Finite active matrix and paired summary regression tests."""

import json
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
from run_active_matrix import quality_cases, main as run_active_matrix  # noqa: E402
from select_matched_recall import main as select_matched_recall  # noqa: E402
from summarize_active import summarize  # noqa: E402


class PipelineTest(unittest.TestCase):
    def test_quality_matrix_excludes_audit_queries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = root / "split.json"
            contract = root / "contract.json"
            selection = root / "selection.json"
            config = root / "config.json"
            companion = root / "companion.v0res"
            output = root / "quality"
            write_json(split, {"development": [0], "selection": [1],
                               "audit": [2]})
            write_json(contract, {"split_path": str(split), "assets": {
                key: {"path": key} for key in
                ("dataset_config", "index", "sidecar")}})
            write_json(selection, {"selected_bits": [64], "choices": [
                {"bits": 64, "primary_theta": 1.1}]})
            write_json(config, {"ef_search": [350], "beta_values": [1.3],
                                "k": 10, "query_start": 0, "query_count": 3})
            companion.write_bytes(b"V0RES001" + bytes(12) +
                                  (64).to_bytes(4, "little"))
            calls = []

            def fake_run(command, check):
                self.assertTrue(check)
                calls.append(command)
                target = Path(command[command.index("--output-dir") + 1])
                write_json(target / "summary.json", {"status": "valid"})

            argv = ["run_active_matrix.py", "--stage", "quality",
                    "--config", str(config), "--contract", str(contract),
                    "--selection", str(selection), "--runner", "runner",
                    "--companion", str(companion), "--output", str(output)]
            with patch("sys.argv", argv), patch(
                    "run_active_matrix.subprocess.run", side_effect=fake_run):
                run_active_matrix()
            self.assertEqual(len(calls), 4)
            self.assertEqual((output / "quality-query-ids.txt").read_text()
                             .splitlines(), ["0", "1"])
            self.assertTrue(all("--query-id-file" in command for command in calls))

    def test_matched_recall_excludes_audit_queries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split, contract = root / "split.json", root / "contract.json"
            quality, config = root / "quality.json", root / "config.json"
            output = root / "matched.json"
            write_json(split, {"development": [0], "selection": [1],
                               "audit": [2]})
            write_json(contract, {"split_path": str(split)})
            write_json(config, {"target_recalls": [0.95],
                                "match_tolerance": 0.01})
            runs = []
            for method, ef, selection_recall, audit_recall in (
                    ("baseline", 350, 0.95, 0.0),
                    ("baseline", 500, 0.90, 1.0),
                    ("approx-no-retry", 350, 0.95, 0.0),
                    ("residual-direct", 350, 0.95, 0.0)):
                directory = root / f"{method}-{ef}"
                directory.mkdir()
                result = directory / "summary.json"
                write_json(result, {"mean_v0_recall_at_k": audit_recall})
                (directory / "query_metrics.csv").write_text(
                    "query_id,v0_recall_at_k\n"
                    f"0,{selection_recall}\n1,{selection_recall}\n"
                    f"2,{audit_recall}\n", encoding="utf-8")
                runs.append({"case": {"method": method, "ef": ef},
                             "result": str(result)})
            write_json(quality, {"stage": "quality", "contract": str(contract),
                                 "runs": runs})
            argv = ["select_matched_recall.py", "--quality", str(quality),
                    "--config", str(config), "--output", str(output)]
            with patch("sys.argv", argv):
                select_matched_recall()
            selected = json.loads(output.read_text())
            self.assertEqual(selected["quality_selection_scope"],
                             "development+selection")
            self.assertEqual(selected["quality_selection_query_count"], 2)
            self.assertEqual(selected["quality_selection"][0]["points"][0]
                             ["case"]["ef"], 350)

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

    def test_preliminary_quality_grid_has_three_methods(self):
        config = json.loads((Path(__file__).resolve().parents[2] /
                             "configs/v0/residual_estimator/active_quality_v1.json")
                            .read_text())
        cases = quality_cases(config, {})
        self.assertEqual(len(cases), 11)
        self.assertEqual([case["method"] for case in cases],
                         ["baseline"] * 5 + ["approx-no-retry"] * 3 +
                         ["residual-direct"] * 3)
        self.assertEqual({case["beta"] for case in cases
                          if case["method"] == "approx-no-retry"}, {1.45})
        self.assertTrue(all(case["theta"] == 1.0 for case in cases
                            if case["method"] == "residual-direct"))

    def test_preliminary_qps_rejects_unmatched_recall(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, contract, selection = (root / name for name in
                                           ("config.json", "contract.json",
                                            "selection.json"))
            write_json(config, {"matched_cases": [
                {"method": "residual-direct", "matched": False}],
                "require_matched": True})
            write_json(contract, {"assets": {}})
            write_json(selection, {})
            argv = ["run_active_matrix.py", "--stage", "qps",
                    "--config", str(config), "--contract", str(contract),
                    "--selection", str(selection), "--runner", "runner",
                    "--companion", str(root / "companion.v0res"),
                    "--output", str(root / "qps")]
            with patch("sys.argv", argv), self.assertRaisesRegex(
                    ValueError, "recall-matched"):
                run_active_matrix()

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
