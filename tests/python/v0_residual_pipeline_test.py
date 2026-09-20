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
from common import mark_complete, sha256, write_json  # noqa: E402
from capture_current_trace import main as capture_current_trace  # noqa: E402
from encode_sketch import edge_dtype, open_edges  # noqa: E402
from run_active_matrix import quality_cases, main as run_active_matrix  # noqa: E402
from select_matched_recall import main as select_matched_recall  # noqa: E402
from select_residual_tuning import main as select_residual_tuning  # noqa: E402
from summarize_active import summarize, main as summarize_active  # noqa: E402
from summarize_residual_tuning import summarize as summarize_residual_tuning  # noqa: E402


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

    def test_residual_tuning_grid_excludes_ef475_and_theta126(self):
        config = json.loads((Path(__file__).resolve().parents[2] /
                             "configs/v0/residual_estimator/residual_tuning_quality_v1.json")
                            .read_text())
        cases = quality_cases(config, {})
        self.assertEqual(len(cases), 17)
        threshold = cases[:-1]
        self.assertEqual({case["ef"] for case in threshold}, {500, 525, 550, 575})
        self.assertTrue(all(1.0 < case["theta"] <= 1.14 for case in threshold))
        self.assertEqual(cases[-1], {"method": "residual-direct", "ef": 600,
                                     "theta": 1.0})

    def test_residual_tuning_supplement_has_only_three_new_cases(self):
        config = json.loads((Path(__file__).resolve().parents[2] /
                             "configs/v0/residual_estimator/residual_tuning_supplement_quality_v1.json")
                            .read_text())
        self.assertEqual(quality_cases(config, {}), [
            {"method": "residual-threshold", "ef": 475, "theta": 1.08},
            {"method": "residual-threshold", "ef": 475, "theta": 1.10},
            {"method": "residual-threshold", "ef": 500, "theta": 1.04},
        ])

    def test_residual_tuning_fine_grid_has_only_two_new_cases(self):
        config = json.loads((Path(__file__).resolve().parents[2] /
                             "configs/v0/residual_estimator/residual_tuning_fine_quality_v1.json")
                            .read_text())
        self.assertEqual(quality_cases(config, {}), [
            {"method": "residual-threshold", "ef": 475, "theta": 1.07},
            {"method": "residual-threshold", "ef": 500, "theta": 1.05},
        ])

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
            write_json(config, {"matched_cases": [
                {"method": "residual-direct", "matched": True}],
                "require_matched": True, "repeats": 3, "blocks": 2})
            with patch("sys.argv", argv), self.assertRaisesRegex(
                    ValueError, "at least five repeats"):
                run_active_matrix()

    def test_residual_tuning_selects_feasible_and_summarizes_without_pq(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            quality_dir, qps_dir = root / "quality", root / "qps"
            quality_dir.mkdir()
            qps_dir.mkdir()
            quality_config, qps_config = root / "quality-config.json", root / "qps-config.json"
            split, contract = root / "split.json", root / "contract.json"
            selected = root / "selected.json"
            write_json(quality_config, {"residual_theta_by_ef": {
                "500": [1.08], "525": [1.10]}, "direct_anchor_ef": 600})
            write_json(qps_config, {"recall_floor": 0.9555, "blocks": 2,
                                    "repeats": 5, "query_count": 3})
            write_json(split, {"development": [0], "selection": [1], "audit": [2]})
            write_json(contract, {"split_path": str(split)})
            ids = quality_dir / "quality-query-ids.txt"
            ids.write_text("0\n1\n", encoding="utf-8")
            cases = quality_cases(json.loads(quality_config.read_text()), {})
            runs = []
            for index, (case, recall) in enumerate(zip(cases, (.95, .96, .9555))):
                directory = quality_dir / f"case-{index}"
                directory.mkdir()
                result = directory / "summary.json"
                write_json(result, {"status": "valid"})
                (directory / "query_metrics.csv").write_text(
                    f"query_id,v0_recall_at_k\n0,{recall}\n1,{recall}\n",
                    encoding="utf-8")
                runs.append({"case": case, "result": str(result),
                             "sha256": sha256(result)})
            quality_manifest = quality_dir / "manifest.json"
            write_json(quality_manifest, {"stage": "quality", "runs": runs,
                "config": str(quality_config), "contract": str(contract),
                "companion_sha256": "companion", "quality_selection_scope":
                "development+selection", "query_id_file_sha256": sha256(ids)})
            mark_complete(quality_dir, {}, {"manifest": sha256(quality_manifest)})
            argv = ["select_residual_tuning.py", "--quality",
                    str(quality_manifest), "--config", str(qps_config),
                    "--output", str(selected)]
            with patch("sys.argv", argv):
                select_residual_tuning()
            frozen = json.loads(selected.read_text())
            self.assertEqual(len(frozen["matched_cases"]), 2)
            self.assertEqual({case["method"] for case in frozen["matched_cases"]},
                             {"residual-threshold", "residual-direct"})
            self.assertEqual(json.loads(selected.with_suffix(".complete.json").read_text())
                             ["output_sha256"], sha256(selected))
            qps_runs = []
            for block in range(2):
                for case in frozen["matched_cases"]:
                    result = qps_dir / f"{block}-{case['method']}.json"
                    write_json(result, {"method": case["method"],
                        "ef_search": case["ef"], "theta": case["theta"],
                        "query_count": 3, "repeats": 5,
                        "qps": 120 if case["method"] == "residual-threshold" else 100,
                        "latency_p95_ns": 1000, "latency_p99_ns": 1200})
                    qps_runs.append({"block": block, "case": case,
                                     "result": str(result),
                                     "sha256": sha256(result)})
            qps_manifest = qps_dir / "manifest.json"
            write_json(qps_manifest, {"stage": "qps", "config": str(selected),
                                     "companion_sha256": "companion",
                                     "runs": qps_runs})
            mark_complete(qps_dir, {"config": sha256(selected)},
                          {"manifest": sha256(qps_manifest)})
            report = summarize_residual_tuning(str(quality_manifest),
                                               str(qps_manifest))
            self.assertEqual(report["winner"]["ef"], 525)
            self.assertEqual(report["winner"]["theta"], 1.10)
            self.assertEqual(len(report["results"]), 2)
            qps_runs.pop()
            write_json(qps_manifest, {"stage": "qps", "config": str(selected),
                                     "companion_sha256": "companion",
                                     "runs": qps_runs})
            mark_complete(qps_dir, {"config": sha256(selected)},
                          {"manifest": sha256(qps_manifest)})
            with self.assertRaisesRegex(ValueError, "incomplete paired"):
                summarize_residual_tuning(str(quality_manifest),
                                          str(qps_manifest))

    def test_residual_tuning_qps_uses_supported_runner_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, contract, selection, companion = (root / name for name in
                ("cases.json", "contract.json", "selection.json", "companion.v0res"))
            cases = [{"method": "residual-threshold", "ef": 525,
                      "theta": 1.10, "matched": True},
                     {"method": "residual-direct", "ef": 600,
                      "theta": 1.0, "matched": True}]
            write_json(config, {"matched_cases": cases, "require_matched": True,
                                "blocks": 2, "repeats": 5, "k": 10,
                                "query_start": 0, "query_count": 3,
                                "warmup_queries": 1, "prefetch": "legacy"})
            write_json(contract, {"config": {"dimension": 960}, "assets": {
                key: {"path": key} for key in ("index", "queries", "sidecar")}})
            write_json(selection, {})
            companion.write_bytes(b"V0RES001" + bytes(12) +
                                  (128).to_bytes(4, "little"))
            calls = []

            def fake_run(command, check):
                self.assertTrue(check)
                calls.append(command)
                target = Path(command[command.index("--output") + 1])
                write_json(target, {"qps": 100.0})

            output = root / "qps"
            argv = ["run_active_matrix.py", "--stage", "qps",
                    "--config", str(config), "--contract", str(contract),
                    "--selection", str(selection), "--runner", "runner",
                    "--companion", str(companion), "--output", str(output)]
            with patch("sys.argv", argv), patch(
                    "run_active_matrix.subprocess.run", side_effect=fake_run):
                run_active_matrix()
            self.assertEqual(len(calls), 4)
            for command in calls:
                self.assertEqual(command[command.index("--repeats") + 1], "5")
                self.assertEqual(command[command.index("--prefetch") + 1],
                                 "legacy")
                self.assertIn("--companion-path", command)
                self.assertIn("--theta", command)
            self.assertTrue((output / "complete.json").is_file())

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

    def test_qps_summary_checks_matched_config_not_a1_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            quality, matched, a1, qps, output = (root / name for name in
                ("quality.json", "matched.json", "a1.json", "qps.json",
                 "summary.json"))
            write_json(quality, {"stage": "quality"})
            write_json(matched, {"source_quality_sha256": sha256(quality)})
            write_json(a1, {"selected_bits": [128]})
            runs = []
            for block in range(2):
                for method in ("baseline", "approx-no-retry", "residual-direct"):
                    result = root / f"{block}-{method}.json"
                    write_json(result, {"qps": 100.0, "latency_p95_ns": 1000,
                                        "latency_p99_ns": 1200})
                    runs.append({"block": block, "case": {
                        "target_recall": 0.9555, "method": method,
                        "matched": True}, "result": str(result),
                        "sha256": sha256(result)})
            write_json(qps, {"stage": "qps", "config": str(matched),
                             "selection": str(a1), "runs": runs})
            argv = ["summarize_active.py", "--quality", str(quality),
                    "--qps", str(qps), "--output", str(output)]
            with patch("sys.argv", argv):
                summarize_active()
            self.assertEqual(len(json.loads(output.read_text())["results"]), 3)
            write_json(matched, {"source_quality_sha256": "wrong"})
            with patch("sys.argv", argv), self.assertRaisesRegex(
                    ValueError, "quality manifest differs"):
                summarize_active()


if __name__ == "__main__":
    unittest.main()
