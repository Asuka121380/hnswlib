"""Regression checks for recall-window selection and paired QPS."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] /
                       "scripts" / "v0" / "residual_estimator"))
from common import mark_complete, sha256, write_json  # noqa: E402
from run_active_matrix import quality_cases  # noqa: E402
from select_targeted_qps import freeze  # noqa: E402
from summarize_targeted_qps import summarize  # noqa: E402


class TargetedQpsTest(unittest.TestCase):
    def test_five_cases_use_baseline_recall_window_and_paired_qps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split, contract, selection = (root / name for name in
                                          ("split.json", "contract.json",
                                           "selection.json"))
            write_json(split, {"development": [0], "selection": [1], "audit": [2]})
            write_json(contract, {"split_path": str(split)})
            write_json(selection, {"selected_bits": [128]})
            source_specs = {
                "active": ({"ef_search_by_method": {"baseline": [405],
                           "approx-no-retry": [465], "residual-direct": [600]},
                           "beta_values": [1.45]}, [.95575, .95537, .95550]),
                "supplement": ({"residual_theta_by_ef": {"500": [1.04]}},
                               [.95475]),
                "fine": ({"residual_theta_by_ef": {"500": [1.05]}},
                         [.95537]),
            }
            manifests = {}
            for name, (config, recalls) in source_specs.items():
                directory = root / name
                directory.mkdir()
                config_path = directory / "config.json"
                write_json(config_path, config)
                ids = directory / "quality-query-ids.txt"
                ids.write_text("0\n1\n", encoding="utf-8")
                runs = []
                for index, (case, recall) in enumerate(
                        zip(quality_cases(config, {}), recalls)):
                    case_dir = directory / f"case-{index}"
                    case_dir.mkdir()
                    result = case_dir / "summary.json"
                    write_json(result, {"status": "valid"})
                    (case_dir / "query_metrics.csv").write_text(
                        f"query_id,v0_recall_at_k\n0,{recall}\n1,{recall}\n",
                        encoding="utf-8")
                    runs.append({"case": case, "result": str(result),
                                 "sha256": sha256(result)})
                manifest = directory / "manifest.json"
                write_json(manifest, {"stage": "quality", "runs": runs,
                    "config": str(config_path), "contract": str(contract),
                    "selection": str(selection), "companion_sha256": "companion",
                    "quality_selection_scope": "development+selection",
                    "query_id_file_sha256": sha256(ids)})
                mark_complete(directory, {"config": sha256(config_path),
                                         "selection": sha256(selection)},
                              {"manifest": sha256(manifest)})
                manifests[name] = str(manifest)
            config_path = root / "targeted.json"
            write_json(config_path, {
                "baseline_ef": 405, "recall_tolerance": .001,
                "candidates": [
                    {"source": "active", "method": "approx-no-retry",
                     "ef": 465, "beta": 1.45},
                    {"source": "supplement", "method": "residual-threshold",
                     "ef": 500, "theta": 1.04},
                    {"source": "fine", "method": "residual-threshold",
                     "ef": 500, "theta": 1.05},
                    {"source": "active", "method": "residual-direct",
                     "ef": 600, "theta": 1.0},
                ], "blocks": 2, "repeats": 5, "query_count": 3})
            frozen = freeze(manifests["active"], manifests["supplement"],
                            manifests["fine"], str(config_path))
            self.assertEqual(frozen["baseline_recall"], .95575)
            self.assertEqual(len(frozen["matched_cases"]), 5)
            self.assertEqual(frozen["matched_cases"][2]["quality_recall"],
                             .95475)
            selected = root / "targeted-cases.json"
            write_json(selected, frozen)
            write_json(selected.with_suffix(".complete.json"),
                       {"output_sha256": sha256(selected)})
            qps_dir = root / "qps"
            qps_dir.mkdir()
            runs = []
            for block in range(2):
                for index, case in enumerate(frozen["matched_cases"]):
                    result = qps_dir / f"{block}-{index}.json"
                    qps = (100, 117, 125, 128, 116)[index] + block
                    write_json(result, {"method": case["method"],
                        "ef_search": case["ef"], "theta": case.get("theta", 1.0),
                        "beta": case.get("beta", 0.0), "query_count": 3,
                        "repeats": 5, "qps": qps,
                        "latency_p95_ns": 1000, "latency_p99_ns": 1200})
                    runs.append({"block": block, "case": case,
                                 "result": str(result), "sha256": sha256(result)})
            qps_manifest = qps_dir / "manifest.json"
            write_json(qps_manifest, {"stage": "qps", "config": str(selected),
                                      "companion_sha256": "companion", "runs": runs})
            mark_complete(qps_dir, {"config": sha256(selected)},
                          {"manifest": sha256(qps_manifest)})
            report = summarize(str(qps_manifest))
            self.assertEqual(len(report["results"]), 5)
            self.assertAlmostEqual(report["threshold_for_25_percent_gain_qps"],
                                   125.625)
            self.assertEqual(report["results"][0]["theta"], 1.05)
            runs.pop()
            write_json(qps_manifest, {"stage": "qps", "config": str(selected),
                                      "companion_sha256": "companion", "runs": runs})
            mark_complete(qps_dir, {"config": sha256(selected)},
                          {"manifest": sha256(qps_manifest)})
            with self.assertRaisesRegex(ValueError, "incomplete paired"):
                summarize(str(qps_manifest))


if __name__ == "__main__":
    unittest.main()
