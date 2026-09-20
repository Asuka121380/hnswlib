"""Frozen recall anchors and paired curve QPS regression checks."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] /
                       "scripts" / "v0" / "residual_estimator"))
from common import mark_complete, sha256, write_json  # noqa: E402
from select_curve_qps import freeze  # noqa: E402
from summarize_curve_qps import summarize  # noqa: E402


class CurveQpsTest(unittest.TestCase):
    def test_five_matched_anchors_and_paired_curve(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_config = Path(__file__).resolve().parents[2] / \
                "configs/v0/residual_estimator/curve_qps_v1.json"
            config = json.loads(source_config.read_text())
            config_path = root / "curve.json"
            write_json(config_path, config)
            split = root / "split.json"
            contract = root / "contract.json"
            write_json(split, {"development": list(range(600)),
                               "selection": list(range(600, 800)),
                               "audit": list(range(800, 1000))})
            write_json(contract, {"split_path": str(split)})
            recalls = {
                ("baseline", 350): .94637, ("baseline", 405): .95575,
                ("baseline", 435): .95937, ("baseline", 500): .965,
                ("baseline", 600): .97175,
                ("approx-no-retry", 400): .94587,
                ("approx-no-retry", 465): .95537,
                ("approx-no-retry", 500): .95862,
                ("approx-no-retry", 570): .96487,
                ("approx-no-retry", 700): .97262,
                ("residual-direct", 500): .94575,
                ("residual-direct", 600): .95550,
                ("residual-direct", 650): .95975,
                ("residual-direct", 750): .96537,
                ("residual-direct", 935): .97112,
                ("residual-threshold", 425, 1.04): .946,
                ("residual-threshold", 500, 1.04): .95475,
                ("residual-threshold", 500, 1.08): .95937,
                ("residual-threshold", 575, 1.04): .95888,
                ("residual-threshold", 500, 1.14): .96425,
                ("residual-threshold", 525, 1.12): .96537,
                ("residual-threshold", 690, 1.08): .971,
            }
            source_points = {name: {} for name in config["quality_sources"]}
            for anchor in config["anchors"]:
                for spec in [anchor["baseline"], *anchor["candidates"]]:
                    case = {key: value for key, value in spec.items()
                            if key != "source"}
                    identity = (case["method"], case["ef"])
                    if case["method"] == "residual-threshold":
                        identity += (case["theta"],)
                    source_points[spec["source"]][tuple(sorted(case.items()))] = \
                        recalls[identity]
            for directory in config["quality_sources"].values():
                (root / directory).mkdir()
                write_json(root / directory / "manifest.json", {})

            def checked(path):
                name = Path(path).parent.name
                source = next(key for key, directory in
                              config["quality_sources"].items() if directory == name)
                return ({"contract": str(contract)}, source_points[source],
                        {"contract_sha256": "same", "query_ids_sha256": "ids",
                         "companion_sha256": "companion",
                         "selection_sha256": "selection"})

            with patch("select_curve_qps.checked_quality", side_effect=checked):
                frozen = freeze(str(root), str(config_path))
                self.assertEqual(len(frozen["matched_cases"]), 22)
                self.assertEqual({case["anchor_ef"] for case in
                                  frozen["matched_cases"]},
                                 {350, 405, 435, 500, 600})
                source_points["refine"][tuple(sorted({
                    "method": "residual-direct", "ef": 935,
                    "theta": 1.0}.items()))] = .969
                with self.assertRaisesRegex(ValueError, "outside recall window"):
                    freeze(str(root), str(config_path))

            selected = root / "curve-cases.json"
            frozen["blocks"] = 2
            write_json(selected, frozen)
            write_json(selected.with_suffix(".complete.json"),
                       {"output_sha256": sha256(selected)})
            qps_dir = root / "qps"
            qps_dir.mkdir()
            runs = []
            for block in range(2):
                for index, case in enumerate(frozen["matched_cases"]):
                    result = qps_dir / f"{block}-{index}.json"
                    speed = {"baseline": 100, "approx-no-retry": 120,
                             "residual-direct": 125,
                             "residual-threshold": 130}[case["method"]]
                    if case["method"] == "residual-threshold" and \
                            case["anchor_ef"] == 435 and case["ef"] == 575:
                        speed = 123
                    write_json(result, {"method": case["method"],
                        "ef_search": case["ef"], "theta": case.get("theta", 1.0),
                        "beta": case.get("beta", 0.0), "query_count": 1000,
                        "repeats": 5, "qps": speed + block,
                        "latency_p95_ns": 1000, "latency_p99_ns": 1200})
                    runs.append({"block": block, "case": case,
                                 "result": str(result), "sha256": sha256(result)})
            qps_manifest = qps_dir / "manifest.json"
            write_json(qps_manifest, {"stage": "qps", "config": str(selected),
                                      "companion_sha256": "companion",
                                      "runs": runs})
            mark_complete(qps_dir, {"config": sha256(selected)},
                          {"manifest": sha256(qps_manifest)})
            report = summarize(str(qps_manifest))
            self.assertEqual(len(report["results"]), 22)
            self.assertEqual(sum(row["curve_point"] for row in
                                 report["results"]), 20)
            self.assertFalse(next(row for row in report["results"] if
                                  row["anchor_ef"] == 435 and row["ef"] == 575)
                             ["curve_point"])
            runs.pop()
            write_json(qps_manifest, {"stage": "qps", "config": str(selected),
                                      "companion_sha256": "companion",
                                      "runs": runs})
            mark_complete(qps_dir, {"config": sha256(selected)},
                          {"manifest": sha256(qps_manifest)})
            with self.assertRaisesRegex(ValueError, "incomplete paired"):
                summarize(str(qps_manifest))


if __name__ == "__main__":
    unittest.main()
