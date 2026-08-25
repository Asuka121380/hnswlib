#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "v0" / "performance_ready"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PerformanceReadyInfrastructureTest(unittest.TestCase):
    def test_balanced_orders_cover_each_position(self) -> None:
        module = load_module("run_latin_square", SCRIPTS / "run_latin_square.py")
        self.assertEqual(len(set(module.BALANCED_ORDERS)), 6)
        for position in range(3):
            counts = {method: 0 for method in module.METHODS}
            for order in module.BALANCED_ORDERS:
                counts[order[position]] += 1
            self.assertEqual(set(counts.values()), {2})

    def test_break_even_pass_and_fail(self) -> None:
        micro = {
            "fast_lut_build_ns": 10,
            "fast_estimator_ns": 1,
            "state_reset_ns": 1,
            "state_mark_ns": 1,
            "direct_record_ns": 1,
            "exact_l2_ns": 100,
        }
        metrics = {
            "query_count": 1,
            "approx_eligible_first_visits": 1,
            "approx_first_pruned": 1,
            "edge_scans": 1,
            "exact_distance_saved": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            micro_path = root / "micro.json"
            metrics_path = root / "metrics.json"
            output_path = root / "gate.json"
            micro_path.write_text(json.dumps(micro), encoding="utf-8")
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            command = [
                sys.executable, str(SCRIPTS / "evaluate_break_even.py"),
                "--microbenchmark", str(micro_path),
                "--metrics-summary", str(metrics_path),
                "--output", str(output_path),
            ]
            self.assertEqual(subprocess.run(command).returncode, 0)
            self.assertEqual(json.loads(output_path.read_text())["status"], "PASS")
            micro["exact_l2_ns"] = 1
            micro_path.write_text(json.dumps(micro), encoding="utf-8")
            self.assertEqual(subprocess.run(command).returncode, 1)
            self.assertEqual(json.loads(output_path.read_text())["status"], "FAIL")

    def test_fast_source_audit(self) -> None:
        completed = subprocess.run([
            sys.executable, str(SCRIPTS / "audit_fast_path_source.py"),
            "--header", str(ROOT / "hnswlib" / "edge_quant_v0.h"),
        ])
        self.assertEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
