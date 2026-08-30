#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "v0" / "performance_ready"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exploratory_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_name": "test-matrix",
        "experiment_role": "exploratory",
        "dataset": "gist1m",
        "dimension": 960,
        "query_start": 0,
        "query_count": 10,
        "k": 10,
        "ef_search": 500,
        "warmup_queries": 2,
        "within_process_repeats": 2,
        "blocks": 2,
        "seed": 17,
        "include_baseline": True,
        "cases": [
            {"beta": 1.3, "mode": "approx-no-retry", "prefetch": "gate"},
            {"beta": 1.35, "mode": "approx-retry", "prefetch": "legacy"},
        ],
    }


class QpsConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module("qps_config_tested", SCRIPTS / "qps_config.py")

    def test_targeted_config_resolves_to_65_results(self) -> None:
        _, resolved = self.module.load_config(
            ROOT / "configs" / "v0" / "qps" / "ef500_targeted.json",
            "formal-exclusive",
        )
        self.assertEqual(resolved["ef_search"], 500)
        self.assertEqual(resolved["active_configuration_count"], 12)
        self.assertEqual(resolved["expected_result_count"], 65)
        self.assertTrue(resolved["exclusive"])

    def test_exploratory_scan_resolves_to_63_results(self) -> None:
        _, resolved = self.module.load_config(
            ROOT / "configs" / "v0" / "qps" / "beta_130_140_scan.json",
            "exploratory-shared",
        )
        self.assertEqual(resolved["active_configuration_count"], 20)
        self.assertEqual(resolved["expected_result_count"], 63)
        self.assertFalse(resolved["exclusive"])

    def test_frozen_full_matrix_maps_to_505_results(self) -> None:
        _, resolved = self.module.load_config(
            ROOT / "configs" / "v0" / "qps" / "examples" /
            "full_beta_ef200_compat.json",
            "formal-exclusive",
        )
        self.assertEqual(resolved["ef_search"], 200)
        self.assertEqual(resolved["active_configuration_count"], 100)
        self.assertEqual(resolved["expected_result_count"], 505)

    def test_explicit_cases_and_contract_hash_are_stable(self) -> None:
        requested = exploratory_config()
        first = self.module.resolve_config(requested, "exploratory-shared")
        second = self.module.resolve_config(
            json.loads(json.dumps(requested)), "exploratory-shared")
        self.assertEqual(first, second)
        self.assertEqual(first["case_source"], "explicit")
        self.assertEqual(first["expected_result_count"], 6)

    def test_formal_experiment_rejects_shared_resources(self) -> None:
        requested = exploratory_config()
        requested["experiment_role"] = "formal"
        requested["blocks"] = 5
        requested["within_process_repeats"] = 5
        with self.assertRaises(self.module.ConfigError):
            self.module.resolve_config(requested, "exploratory-shared")

    def test_ambiguous_and_duplicate_cases_are_rejected(self) -> None:
        requested = exploratory_config()
        requested["betas"] = [1.3]
        requested["modes"] = ["approx-retry"]
        requested["prefetches"] = ["gate"]
        with self.assertRaises(self.module.ConfigError):
            self.module.resolve_config(requested, "exploratory-shared")
        requested = exploratory_config()
        requested["cases"].append(dict(requested["cases"][0]))
        with self.assertRaises(self.module.ConfigError):
            self.module.resolve_config(requested, "exploratory-shared")


class QpsMatrixIntegrationTest(unittest.TestCase):
    def _arguments(self, root: Path, config: Path, run_root: Path) -> list[str]:
        return [
            "run_qps_matrix.py",
            "--config", str(config),
            "--resource-profile", "exploratory-shared",
            "--runner", str(root / "runner"),
            "--index-path", str(root / "index.bin"),
            "--sidecar-path", str(root / "sidecar.bin"),
            "--query-path", str(root / "query.fvecs"),
            "--run-root", str(run_root),
            "--cmake-cache", str(root / "CMakeCache.txt"),
            "--experiment-commit", "abc123",
            "--numa-policy", "test",
            "--power-policy", "test",
            "--cpu-frequency-policy", "test",
        ]

    @staticmethod
    def _fake_runner(command: list[str], check: bool) -> subprocess.CompletedProcess:
        del check
        output = Path(command[command.index("--output") + 1])
        method = command[command.index("--method") + 1]
        beta = command[command.index("--beta") + 1] if "--beta" in command else ""
        prefetch = command[command.index("--prefetch") + 1]
        output.write_text(json.dumps({
            "qps": 100.0 if method == "baseline" else 120.0,
            "latency_p50_ns": 10,
            "latency_p95_ns": 20,
            "latency_p99_ns": 30,
            "result_checksum": f"{method}:{beta}:{prefetch}",
        }), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    def test_end_to_end_manifest_resume_and_summary(self) -> None:
        runner_module = load_module(
            "run_qps_matrix_tested", SCRIPTS / "run_qps_matrix.py")
        summary_module = load_module(
            "summarize_qps_experiment_tested",
            SCRIPTS / "summarize_qps_experiment.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("runner", "index.bin", "sidecar.bin",
                         "query.fvecs", "CMakeCache.txt"):
                (root / name).write_text(name, encoding="utf-8")
            config = root / "config.json"
            config.write_text(
                json.dumps(exploratory_config()), encoding="utf-8")
            run_root = root / "run"
            arguments = self._arguments(root, config, run_root)
            with mock.patch.object(sys, "argv", arguments), mock.patch.object(
                    runner_module.platform, "platform", return_value="test"), \
                    mock.patch.object(
                        runner_module.subprocess, "run",
                        side_effect=self._fake_runner):
                self.assertEqual(runner_module.main(), 0)

            manifest = json.loads(
                (run_root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(manifest["completed"]), 6)
            self.assertEqual(
                manifest["contract"]["resolved_config"]["ef_search"], 500)
            self.assertTrue((run_root / "COMPLETE").is_file())

            with mock.patch.object(sys, "argv", [
                    "summarize_qps_experiment.py", "--run-root", str(run_root)]):
                self.assertEqual(summary_module.main(), 0)
            with (run_root / "qps_summary.csv").open(
                    encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            active = [row for row in rows if row["mode"] != "baseline"]
            self.assertTrue(all(
                abs(float(row["qps_speedup_mean"]) - 1.2) < 1e-9
                for row in active))
            self.assertTrue(all(
                row["resource_profile"] == "exploratory-shared"
                for row in rows))

            resume_arguments = arguments + ["--resume"]
            with mock.patch.object(sys, "argv", resume_arguments), \
                    mock.patch.object(runner_module.subprocess, "run") as rerun:
                self.assertEqual(runner_module.main(), 0)
                rerun.assert_not_called()

    def test_resume_rejects_result_checksum_mismatch(self) -> None:
        runner_module = load_module(
            "run_qps_matrix_checksum_tested", SCRIPTS / "run_qps_matrix.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("runner", "index.bin", "sidecar.bin",
                         "query.fvecs", "CMakeCache.txt"):
                (root / name).write_text(name, encoding="utf-8")
            config = root / "config.json"
            config.write_text(
                json.dumps(exploratory_config()), encoding="utf-8")
            run_root = root / "run"
            arguments = self._arguments(root, config, run_root)
            with mock.patch.object(sys, "argv", arguments), mock.patch.object(
                    runner_module.platform, "platform", return_value="test"), \
                    mock.patch.object(
                        runner_module.subprocess, "run",
                        side_effect=self._fake_runner):
                self.assertEqual(runner_module.main(), 0)
            result = next((run_root / "qps").glob("*.json"))
            result.write_text("{}", encoding="utf-8")
            with mock.patch.object(sys, "argv", arguments + ["--resume"]):
                with self.assertRaises(SystemExit):
                    runner_module.main()


if __name__ == "__main__":
    unittest.main()
