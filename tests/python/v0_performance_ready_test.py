#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import csv
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
    def test_build_contract_distinguishes_portable_and_native(self) -> None:
        required = {
            "CMAKE_BUILD_TYPE": "Release",
            "HNSWLIB_ENABLE_EDGE_QUANT_V0": "ON",
            "HNSWLIB_ENABLE_V0_APPROX_REAL_PRUNING": "ON",
            "HNSWLIB_ENABLE_V0_STRICT_FP_CONTRACT": "OFF",
            "HNSWLIB_PERFORMANCE_COMPARABLE_FLAGS": "ON",
            "HNSWLIB_ENABLE_BASELINE_TRACE": "OFF",
            "HNSWLIB_ENABLE_V0_SHADOW_VALIDATION": "OFF",
            "HNSWLIB_ENABLE_V0_APPROX_SHADOW": "OFF",
        }
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            cache = build / "CMakeCache.txt"
            for variant, native in (("off", "OFF"), ("on", "ON")):
                values = dict(required, HNSWLIB_ENABLE_NATIVE_ARCH=native)
                cache.write_text("".join(
                    f"{key}:STRING={value}\n" for key, value in values.items()),
                    encoding="utf-8")
                completed = subprocess.run([
                    sys.executable, str(SCRIPTS / "check_build_contract.py"),
                    "--build-dir", str(build), "--contract", "performance",
                    "--native-arch", variant,
                ])
                self.assertEqual(completed.returncode, 0)
                wrong = "on" if variant == "off" else "off"
                completed = subprocess.run([
                    sys.executable, str(SCRIPTS / "check_build_contract.py"),
                    "--build-dir", str(build), "--contract", "performance",
                    "--native-arch", wrong,
                ])
                self.assertEqual(completed.returncode, 1)

    def test_component_microbenchmark_summary_uses_block_median(self) -> None:
        module = load_module(
            "run_component_microbenchmark",
            SCRIPTS / "run_component_microbenchmark.py")
        raw = []
        for block in range(5):
            item = {
                "kernel": "raw_fast_v1", "dimension": 960,
                "pq_m": 32, "pq_ksub": 256, "iterations": 100,
                "warmup_iterations": 10, "working_set": 64,
                "state_bytes_per_query": 2000000,
            }
            item.update({metric: float(block + 1) for metric in module.METRICS})
            raw.append(item)
        summary = module.summarize(raw)
        self.assertEqual(summary["blocks"], 5)
        self.assertEqual(summary["fast_estimator_ns"], 3.0)
        self.assertEqual(
            summary["distributions"]["fast_estimator_ns"]["values"],
            [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_balanced_orders_cover_each_position(self) -> None:
        module = load_module("run_latin_square", SCRIPTS / "run_latin_square.py")
        self.assertEqual(len(set(module.BALANCED_ORDERS)), 6)
        for position in range(3):
            counts = {method: 0 for method in module.METHODS}
            for order in module.BALANCED_ORDERS:
                counts[order[position]] += 1
            self.assertEqual(set(counts.values()), {2})

    def test_full_beta_qps_schedule_is_complete_per_block(self) -> None:
        module = load_module(
            "run_full_beta_qps", SCRIPTS / "run_full_beta_qps.py")
        configs = module.build_configurations(
            ["1.00", "1.50"],
            ["approx-no-retry", "approx-retry"],
            ["legacy", "gate"],
        )
        self.assertEqual(len(configs), 9)
        schedule = module.build_schedule(configs, blocks=5, seed=17)
        expected_ids = {config["id"] for config in configs}
        self.assertEqual(len(schedule), 5)
        for block in schedule:
            self.assertEqual({config["id"] for config in block}, expected_ids)
            self.assertEqual(
                sum(config["method"] == "baseline" for config in block), 1)

    def test_full_beta_summary_pairs_qps_by_block(self) -> None:
        metric_fields = (
            "beta", "mode", "status", "query_count", "baseline_recall",
            "v0_recall", "recall_loss", "recall_loss_queries",
            "catastrophic_queries", "dco_reduction", "exact_distance_saved",
            "first_pruned", "retry_exact", "decision_disagreement",
            "near_threshold_disagreement", "relative_difference_max",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_path = root / "metrics.csv"
            with metrics_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=metric_fields)
                writer.writeheader()
                writer.writerow({
                    "beta": 1.0,
                    "mode": "approx-retry",
                    "status": "valid",
                    "query_count": 1000,
                    "baseline_recall": 0.9,
                    "v0_recall": 0.899,
                    "recall_loss": 0.001,
                    "recall_loss_queries": 2,
                    "catastrophic_queries": 0,
                    "dco_reduction": 0.2,
                    "exact_distance_saved": 20,
                    "first_pruned": 21,
                    "retry_exact": 1,
                    "decision_disagreement": 0,
                    "near_threshold_disagreement": 0,
                    "relative_difference_max": 1e-7,
                })
            qps_dir = root / "qps"
            qps_dir.mkdir()
            completed = []
            for block in range(5):
                for config, qps in (
                    ({"id": "baseline", "method": "baseline",
                      "beta": None, "prefetch": "none"}, 100.0),
                    ({"id": "active", "method": "approx-retry",
                      "beta": "1.00", "prefetch": "gate"}, 120.0),
                ):
                    run_id = f"b{block}-{config['id']}"
                    result = qps_dir / f"{run_id}.json"
                    result.write_text(json.dumps({
                        "qps": qps,
                        "latency_p50_ns": 10,
                        "latency_p95_ns": 20,
                        "latency_p99_ns": 30,
                        "result_checksum": config["id"],
                    }), encoding="utf-8")
                    completed.append({
                        "run_id": run_id,
                        "block": block,
                        "position": 0,
                        "config": config,
                        "result": result.name,
                        "sha256": "unused-by-summary",
                    })
            (qps_dir / "manifest.json").write_text(json.dumps({
                "status": "complete",
                "contract": {
                    "experiment_commit": "abc",
                    "blocks": 5,
                    "betas": ["1.00"],
                    "modes": ["approx-retry"],
                    "prefetches": ["gate"],
                    "query_count": 1000,
                },
                "completed": completed,
            }), encoding="utf-8")
            output = root / "combined.csv"
            report = root / "report.json"
            completed_process = subprocess.run([
                sys.executable,
                str(SCRIPTS / "summarize_full_beta_tradeoff.py"),
                "--metrics-summary", str(metrics_path),
                "--qps-dir", str(qps_dir),
                "--output", str(output),
                "--report", str(report),
            ])
            self.assertEqual(completed_process.returncode, 0)
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(float(rows[1]["qps_speedup_mean"]), 1.2)
            self.assertGreater(float(rows[1]["qps_speedup_ci_low"]), 1.0)

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
                "--mode", "approx-retry",
                "--output", str(output_path),
            ]
            self.assertEqual(subprocess.run(command).returncode, 0)
            self.assertEqual(json.loads(output_path.read_text())["status"], "PASS")
            micro["exact_l2_ns"] = 1
            micro_path.write_text(json.dumps(micro), encoding="utf-8")
            self.assertEqual(subprocess.run(command).returncode, 1)
            self.assertEqual(json.loads(output_path.read_text())["status"], "FAIL")

    def test_break_even_uses_matched_pair_net_exact_delta(self) -> None:
        micro = {
            "fast_lut_build_ns": 1,
            "fast_estimator_ns": 1,
            "state_mark_ns": 1000,
            "exact_l2_ns": 10,
        }
        active = {
            "query_count": 1,
            "approx_eligible_first_visits": 1,
            "approx_first_pruned": 100,
            "edge_scans": 1,
            "exact_distance_saved": 1,
            "exact_distance_computed": 100,
        }
        baseline = {
            "exact_distance_computed": 999,
            "baseline_exact_distance_computed": 200,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in
                     ("micro.json", "active.json", "baseline.json", "gate.json")]
            for path, value in zip(paths[:3], (micro, active, baseline)):
                path.write_text(json.dumps(value), encoding="utf-8")
            completed = subprocess.run([
                sys.executable, str(SCRIPTS / "evaluate_break_even.py"),
                "--microbenchmark", str(paths[0]),
                "--metrics-summary", str(paths[1]),
                "--matched-baseline-metrics", str(paths[2]),
                "--mode", "approx-no-retry",
                "--output", str(paths[3]),
            ])
            self.assertEqual(completed.returncode, 0)
            result = json.loads(paths[3].read_text())
            self.assertEqual(result["inputs"]["net_exact_distances_saved"], 100)
            self.assertEqual(
                result["inputs"]["matched_baseline_exact_field"],
                "baseline_exact_distance_computed")
            self.assertEqual(result["component_cost_ns"]["retry_state_mark"], 0)

    def test_fast_source_audit(self) -> None:
        completed = subprocess.run([
            sys.executable, str(SCRIPTS / "audit_fast_path_source.py"),
            "--header", str(ROOT / "hnswlib" / "edge_quant_v0.h"),
        ])
        self.assertEqual(completed.returncode, 0)

    def test_calibration_summary(self) -> None:
        summary = {
            "approx_beta": 1.0,
            "status": "valid",
            "query_count": 3,
            "mean_baseline_recall_at_k": 0.9,
            "mean_v0_recall_at_k": 0.9,
            "mean_recall_loss": 0.0,
            "recall_loss_queries": 0,
            "catastrophic_recall_loss_queries": 0,
            "baseline_relative_exact_dco_reduction": 0.2,
            "exact_distance_saved": 10,
            "approx_first_pruned": 10,
            "approx_retry_exact_distance": 0,
            "fast_reference_decision_disagreement": 0,
            "fast_reference_near_threshold_disagreement": 0,
            "fast_reference_relative_difference_max": 1e-7,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ("approx-no-retry", "approx-retry"):
                run = root / "runs" / f"{mode}-beta1.00"
                run.mkdir(parents=True)
                (run / "summary.json").write_text(
                    json.dumps(summary), encoding="utf-8")
            output = root / "calibration.csv"
            completed = subprocess.run([
                sys.executable,
                str(SCRIPTS / "summarize_calibration.py"),
                "--run-root", str(root),
                "--expected-count", "2",
                "--output", str(output),
            ])
            self.assertEqual(completed.returncode, 0)
            self.assertEqual(len(output.read_text().splitlines()), 3)

    def test_environment_matrix_requires_and_summarizes_four_cells(self) -> None:
        script = SCRIPTS / "summarize_environment_matrix.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cells = []
            for platform_index, platform in enumerate(("testing", "weirdo")):
                for variant_index, variant in enumerate(("portable", "native")):
                    cell = root / f"{platform}-{variant}"
                    (cell / "qps").mkdir(parents=True)
                    (cell / "component").mkdir()
                    (cell / "COMPLETE").write_text("abc\n", encoding="utf-8")
                    (cell / "cell.env").write_text(
                        f"platform={platform}\nbuild_variant={variant}\n"
                        "git_commit=abc\n",
                        encoding="utf-8",
                    )
                    fields = [
                        "configuration_id", "experiment_commit",
                        "resource_profile", "qps_mean", "qps_speedup_mean",
                        "qps_speedup_ci_low", "qps_speedup_ci_high",
                        "checksum_consistent",
                    ]
                    speedup = 1.10 + platform_index * .02 + variant_index * .01
                    with (cell / "qps" / "qps_summary.csv").open(
                            "w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=fields)
                        writer.writeheader()
                        writer.writerows([
                            {"configuration_id": "baseline-ef435",
                             "experiment_commit": "abc",
                             "resource_profile": "exploratory-shared",
                             "qps_mean": 100, "qps_speedup_mean": 1,
                             "qps_speedup_ci_low": 1,
                             "qps_speedup_ci_high": 1,
                             "checksum_consistent": "True"},
                            {"configuration_id":
                             "approx-no-retry-beta1p45-legacy-ef500",
                             "experiment_commit": "abc",
                             "resource_profile": "exploratory-shared",
                             "qps_mean": 100 * speedup,
                             "qps_speedup_mean": speedup,
                             "qps_speedup_ci_low": speedup - .005,
                             "qps_speedup_ci_high": speedup + .005,
                             "checksum_consistent": "True"},
                        ])
                    (cell / "component" / "component_summary.json").write_text(
                        json.dumps({"exact_l2_ns": 200,
                                    "fast_estimator_ns": 30,
                                    "fast_lut_build_ns": 100000}),
                        encoding="utf-8")
                    (cell / "component" / "manifest.json").write_text(
                        json.dumps({"contract": {
                            "experiment_commit": "abc",
                            "resource_profile": "exploratory-shared",
                        }}),
                        encoding="utf-8",
                    )
                    cells.extend(["--cell", f"{platform}={variant}={cell}"])
            output = root / "matrix.csv"
            report = root / "matrix.json"
            completed = subprocess.run([
                sys.executable, str(script), *cells,
                "--output", str(output), "--report", str(report),
            ])
            self.assertEqual(completed.returncode, 0)
            data = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(data["status"], "complete")
            self.assertAlmostEqual(
                data["build_effect_native_minus_portable_percentage_points"]
                    ["testing"],
                1.0,
            )
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 5)


if __name__ == "__main__":
    unittest.main()
