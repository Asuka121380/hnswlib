from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.edge_estimation.export_comparison_data import (  # noqa: E402
    METHODS, export)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def metrics(correct: float) -> dict[str, float | int]:
    return {
        "correct_prune_rate": correct,
        "global_false_prune_rate": 0.001,
        "conditional_false_prune_rate": 0.002,
        "prunable_coverage": correct + 0.1,
        "prune_precision": 0.99,
        "total_prune_rate": correct + 0.001,
        "valid_estimate_coverage": 1.0,
        "p95_query_false_prune_rate": 0.002,
        "queries_with_decisions": 10,
    }


class ComparisonExportTest(unittest.TestCase):
    def test_exports_tidy_validated_tables(self) -> None:
        with tempfile.TemporaryDirectory(prefix="uq-export-") as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            for method_index, method in enumerate(METHODS):
                directory = root / f"quality-v2-{method}"
                analyses = []
                for alpha, correct in ((1.0, 0.5), (1.1, 0.4)):
                    analyses.append({
                        "alpha": alpha,
                        "counts": {
                            "decision_count_s": 100,
                            "valid_estimate_count": 100,
                            "fallback_count": 0,
                            "tp": int(correct * 100), "fp": 0,
                            "fn": 50 - int(correct * 100), "tn": 50,
                        },
                        "metrics": metrics(correct),
                    })
                write_json(directory / "quality.json", {
                    "backend": method, "analyses": analyses})
                write_json(directory / "selection.json", {
                    "selections": {
                        "balanced": {"status": "selected", "alpha": 1.0,
                                     "metrics": metrics(0.5)},
                        "strict": {"status": "selected", "alpha": 1.1,
                                   "metrics": metrics(0.4)},
                    }})
                write_json(directory / "policy.json", {
                    "gates": {
                        "balanced": {
                            "max_global_false_prune_rate": 0.005,
                            "max_p95_query_false_prune_rate": 0.01},
                        "strict": {
                            "max_global_false_prune_rate": 0.001,
                            "max_p95_query_false_prune_rate": 0.0025},
                    }})

            final_batch = 8
            timing_records = []
            timing_methods = {}
            for index, method in enumerate(METHODS):
                elapsed = 1000 + index * 100
                batch = method in {"opq", "jq"}
                timing_records.append({
                    "method": method, "block": 0, "repeat": 0,
                    "position": index, "elapsed_ns": elapsed,
                    "query_count": 10, "eligible_events": 100,
                    "execution_mode": "batch" if batch else "scalar",
                    "query_batch_size": final_batch if batch else 1,
                    "batch_rotation_engine": (
                        "blas_sgemv_sgemm" if batch else None),
                    "native_mode": "test", "checksum": index,
                    "memory_report": {
                        "backend_bytes": 1024, "query_store_bytes": 2048,
                        "scratch_bytes": 64},
                })
                timing_methods[method] = {
                    "block_median_elapsed_ns": [elapsed],
                    "median_elapsed_ns": elapsed,
                    "ns_per_query": elapsed / 10,
                    "ns_per_eligible_edge": elapsed / 100,
                    "paired_speed_ratio_vs_reference": 1000 / elapsed,
                }
            timing_dir = root / f"timing-final-horizontal-b{final_batch}"
            write_json(timing_dir / "result.json", {
                "mode": "randomized_paired_selected_query_preparation",
                "evidence": {"formal_admitted": True},
                "raw_records": timing_records,
                "summary": {"reference": "pq8", "methods": timing_methods},
            })
            write_json(timing_dir / "complete.json", {"stage": "test"})

            for batch_size in (1, 8):
                records = []
                summaries = {}
                for index, method in enumerate(("opq", "jq")):
                    elapsed = 1000 // batch_size + index
                    records.append({
                        "method": method,
                        "batch_rotation_engine": "blas_sgemv_sgemm",
                        "memory_report": {"backend_bytes": 1024},
                    })
                    summaries[method] = {
                        "median_elapsed_ns": elapsed,
                        "ns_per_query": elapsed / 10,
                        "ns_per_eligible_edge": elapsed / 100,
                        "paired_speed_ratio_vs_reference": 1.0,
                    }
                    parity = {
                        "valid": True, "backend": method,
                        "batch_rotation_engine": "blas_sgemv_sgemm",
                        "compared_count": 100,
                        "nonfinite_mismatch_count": 0,
                        "tolerance_failure_count": 0,
                        "max_absolute_error": 1e-6,
                        "max_relative_error": 1e-6,
                        "absolute_tolerance": 1e-4,
                        "relative_tolerance": 1e-5,
                    }
                    write_json(
                        root / f"batch-parity-b{batch_size}-{method}.json",
                        parity)
                    if batch_size == final_batch:
                        write_json(
                            root /
                            f"final-horizontal-b{final_batch}-parity-{method}.json",
                            parity)
                directory = root / f"timing-batch-b{batch_size}"
                write_json(directory / "result.json", {
                    "query_batch_size": batch_size,
                    "raw_records": records,
                    "summary": {"methods": summaries},
                })
                write_json(directory / "complete.json", {"stage": "test"})

            output = Path(temporary) / "export"
            export(root, output, batch_sizes=(1, 8),
                   final_batch_size=final_batch)
            self.assertTrue((output / "complete.json").is_file())
            with (output / "quality_sweep.csv").open(
                    encoding="utf-8", newline="") as stream:
                quality_rows = list(csv.DictReader(stream))
            self.assertEqual(len(quality_rows), len(METHODS) * 2)
            with (output / "operating_points.csv").open(
                    encoding="utf-8", newline="") as stream:
                operating_rows = list(csv.DictReader(stream))
            self.assertEqual(len(operating_rows), len(METHODS) * 2)
            with (output / "timing_summary.csv").open(
                    encoding="utf-8", newline="") as stream:
                timing_rows = list(csv.DictReader(stream))
            self.assertEqual([row["method"] for row in timing_rows],
                             list(METHODS))
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["audit"]["parity_all_passed"])
            self.assertEqual(manifest["audit"]["timing_record_count"], 6)


if __name__ == "__main__":
    unittest.main()
