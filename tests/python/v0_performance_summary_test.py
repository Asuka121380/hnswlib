#!/usr/bin/env python3
"""Tests for the strict V0 performance-matrix summarizer."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "v0" / "summarize_v0_performance.py"


class V0PerformanceSummaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.matrix = self.root / "matrix"
        (self.matrix / "results").mkdir(parents=True)
        (self.matrix / "matrix_complete.json").write_text(
            json.dumps({"status": "complete", "dataset": "gist1m", "repeats": 2}),
            encoding="utf-8",
        )
        for ef_search in (50, 100):
            self._write_run(ef_search, "warmup", 1, 0)
            self._write_run(ef_search, "repeat1", 3, 0)
            self._write_run(ef_search, "repeat2", 3, 10)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_run(
        self,
        ef_search: int,
        label: str,
        query_count: int,
        latency_offset: int,
    ) -> None:
        run_dir = self.matrix / "results" / f"ef{ef_search}" / label
        run_dir.mkdir(parents=True)
        metadata = {
            "format": "hnswlib_v0_search_run",
            "format_version": 1,
            "dataset": "gist1m",
            "mode": "prune",
            "run_id": f"fixture-ef{ef_search}-{label}",
            "dataset_config": "/fixture/dataset.json",
            "index_path": "/fixture/index.bin",
            "sidecar_path": "/fixture/sidecar.v0meta",
            "dimension": 960,
            "n_base": 1_000_000,
            "query_start": 0,
            "query_count": query_count,
            "k": 10,
            "ef_search": ef_search,
            "M_pq": 32,
            "nbits": 8,
            "code_bytes_per_edge": 32,
            "directed_edge_count": 12_611_816,
            "index_sha256": "a" * 64,
            "sidecar_sha256": "b" * 64,
            "codebook_sha256": "c" * 64,
            "sidecar_bytes": 917_036_050,
            "real_pruning_enabled": True,
            "bound_pruned_semantics": "actual_prune",
            "shadow_sample_modulus": 1024,
            "shadow_sample_remainder": 0,
            "producer_git_commit": "fixture-commit",
            "git_branch": "v0-edge-quantisation",
            "working_tree_dirty": "false",
            "shadow_validation_compiled": False,
            "real_pruning_compiled": True,
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (run_dir / "complete.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "mode": "prune",
                    "query_start": 0,
                    "query_count": query_count,
                }
            ),
            encoding="utf-8",
        )

        rows = []
        for query_id in range(query_count):
            rows.append(
                {
                    "query_id": query_id,
                    "baseline_recall_at_k": 0.9,
                    "v0_recall_at_k": 0.9,
                    "results_equal": 1,
                    "baseline_latency_ns": 100 + latency_offset + query_id * 100,
                    "v0_latency_ns": 200 + latency_offset + query_id * 200,
                    "bound_evaluated": 10,
                    "bound_pruned": 1,
                    "exact_fallback": 8,
                    "exact_only_fallback": 1,
                    "exact_distance_saved": 1,
                    "lower_bound_violation": 0,
                    "false_prune": 0,
                }
            )
        with (run_dir / "query_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "format": "hnswlib_v0_search_summary",
            "format_version": 1,
            "status": "valid",
            "query_count": query_count,
            "mismatch_queries": 0,
            "mean_baseline_recall_at_k": 0.9,
            "mean_v0_recall_at_k": 0.9,
            "baseline_latency_ns": sum(row["baseline_latency_ns"] for row in rows),
            "v0_latency_ns": sum(row["v0_latency_ns"] for row in rows),
            "bound_evaluated": sum(row["bound_evaluated"] for row in rows),
            "bound_pruned": sum(row["bound_pruned"] for row in rows),
            "exact_fallback": sum(row["exact_fallback"] for row in rows),
            "exact_only_fallback": sum(row["exact_only_fallback"] for row in rows),
            "exact_distance_saved": sum(row["exact_distance_saved"] for row in rows),
            "lower_bound_violation": 0,
            "false_prune": 0,
            "shadow_records_seen": 0,
            "shadow_records_written": 0,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    def _run(self, output: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--matrix-root",
                str(self.matrix),
                "--output-dir",
                str(output),
                "--expected-ef-search",
                "50,100",
                "--expected-repeats",
                "2",
                "--expected-queries",
                "3",
                "--expected-warmup-queries",
                "1",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )

    def test_generates_strict_csv_json_and_markdown_outputs(self) -> None:
        output = self.root / "summary"
        completed = self._run(output)
        self.assertEqual(0, completed.returncode, completed.stderr)
        aggregate = json.loads(
            (output / "performance_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual("valid", aggregate["status"])
        self.assertEqual([50, 100], aggregate["protocol"]["ef_search_values"])
        self.assertEqual(2, len(aggregate["configurations"]))
        first = aggregate["configurations"][0]
        self.assertEqual(6, first["measured_queries"])
        self.assertEqual(60, first["bound_evaluated"])
        self.assertEqual(6, first["bound_pruned"])
        self.assertEqual(0.1, first["prune_rate"])
        self.assertEqual(1.0, first["results_equal_rate"])
        self.assertTrue((output / "performance_summary.csv").is_file())
        report = (output / "report.md").read_text(encoding="utf-8")
        self.assertIn("V0 real-pruning performance report", report)
        self.assertIn("| 50 |", report)

    def test_rejects_a_counter_that_does_not_close(self) -> None:
        path = self.matrix / "results" / "ef50" / "repeat1" / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["bound_evaluated"] += 1
        path.write_text(json.dumps(summary), encoding="utf-8")
        completed = self._run(self.root / "invalid-summary")
        self.assertNotEqual(0, completed.returncode)
        self.assertIn("bound_evaluated does not close", completed.stderr)


if __name__ == "__main__":
    unittest.main()
