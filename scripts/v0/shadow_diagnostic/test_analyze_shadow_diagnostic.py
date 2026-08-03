#!/usr/bin/env python3
"""Synthetic regression test for analyze_shadow_diagnostic.py."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    script = Path(__file__).with_name("analyze_shadow_diagnostic.py")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run = root / "run"
        out = root / "analysis"
        run.mkdir()
        rows = [
            # oracle/raw/safe all true
            [0, 1, 2, 0, 1.0, 10.0, 14.0, 2.0, 12.0, 13.0, 1, 1, 0, 0],
            # oracle/raw true, safe false: radius loss
            [0, 1, 3, 0, 1.0, 10.0, 12.0, 4.0, 8.0, 11.0, 0, 1, 0, 0],
            # oracle true, raw/safe false: estimator loss
            [1, 4, 5, 0, 1.0, 10.0, 9.0, 2.0, 7.0, 11.0, 0, 1, 0, 0],
            # no pruning opportunity
            [1, 4, 6, 0, 1.0, 10.0, 9.0, 2.0, 7.0, 9.5, 0, 1, 0, 0],
        ]
        header = [
            "query_id", "current_node_id", "candidate_id", "bound_status",
            "current_squared_distance", "threshold", "approximate_squared_distance",
            "error_radius", "lower_bound", "shadow_exact_squared_distance",
            "would_prune", "lower_bound_valid", "lower_bound_violation", "false_prune",
        ]
        with (run / "shadow_records.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        with (run / "query_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "query_id", "baseline_recall_at_k", "v0_recall_at_k", "results_equal",
                "baseline_latency_ns", "v0_latency_ns", "bound_evaluated", "bound_pruned",
                "exact_fallback", "exact_only_fallback", "exact_distance_saved",
                "lower_bound_violation", "false_prune",
            ])
            writer.writerow([0, 1.0, 1.0, 1, 1, 2, 2, 1, 0, 0, 0, 0, 0])
            writer.writerow([1, 1.0, 1.0, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0])
        write_json(run / "summary.json", {
            "status": "valid", "query_count": 2, "mismatch_queries": 0,
            "bound_evaluated": 4, "exact_distance_saved": 0,
            "lower_bound_violation": 0, "false_prune": 0,
            "shadow_records_seen": 4, "shadow_records_written": 4,
        })
        write_json(run / "metadata.json", {
            "mode": "shadow", "run_id": "synthetic", "ef_search": 200,
            "shadow_sample_modulus": 1, "shadow_validation_compiled": True,
            "real_pruning_compiled": False,
        })
        subprocess.run(
            [sys.executable, str(script), "--run-dir", str(run), "--output-dir", str(out),
             "--expected-query-count", "2"],
            check=True,
        )
        quality = json.loads((out / "data_quality.json").read_text(encoding="utf-8"))
        assert quality["gate_passed"] is True
        assert quality["coverage"]["oracle"] == 0.75
        assert quality["coverage"]["raw"] == 0.5
        assert quality["coverage"]["safe"] == 0.25
        for name in (
            "diagnostic_summary.csv", "candidate_metrics.parquet",
            "alpha_sensitivity.csv", "analysis.md",
        ):
            assert (out / name).is_file(), name
        assert len(list((out / "figures").glob("*.png"))) == 5
        assert len(list((out / "figures").glob("*.pdf"))) == 5
    print("synthetic diagnostic regression test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
