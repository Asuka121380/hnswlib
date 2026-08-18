#!/usr/bin/env python3
"""Regression test for Stage 2 retry-shadow aggregation."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    script = repo / "scripts" / "v0" / "margin_retry" / "analyze_retry_shadow.py"
    with tempfile.TemporaryDirectory(prefix="v0-retry-shadow-") as temporary:
        root = Path(temporary)
        runner = root / "runner"
        output = root / "output"
        runner.mkdir()
        (runner / "summary.json").write_text(
            json.dumps({"status": "valid", "query_count": 2, "mismatch_queries": 0, "exact_distance_saved": 0}),
            encoding="utf-8",
        )
        (runner / "metadata.json").write_text(
            json.dumps({"mode": "retry-shadow", "retry_betas": [1.45]}),
            encoding="utf-8",
        )
        stage1 = root / "candidates.json"
        stage1.write_text(json.dumps({"selected_betas": [1.45]}), encoding="utf-8")
        fields = [
            "schema_version", "query_id", "beta", "eligible_first_visits",
            "first_pruned", "first_false_pruned", "pruned_revisited",
            "false_pruned_revisited", "unrevisited_first_pruned",
            "unrevisited_false_pruned", "duplicate_encounters_after_prune",
            "expanded_nodes",
        ]
        with (runner / "retry_shadow_per_query_raw.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(dict(zip(fields, [1, 0, 1.45, 100, 40, 2, 30, 2, 10, 0, 45, 20])))
            writer.writerow(dict(zip(fields, [1, 1, 1.45, 100, 40, 2, 25, 1, 15, 1, 38, 18])))
        record_fields = [
            "schema_version", "query_id", "beta", "candidate_id",
            "first_parent_id", "second_parent_id", "first_expansion_index",
            "second_expansion_index", "revisit_delay_expansions",
            "duplicate_encounters", "first_parent_degree", "candidate_degree",
            "first_threshold", "first_raw_estimate", "first_exact_distance",
            "first_false_prune", "revisited",
        ]
        with (runner / "retry_shadow_records.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=record_fields)
            writer.writeheader()
            writer.writerow(dict(zip(record_fields, [1, 0, 1.45, 9, 1, 2, 3, 5, 2, 1, 8, 7, 1.0, 1.5, 0.9, 1, 1])))
        completed = subprocess.run(
            [sys.executable, str(script), "--runner-output-dir", str(runner), "--stage1-candidates", str(stage1), "--output-dir", str(output), "--skip-figures"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError("analysis failed\n{}\n{}".format(completed.stdout, completed.stderr))
        result = json.loads((output / "candidate_operating_points_with_retry.json").read_text(encoding="utf-8"))
        assert result["correctness_gate_passed"] is True
        assert result["gate2_classification"] == "GO"
        assert (output / "retry_shadow_records.csv.gz").is_file()
        assert (output / "retry_shadow_structural_summary.csv").is_file()
        assert (output / "STAGE2_RETRY_UPPER_BOUND_REPORT.md").is_file()
    print("v0_retry_shadow_analysis_test_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
