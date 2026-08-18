#!/usr/bin/env python3
"""Regression tests for the Stage 1 margin replay."""

from __future__ import annotations

import csv
import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def write_fixture(path: Path) -> None:
    columns = [
        "query_id",
        "current_node_id",
        "candidate_id",
        "bound_status",
        "current_squared_distance",
        "threshold",
        "approximate_squared_distance",
        "error_radius",
        "lower_bound",
        "shadow_exact_squared_distance",
        "would_prune",
        "lower_bound_valid",
        "lower_bound_violation",
        "false_prune",
    ]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        candidate = 0
        for query_id in range(10):
            for index in range(20):
                candidate += 1
                exact = 1.20 if index < 15 else 0.80
                estimate = 1.60 if index < 15 else 0.90
                writer.writerow(
                    {
                        "query_id": query_id,
                        "current_node_id": query_id,
                        "candidate_id": candidate,
                        "bound_status": 0,
                        "current_squared_distance": 0.5,
                        "threshold": 1.0,
                        "approximate_squared_distance": estimate,
                        "error_radius": 0.0,
                        "lower_bound": estimate,
                        "shadow_exact_squared_distance": exact,
                        "would_prune": 0,
                        "lower_bound_valid": 1,
                        "lower_bound_violation": 0,
                        "false_prune": 0,
                    }
                )


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    script = repo / "scripts" / "v0" / "margin_retry" / "replay_margin_frontier.py"
    with tempfile.TemporaryDirectory(prefix="v0-margin-frontier-") as temporary:
        root = Path(temporary)
        input_path = root / "shadow_records.csv.gz"
        output = root / "output"
        write_fixture(input_path)
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--input",
                str(input_path),
                "--output-dir",
                str(output),
                "--betas",
                "1.0,1.25,1.5,2.0",
                "--expected-valid-records",
                "200",
                "--expected-exact-negatives",
                "150",
                "--expected-query-count",
                "10",
                "--skip-figures",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(
                "replay failed\nstdout={}\nstderr={}".format(
                    completed.stdout, completed.stderr
                )
            )
        manifest = json.loads((output / "query_split_manifest.json").read_text(encoding="utf-8"))
        assert manifest["split_counts"] == {"test": 2, "train": 6, "validation": 2}
        assert manifest["pairwise_disjoint"] is True
        candidates = json.loads((output / "candidate_operating_points.json").read_text(encoding="utf-8"))
        assert candidates["gate1_correctness_passed"] is True
        assert candidates["gate1_scientific_passed"] is True
        assert candidates["selected_betas"]
        assert candidates["test_evaluation_passes"] == 1
        test_betas = {float(row["beta"]) for row in candidates["test_rows"]}
        assert test_betas == set(candidates["selected_betas"])
        audit = json.loads((output / "input_audit.json").read_text(encoding="utf-8"))
        assert audit["valid_records"] == 200
        assert audit["exact_negatives"] == 150
        assert (output / "margin_sweep_event_summary.csv").is_file()
        assert (output / "margin_sweep_query_summary.csv").is_file()
        assert (output / "STAGE1_MARGIN_FRONTIER_REPORT.md").is_file()
    print("v0_margin_frontier_test_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
