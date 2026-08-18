#!/usr/bin/env python3
"""Regression test for unified Stage 3–5 active aggregation."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


FIELDS = [
    "query_id",
    "baseline_recall_at_k",
    "v0_recall_at_k",
    "results_equal",
    "ground_truth_hits_lost",
    "result_overlap_at_k",
    "recall_loss",
    "baseline_exact_distance_computed",
    "exact_distance_computed",
    "approx_eligible_first_visits",
    "approx_first_pruned",
    "approx_retry_exact_distance",
]


def write_run(
    root: Path,
    mode: str,
    active_recall: float,
    active_dco: int,
    retry: int,
    beta: float = 1.4,
) -> Path:
    run = root / mode
    run.mkdir()
    (run / "metadata.json").write_text(
        json.dumps({"mode": mode, "approx_beta": beta, "ef_search": 200}),
        encoding="utf-8",
    )
    first_pruned = 40
    summary = {
        "status": "valid",
        "query_count": 2,
        "mean_baseline_recall_at_k": 0.95,
        "mean_v0_recall_at_k": active_recall,
        "mean_recall_loss": 0.95 - active_recall,
        "mismatch_queries": 1 if active_recall != 0.95 else 0,
        "recall_loss_queries": 1 if active_recall < 0.95 else 0,
        "catastrophic_recall_loss_queries": 0,
        "baseline_exact_distance_computed": 200,
        "exact_distance_computed": active_dco,
        "approx_eligible_first_visits": 100,
        "approx_first_pruned": first_pruned,
        "approx_retry_exact_distance": retry,
        "approx_retry_inserted_candidate": retry // 2,
        "approx_retry_inserted_result": retry // 3,
        "exact_distance_saved": first_pruned - retry,
    }
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "complete.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    with (run / "query_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for query_id in range(2):
            writer.writerow(
                {
                    "query_id": query_id,
                    "baseline_recall_at_k": 0.95,
                    "v0_recall_at_k": active_recall,
                    "results_equal": int(active_recall == 0.95),
                    "ground_truth_hits_lost": 0,
                    "result_overlap_at_k": 10,
                    "recall_loss": 0.95 - active_recall,
                    "baseline_exact_distance_computed": 100,
                    "exact_distance_computed": active_dco // 2,
                    "approx_eligible_first_visits": 50,
                    "approx_first_pruned": 20,
                    "approx_retry_exact_distance": retry // 2,
                }
            )
    return run


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    script = repo / "scripts" / "v0" / "margin_retry" / "analyze_active_experiment.py"
    with tempfile.TemporaryDirectory(prefix="v0-active-analysis-") as temporary:
        root = Path(temporary)
        no_retry = write_run(root, "approx-no-retry", 0.80, 120, 0)
        retry = write_run(root, "approx-retry", 0.95, 160, 20)
        output = root / "analysis"
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--run-dir",
                str(no_retry),
                "--run-dir",
                str(retry),
                "--output-dir",
                str(output),
                "--minimum-query-count",
                "2",
                "--expected-betas",
                "1.40",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(f"analysis failed\n{completed.stdout}\n{completed.stderr}")
        decision = json.loads(
            (output / "active_experiment_decision.json").read_text(encoding="utf-8")
        )
        assert decision["gate5_classification"] == "GO"
        assert decision["betas"] == [1.4]
        report = (output / "STAGE3_5_ACTIVE_RECALL_DCO_REPORT.md").read_text(
            encoding="utf-8"
        )
        assert "stage1_fail_active_included" in report
        assert (output / "active_per_query.csv").is_file()

        safe_root = root / "recovery-not-needed"
        safe_root.mkdir()
        safe_no_retry = write_run(
            safe_root, "approx-no-retry", 0.9487, 120, 0, beta=1.55
        )
        safe_retry = write_run(
            safe_root, "approx-retry", 0.9488, 140, 20, beta=1.55
        )
        safe_output = safe_root / "analysis"
        safe_completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--run-dir",
                str(safe_no_retry),
                "--run-dir",
                str(safe_retry),
                "--output-dir",
                str(safe_output),
                "--minimum-query-count",
                "2",
                "--expected-betas",
                "1.55",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if safe_completed.returncode != 0:
            raise AssertionError(
                "recovery-not-needed analysis failed\n"
                f"{safe_completed.stdout}\n{safe_completed.stderr}"
            )
        safe_decision = json.loads(
            (safe_output / "active_experiment_decision.json").read_text(
                encoding="utf-8"
            )
        )
        assert safe_decision["gate5_classification"] == "GO"
        safe_row = safe_decision["formal_retry_rows"][0]
        assert safe_row["recall_recovery"] < 0.50
        assert safe_row["recall_recovery_not_needed"] is True
        assert safe_row["numeric_gate_pass"] is True
        safe_report = (
            safe_output / "STAGE3_5_ACTIVE_RECALL_DCO_REPORT.md"
        ).read_text(encoding="utf-8")
        assert "not needed" in safe_report
    print("v0_active_experiment_analysis_test_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
