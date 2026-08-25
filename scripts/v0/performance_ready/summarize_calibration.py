#!/usr/bin/env python3
"""Validate and summarize raw_fast_v1 calibration runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "beta", "mode", "status", "query_count", "baseline_recall",
    "v0_recall", "recall_loss", "recall_loss_queries",
    "catastrophic_queries", "dco_reduction", "exact_distance_saved",
    "first_pruned", "retry_exact", "decision_disagreement",
    "near_threshold_disagreement", "relative_difference_max",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for path in sorted((args.run_root / "runs").glob("*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        label = path.parent.name
        if "-beta" not in label:
            raise ValueError(f"unexpected run label: {label}")
        rows.append({
            "beta": data["approx_beta"],
            "mode": label.rsplit("-beta", 1)[0],
            "status": data["status"],
            "query_count": data["query_count"],
            "baseline_recall": data["mean_baseline_recall_at_k"],
            "v0_recall": data["mean_v0_recall_at_k"],
            "recall_loss": data["mean_recall_loss"],
            "recall_loss_queries": data["recall_loss_queries"],
            "catastrophic_queries":
                data["catastrophic_recall_loss_queries"],
            "dco_reduction":
                data["baseline_relative_exact_dco_reduction"],
            "exact_distance_saved": data["exact_distance_saved"],
            "first_pruned": data["approx_first_pruned"],
            "retry_exact": data["approx_retry_exact_distance"],
            "decision_disagreement":
                data["fast_reference_decision_disagreement"],
            "near_threshold_disagreement":
                data["fast_reference_near_threshold_disagreement"],
            "relative_difference_max":
                data["fast_reference_relative_difference_max"],
        })

    rows.sort(key=lambda row: (float(row["beta"]), str(row["mode"])))
    if len(rows) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} summaries, found {len(rows)}")
    if any(row["status"] != "valid" for row in rows):
        raise ValueError("one or more calibration runs are invalid")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"beta={float(row['beta']):.3f} "
            f"mode={str(row['mode']):15s} "
            f"recall={float(row['v0_recall']):.6f} "
            f"loss={float(row['recall_loss']):+.6f} "
            f"dco={float(row['dco_reduction']):.3%} "
            f"cat={row['catastrophic_queries']} "
            f"disagree={row['decision_disagreement']}")
    print(f"calibration_complete rows={len(rows)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
