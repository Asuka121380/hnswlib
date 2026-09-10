#!/usr/bin/env python3
"""Join a frozen recall selection with its paired QPS summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "configuration_id", "baseline_id", "beta", "mode", "prefetch",
    "edgepq_ef_search", "baseline_ef_search", "edgepq_recall",
    "baseline_recall", "recall_gap", "qps_mean", "qps_ci_low",
    "qps_ci_high", "qps_speedup_mean", "qps_speedup_ci_low",
    "qps_speedup_ci_high", "latency_p50_ns_median",
    "latency_p95_ns_median", "latency_p99_ns_median",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--qps-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    with args.qps_summary.open(encoding="utf-8", newline="") as handle:
        qps = {row["configuration_id"]: row for row in csv.DictReader(handle)}
    rows = []
    for item in selection["selections"]:
        configuration_id = item["configuration_id"]
        if configuration_id not in qps:
            raise SystemExit(f"missing QPS row: {configuration_id}")
        timing = qps[configuration_id]
        if timing["baseline_id"] != item["baseline_id"]:
            raise SystemExit(f"baseline identity mismatch: {configuration_id}")
        row = {field: item.get(field, timing.get(field, "")) for field in FIELDS}
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"matched_recall_summary_complete rows={len(rows)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
