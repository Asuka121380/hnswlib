#!/usr/bin/env python3
"""Join full-range V0 correctness metrics with block-level QPS results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


FIELDS = (
    "beta", "mode", "prefetch", "query_count", "baseline_recall",
    "v0_recall", "recall_loss", "recall_loss_queries",
    "catastrophic_queries", "dco_reduction", "exact_distance_saved",
    "first_pruned", "retry_exact", "decision_disagreement",
    "near_threshold_disagreement", "relative_difference_max",
    "qps_mean", "qps_ci_low", "qps_ci_high", "qps_speedup_mean",
    "qps_speedup_ci_low", "qps_speedup_ci_high",
    "latency_p50_ns_median", "latency_p95_ns_median",
    "latency_p99_ns_median", "process_replicates",
    "checksum_consistent",
)

T_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def confidence_interval(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean, mean
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    critical = T_975.get(len(values) - 1, 1.96)
    return mean, mean - critical * standard_error, mean + critical * standard_error


def beta_key(value: object) -> str:
    return f"{float(value):.6f}"


def load_metrics(path: Path) -> tuple[dict[tuple[str, str], dict[str, str]], str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("metrics summary is empty")
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row["status"] != "valid":
            raise ValueError("metrics summary contains an invalid row")
        key = (beta_key(row["beta"]), row["mode"])
        if key in result:
            raise ValueError(f"duplicate metrics row: {key}")
        result[key] = row
    return result, rows[0]["baseline_recall"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-summary", required=True, type=Path)
    parser.add_argument("--qps-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    metrics, baseline_recall = load_metrics(args.metrics_summary)
    manifest_path = args.qps_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("QPS manifest is not complete")
    contract = manifest["contract"]
    blocks = int(contract["blocks"])
    completed = manifest["completed"]
    expected = blocks * (
        1 + len(contract["betas"]) *
        len(contract["modes"]) * len(contract["prefetches"]))
    if len(completed) != expected:
        raise ValueError(
            f"expected {expected} QPS results, found {len(completed)}")

    grouped: dict[str, list[tuple[dict[str, object], dict[str, object]]]] = {}
    baseline_by_block: dict[int, float] = {}
    for item in completed:
        result_path = args.qps_dir / str(item["result"])
        data = json.loads(result_path.read_text(encoding="utf-8"))
        config = item["config"]
        config_id = str(config["id"])
        grouped.setdefault(config_id, []).append((item, data))
        if config["method"] == "baseline":
            block = int(item["block"])
            if block in baseline_by_block:
                raise ValueError(f"duplicate baseline in block {block}")
            baseline_by_block[block] = float(data["qps"])
    if len(baseline_by_block) != blocks:
        raise ValueError("one or more blocks lack a baseline result")

    rows: list[dict[str, object]] = []
    faster_with_ci = 0
    for config_id, observations in grouped.items():
        if len(observations) != blocks:
            raise ValueError(
                f"configuration {config_id} has {len(observations)} results")
        config = observations[0][0]["config"]
        qps_values = [float(data["qps"]) for _, data in observations]
        qps_mean, qps_low, qps_high = confidence_interval(qps_values)
        if config["method"] == "baseline":
            speedups = [1.0] * blocks
            metric: dict[str, object] = {
                "query_count": contract["query_count"],
                "baseline_recall": baseline_recall,
                "v0_recall": baseline_recall,
                "recall_loss": 0,
                "recall_loss_queries": 0,
                "catastrophic_queries": 0,
                "dco_reduction": 0,
                "exact_distance_saved": 0,
                "first_pruned": 0,
                "retry_exact": 0,
                "decision_disagreement": 0,
                "near_threshold_disagreement": 0,
                "relative_difference_max": 0,
            }
        else:
            speedups = [
                float(data["qps"]) / baseline_by_block[int(item["block"])]
                for item, data in observations
            ]
            key = (beta_key(config["beta"]), str(config["method"]))
            if key not in metrics:
                raise ValueError(f"missing metrics row for {key}")
            metric = metrics[key]
        speed_mean, speed_low, speed_high = confidence_interval(speedups)
        if config["method"] != "baseline" and speed_low > 1.0:
            faster_with_ci += 1
        checksums = {str(data["result_checksum"]) for _, data in observations}
        rows.append({
            "beta": "" if config["beta"] is None else config["beta"],
            "mode": config["method"],
            "prefetch": config["prefetch"],
            **{name: metric[name] for name in FIELDS[3:16]},
            "qps_mean": qps_mean,
            "qps_ci_low": qps_low,
            "qps_ci_high": qps_high,
            "qps_speedup_mean": speed_mean,
            "qps_speedup_ci_low": speed_low,
            "qps_speedup_ci_high": speed_high,
            "latency_p50_ns_median": statistics.median(
                float(data["latency_p50_ns"]) for _, data in observations),
            "latency_p95_ns_median": statistics.median(
                float(data["latency_p95_ns"]) for _, data in observations),
            "latency_p99_ns_median": statistics.median(
                float(data["latency_p99_ns"]) for _, data in observations),
            "process_replicates": len(observations),
            "checksum_consistent": len(checksums) == 1,
        })

    rows.sort(key=lambda row: (
        -1.0 if row["beta"] == "" else float(row["beta"]),
        str(row["mode"]), str(row["prefetch"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "schema_version": 1,
        "status": "complete",
        "experiment_commit": contract["experiment_commit"],
        "beta_count": len(contract["betas"]),
        "qps_configuration_count": len(rows),
        "process_result_count": len(completed),
        "blocks": blocks,
        "active_configurations_with_speedup_ci_low_above_one": faster_with_ci,
        "metrics_summary": str(args.metrics_summary.resolve()),
        "qps_manifest": str(manifest_path.resolve()),
        "combined_csv": str(args.output.resolve()),
    }
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(
        f"full_beta_tradeoff_complete rows={len(rows)} "
        f"faster_with_ci={faster_with_ci} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
