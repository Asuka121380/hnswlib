#!/usr/bin/env python3
"""Summarize a configurable V0 QPS experiment without fixed beta assumptions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


T_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}
FIELDS = (
    "experiment_name", "experiment_role", "resource_profile", "exclusive",
    "experiment_commit", "contract_sha256", "configuration_id",
    "baseline_id", "ef_search", "beta", "mode", "prefetch",
    "query_start", "query_count", "blocks",
    "within_process_repeats", "qps_mean", "qps_ci_low", "qps_ci_high",
    "qps_speedup_mean", "qps_speedup_ci_low", "qps_speedup_ci_high",
    "latency_p50_ns_median", "latency_p95_ns_median",
    "latency_p99_ns_median", "process_replicates", "checksum_consistent",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def confidence_interval(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean, mean
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    critical = T_975.get(len(values) - 1, 1.96)
    return mean, mean - critical * standard_error, mean + critical * standard_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or args.run_root / "qps_summary.csv"
    report_path = args.report or args.run_root / "qps_summary.json"
    manifest_path = args.run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise SystemExit("QPS manifest is not complete")
    contract = manifest["contract"]
    resolved = contract["resolved_config"]
    completed = manifest.get("completed", [])
    expected = int(resolved["expected_result_count"])
    if len(completed) != expected:
        raise SystemExit(
            f"expected {expected} QPS results, found {len(completed)}")

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    baseline_by_id_block: dict[tuple[str, int], float] = {}
    seen_run_ids: set[str] = set()
    for item in completed:
        run_id = str(item["run_id"])
        if run_id in seen_run_ids:
            raise SystemExit(f"duplicate run ID: {run_id}")
        seen_run_ids.add(run_id)
        result_path = args.run_root / str(item["result"])
        if not result_path.is_file():
            raise SystemExit(f"missing QPS result: {result_path}")
        if sha256(result_path) != item.get("sha256"):
            raise SystemExit(f"QPS result checksum mismatch: {result_path}")
        data = json.loads(result_path.read_text(encoding="utf-8"))
        config = item["config"]
        config_id = str(config["id"])
        grouped.setdefault(config_id, []).append((item, data))
        if config["method"] == "baseline":
            block = int(item["block"])
            key = (config_id, block)
            if key in baseline_by_id_block:
                raise SystemExit(
                    f"duplicate baseline {config_id} in block {block}")
            baseline_by_id_block[key] = float(data["qps"])

    blocks = int(resolved["blocks"])
    baseline_ids = {
        str(observations[0][0]["config"]["id"])
        for observations in grouped.values()
        if observations[0][0]["config"]["method"] == "baseline"
    }
    for baseline_id in baseline_ids:
        if any((baseline_id, block) not in baseline_by_id_block
               for block in range(blocks)):
            raise SystemExit(
                f"one or more blocks lack baseline {baseline_id}")
    expected_configurations = int(resolved["configuration_count"])
    if len(grouped) != expected_configurations:
        raise SystemExit(
            f"expected {expected_configurations} configurations, "
            f"found {len(grouped)}")

    rows: list[dict[str, Any]] = []
    for config_id, observations in grouped.items():
        if len(observations) != blocks:
            raise SystemExit(
                f"configuration {config_id} has {len(observations)} results")
        config = observations[0][0]["config"]
        if any(item["config"] != config for item, _ in observations):
            raise SystemExit(f"configuration identity changed: {config_id}")
        qps_values = [float(data["qps"]) for _, data in observations]
        qps_mean, qps_low, qps_high = confidence_interval(qps_values)
        if config["method"] == "baseline":
            speed_mean, speed_low, speed_high = 1.0, 1.0, 1.0
        elif config.get("baseline_id"):
            baseline_id = str(config["baseline_id"])
            speedups = [
                float(data["qps"]) /
                baseline_by_id_block[(baseline_id, int(item["block"]))]
                for item, data in observations
            ]
            speed_mean, speed_low, speed_high = confidence_interval(speedups)
        else:
            speed_mean = speed_low = speed_high = ""
        if any("result_checksum" not in data for _, data in observations):
            raise SystemExit(f"result checksum is missing: {config_id}")
        checksums = {
            str(data["result_checksum"]) for _, data in observations}
        rows.append({
            "experiment_name": resolved["experiment_name"],
            "experiment_role": resolved["experiment_role"],
            "resource_profile": resolved["resource_profile"],
            "exclusive": resolved["exclusive"],
            "experiment_commit": contract["experiment_commit"],
            "contract_sha256": resolved["contract_sha256"],
            "configuration_id": config["id"],
            "baseline_id": config.get("baseline_id") or "",
            "ef_search": config["ef_search"],
            "beta": "" if config["beta"] is None else config["beta"],
            "mode": config["method"],
            "prefetch": config["prefetch"],
            "query_start": resolved["query_start"],
            "query_count": resolved["query_count"],
            "blocks": blocks,
            "within_process_repeats": resolved["within_process_repeats"],
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
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "schema_version": 1,
        "status": "complete",
        "experiment_name": resolved["experiment_name"],
        "experiment_role": resolved["experiment_role"],
        "resource_profile": resolved["resource_profile"],
        "exclusive": resolved["exclusive"],
        "experiment_commit": contract["experiment_commit"],
        "contract_sha256": resolved["contract_sha256"],
        "configuration_count": len(rows),
        "process_result_count": len(completed),
        "blocks": blocks,
        "manifest": str(manifest_path.resolve()),
        "summary_csv": str(output.resolve()),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"qps_summary_complete rows={len(rows)} "
        f"profile={resolved['resource_profile']} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
