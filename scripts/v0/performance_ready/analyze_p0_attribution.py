#!/usr/bin/env python3
"""Join P0 ordered latency, result parity, and per-query work ledgers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ACTIVE_METRICS = (
    "exact_distance_computed", "expanded_nodes", "edge_scans",
    "duplicate_encounters", "candidate_queue_pushes",
    "candidate_queue_pops", "result_queue_pushes", "result_queue_pops",
    "threshold_updates", "approx_eligible_first_visits",
    "approx_first_pruned", "approx_retry_encountered",
    "approx_retry_exact_distance", "approx_retry_inserted_candidate",
    "approx_retry_inserted_result", "approx_estimator_fallback",
)

BASELINE_METRICS = (
    "exact_distance_computed", "expanded_nodes", "edge_scans",
    "duplicate_encounters", "candidate_queue_pushes",
    "candidate_queue_pops", "result_queue_pushes", "result_queue_pops",
    "threshold_updates",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assignments(values: Iterable[str], name: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{name} must use CONFIG=PATH: {value}")
        config, raw_path = value.split("=", 1)
        if not config or config in result:
            raise SystemExit(f"duplicate or empty {name} config: {config!r}")
        result[config] = Path(raw_path)
    return result


def pair_assignments(values: Iterable[str]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise SystemExit(f"pair must use CANDIDATE=BASELINE: {value}")
        candidate, baseline = value.split("=", 1)
        if not candidate or not baseline:
            raise SystemExit(f"invalid pair: {value}")
        result.append((candidate, baseline))
    return result


def artifact(item: dict[str, Any], prefix: str, root: Path) -> Path | None:
    matches = [
        entry for entry in item.get("artifacts", [])
        if Path(str(entry["path"])).parts[0] == prefix
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise SystemExit(f"multiple {prefix} artifacts for {item['run_id']}")
    path = root / str(matches[0]["path"])
    if not path.is_file() or sha256(path) != matches[0].get("sha256"):
        raise SystemExit(f"missing or corrupt artifact: {path}")
    return path


def read_latency(
    manifest: dict[str, Any], root: Path,
) -> tuple[dict[str, dict[int, list[float]]], list[dict[str, Any]]]:
    by_config: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    raw: list[dict[str, Any]] = []
    for item in manifest.get("completed", []):
        path = artifact(item, "latency_records", root)
        if path is None:
            continue
        config = str(item["config"]["id"])
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                query_id = int(row["query_id"])
                latency = float(row["latency_ns"])
                by_config[config][query_id].append(latency)
                raw.append({
                    "configuration_id": config,
                    "block_id": int(item["block"]),
                    "repeat_id": int(row["repeat_id"]),
                    "query_id": query_id,
                    "latency_ns": int(latency),
                })
    return by_config, raw


def numeric(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def read_metrics(
    paths: dict[str, Path], *, baseline: bool,
) -> dict[str, dict[int, dict[str, float]]]:
    result: dict[str, dict[int, dict[str, float]]] = {}
    fields = BASELINE_METRICS if baseline else ACTIVE_METRICS
    for config, path in paths.items():
        rows: dict[int, dict[str, float]] = {}
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            available = set(reader.fieldnames or [])
            prefix = "baseline_" if baseline else ""
            required = {prefix + field for field in fields}
            missing = required - available
            if missing:
                raise SystemExit(
                    f"metrics {path} lacks fields: {', '.join(sorted(missing))}")
            for row in reader:
                query_id = int(row["query_id"])
                converted: dict[str, float] = {}
                for field in fields:
                    value = numeric(row.get(prefix + field))
                    if value is not None:
                        converted[field] = value
                rows[query_id] = converted
        result[config] = rows
    return result


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    mean_x, mean_y = statistics.fmean(x), statistics.fmean(y)
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    denominator = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if denominator == 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denominator


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        for position in order[start:end]:
            result[position] = rank
        start = end
    return result


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    return ordered[int(fraction * (len(ordered) - 1))]


def confidence_interval(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean, mean
    critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}.get(
        len(values) - 1, 1.96)
    half_width = critical * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def read_qps(manifest: dict[str, Any], root: Path) -> dict[str, dict[int, float]]:
    result: dict[str, dict[int, float]] = defaultdict(dict)
    for item in manifest.get("completed", []):
        path = root / str(item["result"])
        if not path.is_file() or sha256(path) != item.get("sha256"):
            raise SystemExit(f"missing or corrupt QPS result: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        result[str(item["config"]["id"])][int(item["block"])] = float(data["qps"])
    return result


def read_result_checksums(
    manifest: dict[str, Any], root: Path,
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for item in manifest.get("completed", []):
        path = root / str(item["result"])
        if not path.is_file() or sha256(path) != item.get("sha256"):
            raise SystemExit(f"missing or corrupt QPS result: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        checksum = data.get("result_checksum")
        if checksum is not None:
            result[str(item["config"]["id"])].add(str(checksum))
    return result


def read_result_sets(
    manifest: dict[str, Any], root: Path,
) -> tuple[dict[str, dict[int, set[int]]], dict[str, bool]]:
    first: dict[str, dict[int, set[int]]] = {}
    hashes: dict[str, set[str]] = defaultdict(set)
    for item in manifest.get("completed", []):
        path = artifact(item, "result_records", root)
        if path is None:
            continue
        config = str(item["config"]["id"])
        hashes[config].add(sha256(path))
        current: dict[int, set[int]] = defaultdict(set)
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                current[int(row["query_id"])].add(int(row["label"]))
        if config not in first:
            first[config] = dict(current)
    return first, {config: len(values) == 1 for config, values in hashes.items()}


def ground_truth(path: Path, query_ids: Iterable[int], k: int) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    with path.open("rb") as source:
        raw = source.read(4)
        if len(raw) != 4:
            raise SystemExit("ground-truth ivecs is empty")
        dimension = struct.unpack("<i", raw)[0]
        if dimension < k:
            raise SystemExit("ground-truth dimension is smaller than k")
        row_bytes = 4 * (dimension + 1)
        for query_id in sorted(set(query_ids)):
            source.seek(query_id * row_bytes)
            raw_dimension = source.read(4)
            if len(raw_dimension) != 4 or struct.unpack("<i", raw_dimension)[0] != dimension:
                raise SystemExit(f"invalid ground-truth row {query_id}")
            labels = source.read(4 * dimension)
            if len(labels) != 4 * dimension:
                raise SystemExit(f"truncated ground-truth row {query_id}")
            result[query_id] = set(struct.unpack(
                f"<{dimension}i", labels)[:k])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--metrics", action="append", default=[])
    parser.add_argument("--baseline-metrics", action="append", default=[])
    parser.add_argument("--pair", action="append", default=[])
    parser.add_argument("--ground-truth-ivecs", type=Path)
    parser.add_argument(
        "--quality-summary", action="append", default=[],
        help="CONFIG=summary.json; baseline CONFIGs use baseline recall")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise SystemExit("QPS manifest is not complete")
    latencies, raw_latency = read_latency(manifest, args.run_root)
    if not latencies:
        raise SystemExit("run has no ordered latency artifacts")
    metrics = read_metrics(assignments(args.metrics, "metrics"), baseline=False)
    metrics.update(read_metrics(
        assignments(args.baseline_metrics, "baseline-metrics"), baseline=True))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "latency_records.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        fields = ("configuration_id", "block_id", "repeat_id", "query_id",
                  "latency_ns")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(raw_latency)

    joined_rows: list[dict[str, Any]] = []
    correlations: dict[str, dict[str, dict[str, float | None]]] = {}
    for config, per_query in sorted(latencies.items()):
        config_metrics = metrics.get(config, {})
        config_rows: list[dict[str, Any]] = []
        for query_id, samples in sorted(per_query.items()):
            row: dict[str, Any] = {
                "configuration_id": config,
                "query_id": query_id,
                "latency_median_ns": statistics.median(samples),
                "latency_samples": len(samples),
            }
            row.update(config_metrics.get(query_id, {}))
            joined_rows.append(row)
            config_rows.append(row)
        correlations[config] = {}
        for field in sorted({key for row in config_rows for key in row} - {
                "configuration_id", "query_id", "latency_median_ns",
                "latency_samples"}):
            paired = [
                (float(row["latency_median_ns"]), float(row[field]))
                for row in config_rows if field in row
            ]
            x = [value[0] for value in paired]
            y = [value[1] for value in paired]
            correlations[config][field] = {
                "n": len(paired),
                "pearson": pearson(x, y),
                "spearman": pearson(ranks(x), ranks(y)) if len(x) >= 3 else None,
            }

    joined_fields = [
        "configuration_id", "query_id", "latency_median_ns",
        "latency_samples",
    ] + sorted({key for row in joined_rows for key in row} - {
        "configuration_id", "query_id", "latency_median_ns", "latency_samples"})
    with (args.output_dir / "query_attribution.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=joined_fields)
        writer.writeheader()
        writer.writerows(joined_rows)

    pair_rows: list[dict[str, Any]] = []
    pair_summary: list[dict[str, Any]] = []
    qps = read_qps(manifest, args.run_root)
    for candidate, baseline in pair_assignments(args.pair):
        common = sorted(set(latencies.get(candidate, {})) &
                        set(latencies.get(baseline, {})))
        ratios: list[float] = []
        for query_id in common:
            candidate_ns = statistics.median(latencies[candidate][query_id])
            baseline_ns = statistics.median(latencies[baseline][query_id])
            ratio = candidate_ns / baseline_ns
            ratios.append(ratio)
            pair_rows.append({
                "candidate": candidate, "baseline": baseline,
                "query_id": query_id, "candidate_latency_ns": candidate_ns,
                "baseline_latency_ns": baseline_ns, "latency_ratio": ratio,
                "latency_delta_ns": candidate_ns - baseline_ns,
            })
        common_blocks = sorted(set(qps.get(candidate, {})) & set(qps.get(baseline, {})))
        block_speedups = [
            qps[candidate][block] / qps[baseline][block]
            for block in common_blocks
        ]
        speed_mean, speed_low, speed_high = confidence_interval(block_speedups)
        pair_summary.append({
            "candidate": candidate, "baseline": baseline, "queries": len(common),
            "latency_ratio_median": statistics.median(ratios) if ratios else None,
            "latency_ratio_p95": quantile(ratios, .95) if ratios else None,
            "latency_ratio_p99": quantile(ratios, .99) if ratios else None,
            "paired_qps_blocks": len(common_blocks),
            "paired_qps_speedup_mean": speed_mean,
            "paired_qps_speedup_ci_low": speed_low,
            "paired_qps_speedup_ci_high": speed_high,
        })
    with (args.output_dir / "pairwise_query_latency.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        fields = ("candidate", "baseline", "query_id",
                  "candidate_latency_ns", "baseline_latency_ns",
                  "latency_ratio", "latency_delta_ns")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pair_rows)

    result_sets, result_consistency = read_result_sets(manifest, args.run_root)
    result_checksums = read_result_checksums(manifest, args.run_root)
    for summary in pair_summary:
        candidate = str(summary["candidate"])
        baseline = str(summary["baseline"])
        common_results = sorted(
            set(result_sets.get(candidate, {})) &
            set(result_sets.get(baseline, {})))
        summary["paired_result_queries"] = len(common_results)
        summary["top_k_label_sets_identical"] = (
            all(result_sets[candidate][query_id] ==
                result_sets[baseline][query_id]
                for query_id in common_results)
            if common_results else None)
        candidate_checksums = result_checksums.get(candidate, set())
        baseline_checksums = result_checksums.get(baseline, set())
        summary["result_checksums_identical"] = (
            candidate_checksums == baseline_checksums
            if candidate_checksums and baseline_checksums else None)
    recalls: dict[str, float] = {}
    if args.ground_truth_ivecs and result_sets:
        query_ids = {query for values in result_sets.values() for query in values}
        truth = ground_truth(args.ground_truth_ivecs, query_ids, args.k)
        for config, queries in result_sets.items():
            recalls[config] = statistics.fmean(
                len(labels & truth[query_id]) / args.k
                for query_id, labels in queries.items())

    quality_recalls: dict[str, float] = {}
    recall_deltas: dict[str, float] = {}
    for config, path in assignments(
            args.quality_summary, "quality-summary").items():
        data = json.loads(path.read_text(encoding="utf-8"))
        field = ("mean_baseline_recall_at_k" if config.startswith("baseline")
                 else "mean_v0_recall_at_k")
        quality_recalls[config] = float(data[field])
        if config in recalls:
            recall_deltas[config] = recalls[config] - quality_recalls[config]

    observations = manifest.get("resource_observations", [])
    affinity_sets = [entry.get("allowed_cpus") for entry in observations]
    report = {
        "schema_version": 1,
        "status": "complete",
        "run_root": str(args.run_root.resolve()),
        "single_cpu_affinity_observed": bool(affinity_sets) and all(
            isinstance(value, list) and len(value) == 1 for value in affinity_sets),
        "affinity_sets": affinity_sets,
        "correlations": correlations,
        "pairs": pair_summary,
        "result_records_consistent_across_blocks": result_consistency,
        "performance_result_recall_at_k": recalls,
        "quality_reference_recall_at_k": quality_recalls,
        "performance_minus_quality_recall": recall_deltas,
    }
    (args.output_dir / "attribution_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"p0_attribution_complete configs={len(latencies)} "
        f"rows={len(joined_rows)} output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
