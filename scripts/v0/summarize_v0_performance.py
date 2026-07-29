#!/usr/bin/env python3
"""Validate and summarize a repeated V0 real-pruning performance matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SUMMARY_COUNTERS = (
    "bound_evaluated",
    "bound_pruned",
    "exact_fallback",
    "exact_only_fallback",
    "exact_distance_saved",
    "lower_bound_violation",
    "false_prune",
)

CSV_FIELDS = (
    "query_id",
    "baseline_recall_at_k",
    "v0_recall_at_k",
    "results_equal",
    "baseline_latency_ns",
    "v0_latency_ns",
    *SUMMARY_COUNTERS,
)

COMMON_METADATA_FIELDS = (
    "dataset",
    "dimension",
    "n_base",
    "k",
    "M_pq",
    "nbits",
    "code_bytes_per_edge",
    "directed_edge_count",
    "index_sha256",
    "sidecar_sha256",
    "codebook_sha256",
    "sidecar_bytes",
    "producer_git_commit",
    "git_branch",
    "working_tree_dirty",
    "shadow_validation_compiled",
    "real_pruning_compiled",
)

OUTPUT_FIELDS = (
    "ef_search",
    "repeats",
    "queries_per_repeat",
    "measured_queries",
    "mean_baseline_recall_at_k",
    "mean_v0_recall_at_k",
    "results_equal_rate",
    "baseline_latency_us_mean",
    "baseline_latency_us_p50",
    "baseline_latency_us_p95",
    "baseline_latency_us_p99",
    "v0_latency_us_mean",
    "v0_latency_us_p50",
    "v0_latency_us_p95",
    "v0_latency_us_p99",
    "baseline_qps",
    "v0_qps",
    "v0_over_baseline",
    "baseline_over_v0_speedup",
    "repeat_v0_over_baseline_mean",
    "repeat_v0_over_baseline_stddev",
    "bound_evaluated",
    "bound_pruned",
    "prune_rate",
    "exact_fallback",
    "exact_only_fallback",
    "exact_distance_saved",
    "exact_distance_saved_per_query",
    "lower_bound_violation",
    "false_prune",
)


@dataclass(frozen=True)
class Run:
    path: Path
    metadata: dict[str, Any]
    summary: dict[str, Any]
    baseline_recall: list[float]
    v0_recall: list[float]
    results_equal: list[int]
    baseline_latency_ns: list[int]
    v0_latency_ns: list[int]
    counters: dict[str, list[int]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _atomic_text(path: Path, text: str) -> None:
    partial = Path(str(path) + ".partial")
    if path.exists() or partial.exists():
        raise RuntimeError(f"refusing to overwrite output: {path}")
    with partial.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


def _integer(row: dict[str, str], field: str, path: Path) -> int:
    try:
        return int(row[field])
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"invalid integer field {field} in {path}") from exc


def _number(row: dict[str, str], field: str, path: Path) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"invalid numeric field {field} in {path}") from exc
    _require(math.isfinite(value), f"non-finite field {field} in {path}")
    return value


def _load_run(
    run_dir: Path,
    *,
    dataset: str,
    ef_search: int,
    expected_queries: int,
    allow_dirty: bool,
) -> Run:
    metadata_path = run_dir / "metadata.json"
    summary_path = run_dir / "summary.json"
    complete_path = run_dir / "complete.json"
    query_path = run_dir / "query_metrics.csv"
    for path in (metadata_path, summary_path, complete_path, query_path):
        _require(path.is_file(), f"missing run artifact: {path}")

    metadata = _load_json(metadata_path)
    summary = _load_json(summary_path)
    complete = _load_json(complete_path)

    _require(
        metadata.get("format") == "hnswlib_v0_search_run"
        and metadata.get("format_version") == 1,
        f"unsupported metadata format: {metadata_path}",
    )
    _require(
        summary.get("format") == "hnswlib_v0_search_summary"
        and summary.get("format_version") == 1,
        f"unsupported summary format: {summary_path}",
    )
    _require(summary.get("status") == "valid", f"invalid run summary: {summary_path}")
    _require(complete.get("status") == "complete", f"incomplete run: {complete_path}")
    _require(complete.get("mode") == "prune", f"completion mode is not prune: {complete_path}")
    _require(metadata.get("dataset") == dataset, f"dataset mismatch: {metadata_path}")
    _require(metadata.get("mode") == "prune", f"metadata mode is not prune: {metadata_path}")
    _require(metadata.get("query_start") == 0, f"query_start must be zero: {metadata_path}")
    _require(
        metadata.get("query_count") == expected_queries
        and summary.get("query_count") == expected_queries
        and complete.get("query_count") == expected_queries,
        f"query-count mismatch: {run_dir}",
    )
    _require(metadata.get("ef_search") == ef_search, f"efSearch mismatch: {metadata_path}")
    _require(metadata.get("real_pruning_enabled") is True, f"pruning disabled: {metadata_path}")
    _require(metadata.get("real_pruning_compiled") is True, f"pruning not compiled: {metadata_path}")
    _require(
        metadata.get("shadow_validation_compiled") is False,
        f"formal performance run compiled shadow validation: {metadata_path}",
    )
    _require(
        metadata.get("bound_pruned_semantics") == "actual_prune",
        f"bound-pruned semantics mismatch: {metadata_path}",
    )
    if not allow_dirty:
        _require(
            metadata.get("working_tree_dirty") == "false",
            f"formal run used a dirty working tree: {metadata_path}",
        )

    _require(summary.get("mismatch_queries") == 0, f"result mismatch: {summary_path}")
    _require(summary.get("lower_bound_violation") == 0, f"lower-bound violation: {summary_path}")
    _require(summary.get("false_prune") == 0, f"false prune: {summary_path}")
    _require(
        summary.get("bound_pruned") == summary.get("exact_distance_saved"),
        f"pruned/saved mismatch: {summary_path}",
    )

    baseline_recall: list[float] = []
    v0_recall: list[float] = []
    results_equal: list[int] = []
    baseline_latency_ns: list[int] = []
    v0_latency_ns: list[int] = []
    counters = {field: [] for field in SUMMARY_COUNTERS}
    query_ids: list[int] = []
    with query_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames is not None, f"missing CSV header: {query_path}")
        missing = sorted(set(CSV_FIELDS) - set(reader.fieldnames))
        _require(not missing, f"missing CSV fields {missing}: {query_path}")
        for row in reader:
            query_ids.append(_integer(row, "query_id", query_path))
            baseline_recall.append(_number(row, "baseline_recall_at_k", query_path))
            v0_recall.append(_number(row, "v0_recall_at_k", query_path))
            results_equal.append(_integer(row, "results_equal", query_path))
            baseline_latency_ns.append(_integer(row, "baseline_latency_ns", query_path))
            v0_latency_ns.append(_integer(row, "v0_latency_ns", query_path))
            for field in SUMMARY_COUNTERS:
                counters[field].append(_integer(row, field, query_path))

    _require(len(query_ids) == expected_queries, f"CSV row-count mismatch: {query_path}")
    _require(query_ids == list(range(expected_queries)), f"query IDs are not contiguous: {query_path}")
    _require(all(value > 0 for value in baseline_latency_ns), f"non-positive baseline latency: {query_path}")
    _require(all(value > 0 for value in v0_latency_ns), f"non-positive V0 latency: {query_path}")
    _require(
        all(0.0 <= value <= 1.0 for value in baseline_recall + v0_recall),
        f"recall outside [0, 1]: {query_path}",
    )
    _require(all(value in (0, 1) for value in results_equal), f"invalid results_equal: {query_path}")
    _require(
        all(value >= 0 for values in counters.values() for value in values),
        f"negative counter: {query_path}",
    )

    _require(
        sum(baseline_latency_ns) == int(summary["baseline_latency_ns"]),
        f"baseline latency does not close to summary: {run_dir}",
    )
    _require(
        sum(v0_latency_ns) == int(summary["v0_latency_ns"]),
        f"V0 latency does not close to summary: {run_dir}",
    )
    for field in SUMMARY_COUNTERS:
        _require(
            sum(counters[field]) == int(summary[field]),
            f"{field} does not close to summary: {run_dir}",
        )
    _require(
        results_equal.count(0) == int(summary["mismatch_queries"]),
        f"mismatch count does not close to summary: {run_dir}",
    )
    _require(
        math.isclose(
            math.fsum(baseline_recall) / expected_queries,
            float(summary["mean_baseline_recall_at_k"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        f"baseline recall does not close to summary: {run_dir}",
    )
    _require(
        math.isclose(
            math.fsum(v0_recall) / expected_queries,
            float(summary["mean_v0_recall_at_k"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        f"V0 recall does not close to summary: {run_dir}",
    )

    return Run(
        path=run_dir,
        metadata=metadata,
        summary=summary,
        baseline_recall=baseline_recall,
        v0_recall=v0_recall,
        results_equal=results_equal,
        baseline_latency_ns=baseline_latency_ns,
        v0_latency_ns=v0_latency_ns,
        counters=counters,
    )


def _percentile(values: Iterable[int], percentile: float) -> float:
    ordered = sorted(values)
    _require(bool(ordered), "cannot calculate a percentile of an empty sequence")
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _same_contract(runs: list[Run]) -> dict[str, Any]:
    _require(bool(runs), "no runs were loaded")
    common = {field: runs[0].metadata[field] for field in COMMON_METADATA_FIELDS}
    for run in runs[1:]:
        for field, expected in common.items():
            _require(
                run.metadata.get(field) == expected,
                f"runs disagree on {field}: {run.path}",
            )
    return common


def _aggregate(ef_search: int, runs: list[Run], queries_per_repeat: int) -> dict[str, Any]:
    baseline_latency = [value for run in runs for value in run.baseline_latency_ns]
    v0_latency = [value for run in runs for value in run.v0_latency_ns]
    baseline_recall = [value for run in runs for value in run.baseline_recall]
    v0_recall = [value for run in runs for value in run.v0_recall]
    results_equal = [value for run in runs for value in run.results_equal]
    counters = {
        field: sum(sum(run.counters[field]) for run in runs)
        for field in SUMMARY_COUNTERS
    }
    measured_queries = len(baseline_latency)
    baseline_total = sum(baseline_latency)
    v0_total = sum(v0_latency)
    repeat_ratios = [
        sum(run.v0_latency_ns) / sum(run.baseline_latency_ns)
        for run in runs
    ]
    row = {
        "ef_search": ef_search,
        "repeats": len(runs),
        "queries_per_repeat": queries_per_repeat,
        "measured_queries": measured_queries,
        "mean_baseline_recall_at_k": math.fsum(baseline_recall) / measured_queries,
        "mean_v0_recall_at_k": math.fsum(v0_recall) / measured_queries,
        "results_equal_rate": sum(results_equal) / measured_queries,
        "baseline_latency_us_mean": baseline_total / measured_queries / 1e3,
        "baseline_latency_us_p50": _percentile(baseline_latency, 50) / 1e3,
        "baseline_latency_us_p95": _percentile(baseline_latency, 95) / 1e3,
        "baseline_latency_us_p99": _percentile(baseline_latency, 99) / 1e3,
        "v0_latency_us_mean": v0_total / measured_queries / 1e3,
        "v0_latency_us_p50": _percentile(v0_latency, 50) / 1e3,
        "v0_latency_us_p95": _percentile(v0_latency, 95) / 1e3,
        "v0_latency_us_p99": _percentile(v0_latency, 99) / 1e3,
        "baseline_qps": measured_queries * 1e9 / baseline_total,
        "v0_qps": measured_queries * 1e9 / v0_total,
        "v0_over_baseline": v0_total / baseline_total,
        "baseline_over_v0_speedup": baseline_total / v0_total,
        "repeat_v0_over_baseline_mean": statistics.mean(repeat_ratios),
        "repeat_v0_over_baseline_stddev": (
            statistics.stdev(repeat_ratios) if len(repeat_ratios) > 1 else 0.0
        ),
        **counters,
        "prune_rate": (
            counters["bound_pruned"] / counters["bound_evaluated"]
            if counters["bound_evaluated"]
            else 0.0
        ),
        "exact_distance_saved_per_query": (
            counters["exact_distance_saved"] / measured_queries
        ),
    }
    return row


def _report(aggregate: dict[str, Any]) -> str:
    common = aggregate["common_metadata"]
    protocol = aggregate["protocol"]
    rows = aggregate["configurations"]
    lines = [
        "# V0 real-pruning performance report",
        "",
        f"- Dataset: {common['dataset']}",
        f"- Index dimension: {common['dimension']}",
        f"- HNSW query k: {common['k']}",
        f"- PQ configuration: M={common['M_pq']}, nbits={common['nbits']}",
        f"- Code bytes per directed edge: {common['code_bytes_per_edge']}",
        f"- Sidecar bytes: {common['sidecar_bytes']}",
        f"- Producer commit: `{common['producer_git_commit']}`",
        f"- Measured repeats per efSearch: {protocol['measured_repeats']}",
        f"- Queries per measured repeat: {protocol['queries_per_repeat']}",
        f"- Warm-up queries per efSearch: {protocol['warmup_queries']}",
        "",
        "| efSearch | Recall@10 baseline/V0 | equal | prune rate | saved/query | "
        "baseline mean us | V0 mean us | baseline P99 us | V0 P99 us | "
        "V0 / baseline | speedup |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['ef_search']} | "
            f"{row['mean_baseline_recall_at_k']:.6f} / "
            f"{row['mean_v0_recall_at_k']:.6f} | "
            f"{row['results_equal_rate']:.6f} | "
            f"{100.0 * row['prune_rate']:.6f}% | "
            f"{row['exact_distance_saved_per_query']:.6f} | "
            f"{row['baseline_latency_us_mean']:.3f} | "
            f"{row['v0_latency_us_mean']:.3f} | "
            f"{row['baseline_latency_us_p99']:.3f} | "
            f"{row['v0_latency_us_p99']:.3f} | "
            f"{row['v0_over_baseline']:.4f} | "
            f"{row['baseline_over_v0_speedup']:.4f}x |"
        )
    lines.extend(
        [
            "",
            "Latency percentiles pool all measured per-query observations across the "
            "five repeats. Warm-up runs are validated but excluded.",
            "",
            "QPS and latency cover the timed search calls only; index loading, sidecar "
            "loading, result serialization, and Slurm startup are excluded.",
            "",
            "Each query measures baseline first and V0 second in the same runner. "
            "The five-repeat dispersion is recorded in the JSON and CSV outputs.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_positive_list(value: str, name: str) -> list[int]:
    try:
        parsed = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a comma-separated integer list") from exc
    _require(parsed and all(item > 0 for item in parsed), f"{name} values must be positive")
    _require(len(set(parsed)) == len(parsed), f"{name} values must be unique")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and summarize a V0 real-pruning performance matrix."
    )
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-ef-search", default="50,100,200,400")
    parser.add_argument("--expected-repeats", type=int, default=5)
    parser.add_argument("--expected-queries", type=int, default=1000)
    parser.add_argument("--expected-warmup-queries", type=int, default=100)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    try:
        ef_values = _parse_positive_list(args.expected_ef_search, "expected efSearch")
        _require(args.expected_repeats > 0, "expected repeats must be positive")
        _require(args.expected_queries > 0, "expected queries must be positive")
        _require(args.expected_warmup_queries > 0, "expected warm-up queries must be positive")
        _require(args.matrix_root.is_dir(), f"matrix root does not exist: {args.matrix_root}")
        marker = _load_json(args.matrix_root / "matrix_complete.json")
        _require(marker.get("status") == "complete", "matrix completion marker is not complete")
        _require(
            marker.get("repeats") == args.expected_repeats,
            "matrix completion marker repeat count does not match",
        )
        dataset = str(marker["dataset"])

        results_root = args.matrix_root / "results"
        _require(results_root.is_dir(), f"results directory does not exist: {results_root}")
        actual_ef = sorted(
            int(path.name[2:])
            for path in results_root.iterdir()
            if path.is_dir() and path.name.startswith("ef") and path.name[2:].isdigit()
        )
        _require(actual_ef == sorted(ef_values), f"matrix contains efSearch={actual_ef}, expected {ef_values}")

        warmups: list[Run] = []
        measured_by_ef: dict[int, list[Run]] = {}
        for ef_search in ef_values:
            ef_dir = results_root / f"ef{ef_search}"
            warmups.append(
                _load_run(
                    ef_dir / "warmup",
                    dataset=dataset,
                    ef_search=ef_search,
                    expected_queries=args.expected_warmup_queries,
                    allow_dirty=args.allow_dirty,
                )
            )
            runs = []
            for repeat in range(1, args.expected_repeats + 1):
                runs.append(
                    _load_run(
                        ef_dir / f"repeat{repeat}",
                        dataset=dataset,
                        ef_search=ef_search,
                        expected_queries=args.expected_queries,
                        allow_dirty=args.allow_dirty,
                    )
                )
            actual_repeat_dirs = sorted(
                path.name
                for path in ef_dir.iterdir()
                if path.is_dir() and path.name.startswith("repeat")
            )
            expected_repeat_dirs = [
                f"repeat{repeat}" for repeat in range(1, args.expected_repeats + 1)
            ]
            _require(
                actual_repeat_dirs == expected_repeat_dirs,
                f"repeat directories mismatch under {ef_dir}",
            )
            measured_by_ef[ef_search] = runs

        all_runs = warmups + [
            run
            for ef_search in ef_values
            for run in measured_by_ef[ef_search]
        ]
        common = _same_contract(all_runs)
        rows = [
            _aggregate(ef_search, measured_by_ef[ef_search], args.expected_queries)
            for ef_search in ef_values
        ]
        _require(all(row["results_equal_rate"] == 1.0 for row in rows), "not all results match baseline")
        _require(all(row["lower_bound_violation"] == 0 for row in rows), "lower-bound violation detected")
        _require(all(row["false_prune"] == 0 for row in rows), "false prune detected")
        _require(
            all(row["bound_pruned"] == row["exact_distance_saved"] for row in rows),
            "aggregate pruned/saved counts differ",
        )

        if args.output_dir.exists():
            _require(
                not any(args.output_dir.iterdir()),
                f"output directory is not empty: {args.output_dir}",
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        aggregate = {
            "format": "hnswlib_v0_performance_summary",
            "format_version": 1,
            "status": "valid",
            "matrix_root": str(args.matrix_root.resolve()),
            "common_metadata": common,
            "protocol": {
                "ef_search_values": ef_values,
                "measured_repeats": args.expected_repeats,
                "queries_per_repeat": args.expected_queries,
                "warmup_queries": args.expected_warmup_queries,
                "warmup_excluded_from_metrics": True,
                "latency_percentiles": "linear interpolation over pooled query-level latencies",
                "qps_scope": "timed search calls only",
            },
            "configurations": rows,
        }

        csv_path = args.output_dir / "performance_summary.csv"
        csv_partial = Path(str(csv_path) + ".partial")
        with csv_partial.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        csv_partial.replace(csv_path)
        _atomic_text(
            args.output_dir / "performance_summary.json",
            json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        )
        _atomic_text(args.output_dir / "report.md", _report(aggregate))
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
