from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.edge_estimation.contracts import (  # noqa: E402
    atomic_output_dir, file_entry, load_strict_json, sha256_file)


METHODS = ("pq8", "pq4", "opq", "prq", "jq", "rabitq")
PROFILES = ("balanced", "strict")
QUALITY_METRICS = (
    "correct_prune_rate",
    "global_false_prune_rate",
    "conditional_false_prune_rate",
    "prunable_coverage",
    "prune_precision",
    "total_prune_rate",
    "valid_estimate_coverage",
    "p95_query_false_prune_rate",
    "queries_with_decisions",
)


class ExportError(ValueError):
    pass


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExportError(f"{name} must be an object")
    return value


def _list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ExportError(f"{name} must be an array")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, fields: Sequence[str],
               rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields),
                                extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _metric_columns(metrics: Mapping[str, Any]) -> dict[str, object]:
    return {name: metrics.get(name) for name in QUALITY_METRICS}


def _selected_alpha(selections: Mapping[str, Any], profile: str) -> float:
    selection = _mapping(selections.get(profile), f"selection.{profile}")
    if selection.get("status") != "selected":
        raise ExportError(f"{profile} operating point was not selected")
    return float(selection["alpha"])


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def collect_quality(run_root: Path, sources: set[Path]) -> tuple[
        list[dict[str, object]], list[dict[str, object]]]:
    sweep_rows: list[dict[str, object]] = []
    operating_rows: list[dict[str, object]] = []
    for method in METHODS:
        directory = run_root / f"quality-v2-{method}"
        quality_path = directory / "quality.json"
        selection_path = directory / "selection.json"
        policy_path = directory / "policy.json"
        for path in (quality_path, selection_path, policy_path):
            if not path.is_file():
                raise FileNotFoundError(path)
            sources.add(path)
        quality = _mapping(load_strict_json(quality_path), str(quality_path))
        selection = _mapping(
            load_strict_json(selection_path), str(selection_path))
        policy = _mapping(load_strict_json(policy_path), str(policy_path))
        analyses = _list(quality.get("analyses"), f"{method}.analyses")
        selections = _mapping(selection.get("selections"),
                              f"{method}.selections")
        gates = _mapping(policy.get("gates"), f"{method}.gates")
        selected = {profile: _selected_alpha(selections, profile)
                    for profile in PROFILES}
        for raw in analyses:
            analysis = _mapping(raw, f"{method}.analysis")
            alpha = float(analysis["alpha"])
            counts = _mapping(analysis.get("counts"), f"{method}.counts")
            metrics = _mapping(analysis.get("metrics"), f"{method}.metrics")
            row: dict[str, object] = {
                "method": method,
                "backend": quality.get("backend"),
                "alpha": alpha,
                **{name: counts.get(name) for name in
                   ("decision_count_s", "valid_estimate_count",
                    "fallback_count", "tp", "fp", "fn", "tn")},
                **_metric_columns(metrics),
            }
            for profile in PROFILES:
                gate = _mapping(gates.get(profile), f"gates.{profile}")
                global_fp = metrics.get("global_false_prune_rate")
                p95_fp = metrics.get("p95_query_false_prune_rate")
                row[f"passes_{profile}"] = bool(
                    global_fp is not None and p95_fp is not None and
                    float(global_fp) <= float(gate["max_global_false_prune_rate"]) and
                    float(p95_fp) <= float(gate["max_p95_query_false_prune_rate"]))
                row[f"selected_{profile}"] = _close(alpha, selected[profile])
            sweep_rows.append(row)
        for profile in PROFILES:
            value = _mapping(selections[profile],
                             f"{method}.selection.{profile}")
            metrics = _mapping(value.get("metrics"),
                               f"{method}.{profile}.metrics")
            gate = _mapping(gates[profile], f"gates.{profile}")
            operating_rows.append({
                "method": method,
                "profile": profile,
                "alpha": float(value["alpha"]),
                **_metric_columns(metrics),
                "max_global_false_prune_rate":
                    gate["max_global_false_prune_rate"],
                "max_p95_query_false_prune_rate":
                    gate["max_p95_query_false_prune_rate"],
            })
    return sweep_rows, operating_rows


def collect_final_timing(run_root: Path, batch_size: int,
                         sources: set[Path]) -> tuple[
        list[dict[str, object]], list[dict[str, object]]]:
    directory = run_root / f"timing-final-horizontal-b{batch_size}"
    result_path = directory / "result.json"
    complete_path = directory / "complete.json"
    for path in (result_path, complete_path):
        if not path.is_file():
            raise FileNotFoundError(path)
        sources.add(path)
    result = _mapping(load_strict_json(result_path), str(result_path))
    if (result.get("mode") != "randomized_paired_selected_query_preparation" or
            _mapping(result.get("evidence"), "timing.evidence").get(
                "formal_admitted") is not True):
        raise ExportError("final horizontal timing is not formally admitted")
    raw = _list(result.get("raw_records"), "timing.raw_records")
    summary = _mapping(result.get("summary"), "timing.summary")
    methods_summary = _mapping(summary.get("methods"), "timing.summary.methods")
    if set(methods_summary) != set(METHODS):
        raise ExportError("final timing does not contain all six methods")
    raw_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    method_records: dict[str, list[Mapping[str, Any]]] = {
        method: [] for method in METHODS}
    for index, raw_record in enumerate(raw):
        record = _mapping(raw_record, f"timing.raw_records[{index}]")
        method = str(record.get("method"))
        if method not in method_records:
            raise ExportError(f"unexpected timing method: {method}")
        method_records[method].append(record)
        elapsed = float(record["elapsed_ns"])
        queries = int(record["query_count"])
        edges = int(record["eligible_events"])
        memory = _mapping(record.get("memory_report"), "timing.memory_report")
        raw_rows.append({
            "method": method,
            "block": record["block"],
            "repeat": record["repeat"],
            "position": record["position"],
            "elapsed_ns": int(record["elapsed_ns"]),
            "elapsed_ms": elapsed / 1e6,
            "query_count": queries,
            "eligible_edges": edges,
            "ns_per_query": elapsed / queries,
            "ns_per_eligible_edge": None if not edges else elapsed / edges,
            "execution_mode": record.get("execution_mode"),
            "query_batch_size": record.get("query_batch_size"),
            "batch_rotation_engine": record.get("batch_rotation_engine"),
            "native_mode": record.get("native_mode"),
            "checksum": record.get("checksum"),
            "backend_bytes": memory.get("backend_bytes"),
            "query_store_bytes": memory.get("query_store_bytes"),
            "scratch_bytes": memory.get("scratch_bytes"),
        })
    counts = {method: len(records) for method, records in method_records.items()}
    if len(set(counts.values())) != 1 or not next(iter(counts.values())):
        raise ExportError(f"final timing record counts are unbalanced: {counts}")
    for method in METHODS:
        item = _mapping(methods_summary[method], f"timing.summary.{method}")
        records = method_records[method]
        block_medians = [float(value) for value in _list(
            item.get("block_median_elapsed_ns"),
            f"timing.summary.{method}.block_medians")]
        mean = statistics.mean(block_medians)
        cv = (0.0 if len(block_medians) < 2 or mean == 0 else
              statistics.stdev(block_medians) / mean * 100.0)
        memory = _mapping(records[0]["memory_report"],
                          f"timing.{method}.memory_report")
        execution_modes = {str(record.get("execution_mode")) for record in records}
        batch_sizes = {int(record.get("query_batch_size", 1)) for record in records}
        if len(execution_modes) != 1 or len(batch_sizes) != 1:
            raise ExportError(f"{method} execution contract changed")
        summary_rows.append({
            "method": method,
            "reference": summary.get("reference"),
            "execution_mode": next(iter(execution_modes)),
            "query_batch_size": next(iter(batch_sizes)),
            "record_count": len(records),
            "block_count": len(block_medians),
            "median_elapsed_ns": item["median_elapsed_ns"],
            "median_elapsed_ms": float(item["median_elapsed_ns"]) / 1e6,
            "ns_per_query": item["ns_per_query"],
            "ns_per_eligible_edge": item["ns_per_eligible_edge"],
            "paired_speed_ratio_vs_reference":
                item["paired_speed_ratio_vs_reference"],
            "block_min_ms": min(block_medians) / 1e6,
            "block_max_ms": max(block_medians) / 1e6,
            "block_cv_percent": cv,
            "backend_bytes": memory.get("backend_bytes"),
            "backend_mib": float(memory.get("backend_bytes", 0)) / 2**20,
            "query_store_bytes": memory.get("query_store_bytes"),
            "scratch_bytes": memory.get("scratch_bytes"),
            "checksum_count": len({record.get("checksum") for record in records}),
        })
    return raw_rows, summary_rows


def collect_batch_scaling(run_root: Path, batch_sizes: Sequence[int],
                          sources: set[Path]) -> tuple[
        list[dict[str, object]], list[dict[str, object]]]:
    scaling: list[dict[str, object]] = []
    parity_rows: list[dict[str, object]] = []
    baseline: dict[str, float] = {}
    for batch_size in batch_sizes:
        result_path = run_root / f"timing-batch-b{batch_size}" / "result.json"
        complete_path = result_path.parent / "complete.json"
        for path in (result_path, complete_path):
            if not path.is_file():
                raise FileNotFoundError(path)
            sources.add(path)
        result = _mapping(load_strict_json(result_path), str(result_path))
        if int(result.get("query_batch_size", 0)) != batch_size:
            raise ExportError(f"batch result identity mismatch: {batch_size}")
        records = [_mapping(value, "batch.raw_record") for value in
                   _list(result.get("raw_records"), "batch.raw_records")]
        engines = {record.get("batch_rotation_engine") for record in records}
        if engines != {"blas_sgemv_sgemm"}:
            raise ExportError(f"batch {batch_size} did not use BLAS: {engines}")
        summary = _mapping(
            _mapping(result.get("summary"), "batch.summary").get("methods"),
            "batch.summary.methods")
        for method in ("opq", "jq"):
            item = _mapping(summary.get(method), f"batch.{method}")
            method_records = [record for record in records
                              if record.get("method") == method]
            if not method_records:
                raise ExportError(f"batch {batch_size} has no {method} records")
            memory = _mapping(method_records[0]["memory_report"],
                              f"batch.{method}.memory")
            ns_query = float(item["ns_per_query"])
            if batch_size == batch_sizes[0]:
                baseline[method] = ns_query
            scaling.append({
                "batch_size": batch_size,
                "method": method,
                "record_count": len(method_records),
                "median_elapsed_ns": item["median_elapsed_ns"],
                "median_elapsed_ms": float(item["median_elapsed_ns"]) / 1e6,
                "ns_per_query": ns_query,
                "ns_per_eligible_edge": item["ns_per_eligible_edge"],
                "speedup_vs_batch1": baseline[method] / ns_query,
                "paired_speed_ratio_vs_opq":
                    item["paired_speed_ratio_vs_reference"],
                "backend_bytes": memory.get("backend_bytes"),
                "backend_mib": float(memory.get("backend_bytes", 0)) / 2**20,
                "batch_rotation_engine": "blas_sgemv_sgemm",
            })
            parity_path = run_root / f"batch-parity-b{batch_size}-{method}.json"
            parity_rows.append(_parity_row(
                parity_path, "batch-scaling", method, batch_size, sources))
    return scaling, parity_rows


def _parity_row(path: Path, scope: str, method: str, batch_size: int,
                sources: set[Path]) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    sources.add(path)
    value = _mapping(load_strict_json(path), str(path))
    valid = bool(
        value.get("valid") is True and
        value.get("batch_rotation_engine") == "blas_sgemv_sgemm" and
        int(value.get("nonfinite_mismatch_count", -1)) == 0 and
        int(value.get("tolerance_failure_count", -1)) == 0)
    if not valid:
        raise ExportError(f"batch parity is not passing: {path}")
    return {
        "scope": scope,
        "batch_size": batch_size,
        "method": method,
        "valid": valid,
        "backend": value.get("backend"),
        "batch_rotation_engine": value.get("batch_rotation_engine"),
        "compared_count": value.get("compared_count"),
        "nonfinite_mismatch_count": value.get("nonfinite_mismatch_count"),
        "tolerance_failure_count": value.get("tolerance_failure_count"),
        "max_absolute_error": value.get("max_absolute_error"),
        "max_relative_error": value.get("max_relative_error"),
        "absolute_tolerance": value.get("absolute_tolerance"),
        "relative_tolerance": value.get("relative_tolerance"),
    }


QUALITY_SWEEP_FIELDS = (
    "method", "backend", "alpha", "decision_count_s",
    "valid_estimate_count", "fallback_count", "tp", "fp", "fn", "tn",
    *QUALITY_METRICS, "passes_balanced", "passes_strict",
    "selected_balanced", "selected_strict")
OPERATING_FIELDS = (
    "method", "profile", "alpha", *QUALITY_METRICS,
    "max_global_false_prune_rate", "max_p95_query_false_prune_rate")
TIMING_RAW_FIELDS = (
    "method", "block", "repeat", "position", "elapsed_ns", "elapsed_ms",
    "query_count", "eligible_edges", "ns_per_query", "ns_per_eligible_edge",
    "execution_mode", "query_batch_size", "batch_rotation_engine",
    "native_mode", "checksum", "backend_bytes", "query_store_bytes",
    "scratch_bytes")
TIMING_SUMMARY_FIELDS = (
    "method", "reference", "execution_mode", "query_batch_size",
    "record_count", "block_count", "median_elapsed_ns", "median_elapsed_ms",
    "ns_per_query", "ns_per_eligible_edge",
    "paired_speed_ratio_vs_reference", "block_min_ms", "block_max_ms",
    "block_cv_percent", "backend_bytes", "backend_mib", "query_store_bytes",
    "scratch_bytes", "checksum_count")
BATCH_SCALING_FIELDS = (
    "batch_size", "method", "record_count", "median_elapsed_ns",
    "median_elapsed_ms", "ns_per_query", "ns_per_eligible_edge",
    "speedup_vs_batch1", "paired_speed_ratio_vs_opq", "backend_bytes",
    "backend_mib", "batch_rotation_engine")
PARITY_FIELDS = (
    "scope", "batch_size", "method", "valid", "backend",
    "batch_rotation_engine", "compared_count", "nonfinite_mismatch_count",
    "tolerance_failure_count", "max_absolute_error", "max_relative_error",
    "absolute_tolerance", "relative_tolerance")


README = """# Unified quantizer comparison data

This directory is generated from frozen experiment JSON; do not edit the CSVs by hand.

- `quality_sweep.csv`: one row per method and alpha.
- `operating_points.csv`: balanced and strict selections for every method.
- `timing_raw.csv`: randomized paired timing observations.
- `timing_summary.csv`: method-level component timing and memory summary.
- `batch_scaling.csv`: OPQ/JQ batch-size scaling data.
- `batch_parity.csv`: scalar-versus-batch numerical parity evidence.
- `comparison_summary.json`: machine-readable joined summary.
- `manifest.json`: source identities and export audit.

`correct_prune_rate` is TP divided by all valid decisions; it is not Recall@K.
Timing measures estimator replay on the frozen trace; it is not end-to-end HNSW QPS.
OPQ and JQ use the selected batched query-preparation path in final timing, while the
other four methods use their scalar path. Rates are stored as fractions, not percentages.
"""


def export(run_root: Path, output: Path, *, batch_sizes: Sequence[int],
           final_batch_size: int) -> None:
    if not batch_sizes or any(value <= 0 for value in batch_sizes):
        raise ExportError("batch sizes must be positive")
    if batch_sizes[0] != 1 or len(set(batch_sizes)) != len(batch_sizes):
        raise ExportError("batch sizes must be unique and start with 1")
    sources: set[Path] = set()
    quality_sweep, operating = collect_quality(run_root, sources)
    timing_raw, timing_summary = collect_final_timing(
        run_root, final_batch_size, sources)
    batch_scaling, parity = collect_batch_scaling(
        run_root, batch_sizes, sources)
    for method in ("opq", "jq"):
        parity.append(_parity_row(
            run_root /
            f"final-horizontal-b{final_batch_size}-parity-{method}.json",
            "final-horizontal", method, final_batch_size, sources))

    with atomic_output_dir(output) as partial:
        files: list[Path] = []
        for name, fields, rows in (
            ("quality_sweep.csv", QUALITY_SWEEP_FIELDS, quality_sweep),
            ("operating_points.csv", OPERATING_FIELDS, operating),
            ("timing_raw.csv", TIMING_RAW_FIELDS, timing_raw),
            ("timing_summary.csv", TIMING_SUMMARY_FIELDS, timing_summary),
            ("batch_scaling.csv", BATCH_SCALING_FIELDS, batch_scaling),
            ("batch_parity.csv", PARITY_FIELDS, parity),
        ):
            path = partial / name
            _write_csv(path, fields, rows)
            files.append(path)
        summary_path = partial / "comparison_summary.json"
        _write_json(summary_path, {
            "schema_version": 1,
            "methods": list(METHODS),
            "batch_sizes": list(batch_sizes),
            "final_batch_size": final_batch_size,
            "quality_operating_points": operating,
            "timing_summary": timing_summary,
            "batch_scaling": batch_scaling,
            "parity": parity,
        })
        files.append(summary_path)
        readme_path = partial / "README.md"
        readme_path.write_text(README, encoding="utf-8", newline="\n")
        files.append(readme_path)
        manifest_path = partial / "manifest.json"
        _write_json(manifest_path, {
            "schema_version": 1,
            "stage": "comparison-data-export",
            "run_root": str(run_root),
            "audit": {
                "quality_method_count": len(METHODS),
                "operating_point_count": len(operating),
                "timing_record_count": len(timing_raw),
                "batch_scaling_row_count": len(batch_scaling),
                "parity_all_passed": all(row["valid"] for row in parity),
            },
            "sources": [{
                "path": str(path.relative_to(run_root)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            } for path in sorted(sources)],
            "outputs": [file_entry(path) for path in files],
        })
        files.append(manifest_path)
        _write_json(partial / "complete.json", {
            "schema_version": 1,
            "stage": "comparison-data-export",
            "outputs": [file_entry(path) for path in files],
        })
    print(json.dumps({
        "status": "complete",
        "out": str(output.resolve()),
        "quality_sweep_rows": len(quality_sweep),
        "operating_point_rows": len(operating),
        "timing_rows": len(timing_raw),
        "batch_scaling_rows": len(batch_scaling),
        "parity_rows": len(parity),
    }, sort_keys=True))


def _batch_sizes(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("batch sizes must be CSV integers") from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and export tidy six-method comparison data")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-sizes", type=_batch_sizes,
                        default=(1, 8, 32, 128, 600))
    parser.add_argument("--final-batch-size", type=int, default=128)
    args = parser.parse_args(argv)
    export(args.run_root.resolve(), args.out.resolve(),
           batch_sizes=args.batch_sizes,
           final_batch_size=args.final_batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
