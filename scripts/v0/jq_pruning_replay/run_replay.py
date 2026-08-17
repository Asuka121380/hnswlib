#!/usr/bin/env python3
"""Run one offline counterfactual JQ pruning replay."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    from scripts.v0.jq_pruning_replay.jq_quantizer import JQConfig, JQQuantizer
    from scripts.v0.jq_pruning_replay.trace_inputs import (
        FvecsMemmap,
        load_internal_to_label,
        load_shadow_records,
        sha256_file,
        validate_frozen_shadow_contract,
    )
    from scripts.v0.jq_pruning_replay.v0_bound_replay import (
        evaluate_v0_bound,
        wilson_interval,
    )
else:
    from .jq_quantizer import JQConfig, JQQuantizer
    from .trace_inputs import (
        FvecsMemmap,
        load_internal_to_label,
        load_shadow_records,
        sha256_file,
        validate_frozen_shadow_contract,
    )
    from .v0_bound_replay import evaluate_v0_bound, wilson_interval


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    partial = Path(str(path) + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        partial.replace(path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _distance_squared_float32(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    difference = np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)
    # Match hnswlib's operational float32 accumulation closely enough for the
    # frozen trace parity gate.  Exact equality is not assumed across SIMD paths.
    return np.sum(difference * difference, axis=1, dtype=np.float32).astype(np.float64)


def _validate_dataset_contract(
    dataset_dir: Path,
    base_path: Path,
    query_path: Path,
    base: FvecsMemmap,
    queries: FvecsMemmap,
    *,
    verify_large_hashes: bool,
) -> dict[str, Any]:
    dataset_path = dataset_dir / "dataset.json"
    contract = json.loads(dataset_path.read_text(encoding="utf-8"))
    expected = {
        "dataset": "gist1m",
        "dimension": base.dimension,
        "n_base": base.count,
        "n_query": queries.count,
    }
    failures = [
        f"dataset contract {key}={contract.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if contract.get(key) != value
    ]
    declared_sizes = contract.get("file_sizes", {})
    if declared_sizes.get("base_bytes") != base_path.stat().st_size:
        failures.append("base file size disagrees with dataset.json")
    if declared_sizes.get("query_bytes") != query_path.stat().st_size:
        failures.append("query file size disagrees with dataset.json")
    verified_hashes: dict[str, str] = {}
    if verify_large_hashes:
        declared_hashes = contract.get("checksums", {})
        for name, path in (("base", base_path), ("query", query_path)):
            actual = sha256_file(path)
            verified_hashes[name] = actual
            if actual != declared_hashes.get(f"{name}_sha256"):
                failures.append(f"{name} SHA-256 disagrees with dataset.json")
    if failures:
        raise ValueError("dataset contract failed: " + "; ".join(failures))
    return {
        "dataset_json": str(dataset_path),
        "dataset_json_sha256": sha256_file(dataset_path),
        "declared_checksums": contract.get("checksums", {}),
        "verified_large_hashes": verified_hashes,
    }


def _build_unique_edges(current: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.column_stack((current, candidate)).astype(np.uint64, copy=False)
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    return unique, inverse.astype(np.int64, copy=False)


def _make_edge_directions(
    base: FvecsMemmap,
    mapping: np.ndarray,
    unique_edges: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = unique_edges.shape[0]
    directions = np.empty((count, base.dimension), dtype=np.float32)
    lengths = np.empty(count, dtype=np.float64)
    zero = np.zeros(count, dtype=np.bool_)
    for start in range(0, count, batch_size):
        end = min(count, start + batch_size)
        current_labels = np.asarray(mapping[unique_edges[start:end, 0]], dtype=np.int64)
        candidate_labels = np.asarray(mapping[unique_edges[start:end, 1]], dtype=np.int64)
        source = base.take(current_labels)
        target = base.take(candidate_labels)
        difference = target.astype(np.float64) - source.astype(np.float64)
        squared = np.sum(difference * difference, axis=1, dtype=np.float64)
        edge_length = np.sqrt(squared)
        is_zero = squared == 0.0
        safe_length = np.where(is_zero, 1.0, edge_length)
        directions[start:end] = (difference / safe_length[:, np.newaxis]).astype(np.float32)
        directions[start:end][is_zero] = 0.0
        lengths[start:end] = edge_length
        zero[start:end] = is_zero
    return directions, lengths, zero


def _validate_trace_distances(
    records,
    base: FvecsMemmap,
    queries: FvecsMemmap,
    mapping: np.ndarray,
    *,
    sample_count: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    count = min(records.count, sample_count)
    indices = np.linspace(0, records.count - 1, count, dtype=np.int64)
    query = queries.take(records.query_id[indices].astype(np.int64))
    current = base.take(
        np.asarray(mapping[records.current_node_id[indices]], dtype=np.int64)
    )
    candidate = base.take(
        np.asarray(mapping[records.candidate_id[indices]], dtype=np.int64)
    )
    current_replayed = _distance_squared_float32(query, current)
    candidate_replayed = _distance_squared_float32(query, candidate)
    current_ok = np.isclose(
        current_replayed,
        records.current_squared_distance[indices],
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    candidate_ok = np.isclose(
        candidate_replayed,
        records.shadow_exact_squared_distance[indices],
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    return {
        "sample_count": int(count),
        "current_distance_failures": int(np.count_nonzero(~current_ok)),
        "candidate_distance_failures": int(np.count_nonzero(~candidate_ok)),
        "max_current_absolute_error": float(
            np.max(np.abs(current_replayed - records.current_squared_distance[indices]))
        ),
        "max_candidate_absolute_error": float(
            np.max(
                np.abs(
                    candidate_replayed
                    - records.shadow_exact_squared_distance[indices]
                )
            )
        ),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    stage_dir = args.frozen_stage_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    shadow_path = stage_dir / "shadow_records.csv.gz"
    metadata_path = stage_dir / "metadata.json"
    base_path = dataset_dir / "gist_base.fvecs"
    query_path = dataset_dir / "gist_query.fvecs"
    mapping_path = dataset_dir / "gist1m_internal_to_label.npy"

    final_dir = args.output_dir.resolve()
    partial_dir = Path(str(final_dir) + ".partial")
    if final_dir.exists() or partial_dir.exists():
        raise FileExistsError(f"refusing to overwrite replay output: {final_dir}")
    partial_dir.mkdir(parents=True)
    success = False
    try:
        records = load_shadow_records(shadow_path, max_rows=args.max_rows)
        frozen_contract = None
        if args.enforce_frozen_contract:
            if args.max_rows is not None:
                raise ValueError("--enforce-frozen-contract cannot be combined with --max-rows")
            frozen_contract = validate_frozen_shadow_contract(records, metadata_path)

        base = FvecsMemmap(base_path, expected_dimension=args.dimension)
        queries = FvecsMemmap(query_path, expected_dimension=args.dimension)
        dataset_contract = _validate_dataset_contract(
            dataset_dir,
            base_path,
            query_path,
            base,
            queries,
            verify_large_hashes=args.verify_large_input_sha256,
        )
        mapping = load_internal_to_label(mapping_path, expected_count=base.count)
        if records.query_id.max() >= queries.count:
            raise IndexError("trace query_id exceeds query dataset")
        if max(records.current_node_id.max(), records.candidate_id.max()) >= mapping.size:
            raise IndexError("trace internal node ID exceeds mapping")

        distance_parity = _validate_trace_distances(
            records,
            base,
            queries,
            mapping,
            sample_count=args.parity_sample_count,
            absolute_tolerance=args.distance_atol,
            relative_tolerance=args.distance_rtol,
        )
        if (
            distance_parity["current_distance_failures"]
            or distance_parity["candidate_distance_failures"]
        ):
            raise RuntimeError(f"trace distance parity failed: {distance_parity}")

        unique_edges, event_edge_index = _build_unique_edges(
            records.current_node_id, records.candidate_id
        )
        directions, edge_lengths, zero_edges = _make_edge_directions(
            base,
            mapping,
            unique_edges,
            batch_size=args.vector_batch_size,
        )

        config = JQConfig(
            dimension=args.dimension,
            subspaces=args.subspaces,
            bits=8,
            rotation_seed=args.rotation_seed,
        )
        quantizer = JQQuantizer(config)
        rotated_directions = quantizer.rotate(
            directions, batch_size=args.vector_batch_size
        )
        training_mask = ~zero_edges
        if np.count_nonzero(training_mask) < 2:
            raise RuntimeError("not enough non-zero unique edges to fit JQ")
        quantizer.fit_rotated(rotated_directions[training_mask])
        codes = quantizer.encode_rotated(
            rotated_directions, batch_size=args.quantizer_batch_size
        )
        direction_error = quantizer.direction_error_upper(
            rotated_directions,
            codes,
            batch_size=args.vector_batch_size,
        )
        direction_error[zero_edges] = np.inf

        artifact_manifest = quantizer.save_artifacts(partial_dir / "jq_artifact")
        valid_count = prune_count = violation_count = invalid_count = 0
        rows_written = 0
        event_output = None
        csv_writer = None
        if args.write_events:
            event_output = gzip.open(
                partial_dir / "event_results.csv.gz",
                "wt",
                encoding="utf-8",
                newline="",
            )
            csv_writer = csv.writer(event_output)
            csv_writer.writerow(
                [
                    "query_id",
                    "current_node_id",
                    "candidate_id",
                    "threshold",
                    "edge_length",
                    "direction_error_upper",
                    "residual_direction_inner_product_upper",
                    "lower_bound",
                    "shadow_exact_squared_distance",
                    "valid",
                    "would_prune",
                    "lower_bound_violation",
                ]
            )

        rotation64 = quantizer.rotation.astype(np.float64)
        try:
            for start in range(0, records.count, args.event_batch_size):
                end = min(records.count, start + args.event_batch_size)
                event_slice = slice(start, end)
                query = queries.take(records.query_id[event_slice].astype(np.int64))
                current = base.take(
                    np.asarray(mapping[records.current_node_id[event_slice]], dtype=np.int64)
                )
                residual = query.astype(np.float64) - current.astype(np.float64)
                rotated_residual = residual @ rotation64.T
                edge_index = event_edge_index[event_slice]
                dot_upper = quantizer.selected_inner_product_upper(
                    rotated_residual, codes[edge_index]
                )
                bound = evaluate_v0_bound(
                    records.current_squared_distance[event_slice],
                    edge_lengths[edge_index],
                    dot_upper,
                    direction_error[edge_index],
                    dimension=args.dimension,
                )
                eligible = records.recorded_valid[event_slice] & ~zero_edges[edge_index]
                valid = eligible & bound.valid
                would_prune = valid & (bound.lower_bound > records.threshold[event_slice])
                tolerance = args.bound_safety_atol + args.bound_safety_rtol * np.maximum(
                    1.0, np.abs(records.shadow_exact_squared_distance[event_slice])
                )
                violation = valid & (
                    bound.lower_bound
                    > records.shadow_exact_squared_distance[event_slice] + tolerance
                )
                valid_count += int(np.count_nonzero(valid))
                prune_count += int(np.count_nonzero(would_prune))
                violation_count += int(np.count_nonzero(violation))
                invalid_count += int(np.count_nonzero(eligible & ~bound.valid))

                if csv_writer is not None:
                    for row in range(end - start):
                        csv_writer.writerow(
                            [
                                int(records.query_id[start + row]),
                                int(records.current_node_id[start + row]),
                                int(records.candidate_id[start + row]),
                                repr(float(records.threshold[start + row])),
                                repr(float(edge_lengths[edge_index[row]])),
                                repr(float(direction_error[edge_index[row]])),
                                repr(float(dot_upper[row])),
                                repr(float(bound.lower_bound[row])),
                                repr(float(records.shadow_exact_squared_distance[start + row])),
                                int(valid[row]),
                                int(would_prune[row]),
                                int(violation[row]),
                            ]
                        )
                        rows_written += 1
        finally:
            if event_output is not None:
                event_output.close()

        if violation_count:
            raise RuntimeError(
                f"JQ replay produced {violation_count} lower-bound violations"
            )
        interval = wilson_interval(prune_count, valid_count)
        summary = {
            "format": "hnswlib_jq_pruning_replay_summary",
            "format_version": 1,
            "status": "valid",
            "dimension": args.dimension,
            "subspaces": args.subspaces,
            "bits": 8,
            "code_size_bytes": args.subspaces,
            "rotation_seed": args.rotation_seed,
            "row_count": records.count,
            "unique_edge_count": int(unique_edges.shape[0]),
            "zero_length_unique_edge_count": int(np.count_nonzero(zero_edges)),
            "valid_event_count": valid_count,
            "invalid_bound_event_count": invalid_count,
            "theoretical_prune_count": prune_count,
            "theoretical_prune_rate": prune_count / valid_count if valid_count else 0.0,
            "theoretical_prune_rate_wilson95_low": interval[0],
            "theoretical_prune_rate_wilson95_high": interval[1],
            "lower_bound_violation_count": violation_count,
            "event_rows_written": rows_written,
            "frozen_pq_reference": records.frozen_statistics(),
            "distance_parity": distance_parity,
            "jq_artifact": artifact_manifest,
            "quantizer_fit_source": "all_nonzero_unique_directed_edges_in_frozen_trace_sample",
        }
        manifest = {
            "format": "hnswlib_jq_pruning_replay_manifest",
            "format_version": 1,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "command": sys.argv,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "producer_git_commit": _git_commit(args.repo.resolve()),
            "inputs": {
                "shadow_records": str(shadow_path),
                "shadow_records_sha256": sha256_file(shadow_path),
                "metadata": str(metadata_path),
                "metadata_sha256": sha256_file(metadata_path),
                "base": str(base_path),
                "query": str(query_path),
                "mapping": str(mapping_path),
                "mapping_sha256": sha256_file(mapping_path),
                "dataset_contract": dataset_contract,
            },
            "frozen_contract_enforced": args.enforce_frozen_contract,
            "frozen_contract": frozen_contract,
            "parameters": vars(args) | {"output_dir": str(args.output_dir)},
        }
        # Path values in argparse are not JSON serializable.
        manifest["parameters"] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in manifest["parameters"].items()
        }
        _atomic_json(partial_dir / "summary.json", summary)
        _atomic_json(partial_dir / "run_manifest.json", manifest)
        _atomic_json(
            partial_dir / "complete.json",
            {"status": "complete", "summary": "summary.json"},
        )
        partial_dir.replace(final_dir)
        success = True
        return summary
    finally:
        if not success and partial_dir.exists() and args.clean_failed_partial:
            shutil.rmtree(partial_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--frozen-stage-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dimension", type=int, default=960)
    parser.add_argument("--subspaces", type=int, choices=(32, 120), default=32)
    parser.add_argument("--rotation-seed", type=int, choices=(1234, 17, 42), default=1234)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--enforce-frozen-contract", action="store_true")
    parser.add_argument("--write-events", action="store_true")
    parser.add_argument("--verify-large-input-sha256", action="store_true")
    parser.add_argument("--vector-batch-size", type=int, default=2048)
    parser.add_argument("--quantizer-batch-size", type=int, default=1024)
    parser.add_argument("--event-batch-size", type=int, default=2048)
    parser.add_argument("--parity-sample-count", type=int, default=1000)
    parser.add_argument("--distance-atol", type=float, default=2e-5)
    parser.add_argument("--distance-rtol", type=float, default=2e-5)
    parser.add_argument("--bound-safety-atol", type=float, default=1e-10)
    parser.add_argument("--bound-safety-rtol", type=float, default=1e-10)
    parser.add_argument("--clean-failed-partial", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if min(
        args.vector_batch_size,
        args.quantizer_batch_size,
        args.event_batch_size,
        args.parity_sample_count,
    ) <= 0:
        raise SystemExit("batch sizes and parity sample count must be positive")
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
