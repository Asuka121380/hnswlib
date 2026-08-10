#!/usr/bin/env python3
"""Validate a Stage 1 EQ-RCP operational dataset independently of Parquet bytes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from operational_dataset import (
    EDGE_SCHEMA,
    EVENT_SCHEMA,
    FORMAT_NAME,
    FORMAT_VERSION,
    QUERY_SCHEMA,
    Fvecs,
    canonical_table_sha256,
    dataset_semantic_sha256_from_parts,
    read_json,
    sha256_file,
    write_json,
)


def _column(table: pa.Table, name: str, dtype: Any | None = None) -> np.ndarray:
    values = np.asarray(table[name].combine_chunks().to_numpy(zero_copy_only=False))
    return values.astype(dtype, copy=False) if dtype is not None else values


def _load_table(dataset_dir: Path, name: str, schema: pa.Schema) -> pa.Table:
    path = dataset_dir / f"{name}.parquet"
    table = pq.read_table(path)
    if table.schema != schema:
        raise ValueError(f"{name} schema mismatch: {table.schema} != {schema}")
    return table


def _geometry_indices(event_count: int, limit: int) -> np.ndarray:
    if limit <= 0 or limit >= event_count:
        return np.arange(event_count, dtype=np.int64)
    # Evenly spaced selection is deterministic and covers the whole trace order.
    return np.unique(np.linspace(0, event_count - 1, limit, dtype=np.int64))


def validate_dataset(
    dataset_dir: Path, geometry_limit: int = 4096
) -> dict[str, Any]:
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("format") != FORMAT_NAME or manifest.get("version") != FORMAT_VERSION:
        raise ValueError("unsupported dataset manifest format/version")

    queries = _load_table(dataset_dir, "queries", QUERY_SCHEMA)
    edges = _load_table(dataset_dir, "edges", EDGE_SCHEMA)
    events = _load_table(dataset_dir, "events", EVENT_SCHEMA)
    tables = {"queries": queries, "edges": edges, "events": events}
    schemas = {"queries": QUERY_SCHEMA, "edges": EDGE_SCHEMA, "events": EVENT_SCHEMA}
    semantic_hashes: dict[str, str] = {}
    for name, table in tables.items():
        expected = manifest["outputs"][name]
        if table.num_rows != expected["rows"]:
            raise ValueError(f"{name} row count mismatch")
        semantic = canonical_table_sha256(table, schemas[name])
        if semantic != expected["semantic_sha256"]:
            raise ValueError(f"{name} semantic SHA-256 mismatch")
        file_path = dataset_dir / expected["relative_path"]
        file_hash = sha256_file(file_path)
        if file_hash != expected["file_sha256"]:
            raise ValueError(f"{name} Parquet file SHA-256 mismatch")
        semantic_hashes[name] = semantic

    query_ids = _column(queries, "query_id", np.int64)
    event_query = _column(events, "query_id", np.int64)
    event_id = _column(events, "event_id", np.int64)
    edge_id = _column(edges, "edge_id", np.int64)
    event_edge = _column(events, "edge_id", np.int64)
    if not np.array_equal(event_id, np.arange(len(event_id), dtype=np.int64)):
        raise ValueError("event ids are not contiguous trace-order ids")
    if not np.array_equal(edge_id, np.arange(len(edge_id), dtype=np.int64)):
        raise ValueError("edge ids are not contiguous canonical ids")
    if len(np.unique(query_ids)) != len(query_ids):
        raise ValueError("query table contains duplicate query ids")
    if not np.all(np.isin(event_query, query_ids)):
        raise ValueError("event references a query outside query table")
    if event_edge.min() < 0 or event_edge.max() >= len(edge_id):
        raise ValueError("event references an edge outside edge table")

    source = _column(events, "source_internal_id", np.int64)
    target = _column(events, "target_internal_id", np.int64)
    layer = _column(events, "graph_layer", np.int32)
    edge_source = _column(edges, "source_internal_id", np.int64)
    edge_target = _column(edges, "target_internal_id", np.int64)
    edge_layer = _column(edges, "graph_layer", np.int32)
    if not np.array_equal(source, edge_source[event_edge]):
        raise ValueError("event/edge source mismatch")
    if not np.array_equal(target, edge_target[event_edge]):
        raise ValueError("event/edge target mismatch")
    if not np.array_equal(layer, edge_layer[event_edge]):
        raise ValueError("event/edge layer mismatch")
    event_edge_squared = _column(events, "edge_squared_length", np.float64)
    edge_squared = _column(edges, "edge_squared_length", np.float64)
    tolerance = float(manifest["geometry_audit"]["distance_abs_tolerance"])
    if not np.allclose(event_edge_squared, edge_squared[event_edge], rtol=0.0, atol=tolerance):
        raise ValueError("event/edge length mismatch")

    unique_event_query, event_counts = np.unique(event_query, return_counts=True)
    query_event_counts = _column(queries, "event_count", np.int64)
    query_count_map = dict(zip(unique_event_query.tolist(), event_counts.tolist()))
    if any(query_count_map.get(int(qid), 0) != int(count) for qid, count in zip(query_ids, query_event_counts)):
        raise ValueError("query event counts are inconsistent")
    per_query_weight = _column(events, "per_query_weight", np.float64)
    weighted = np.bincount(
        np.searchsorted(query_ids, event_query), weights=per_query_weight, minlength=len(query_ids)
    )
    if not np.allclose(weighted, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("per-query weights do not sum to one")

    split_manifest = read_json(dataset_dir / manifest["split_manifest"])
    splits = queries["split"].to_pylist()
    split_counts = {name: splits.count(name) for name in sorted(set(splits))}
    if split_counts != split_manifest["query_counts"]:
        raise ValueError("split query counts differ from split manifest")
    if not all(str(name).startswith("development_") for name in splits) and manifest["development_only"]:
        raise ValueError("development dataset uses a non-development split name")

    threshold = _column(events, "threshold", np.float64)
    candidate = _column(events, "candidate_squared_distance", np.float64)
    good = _column(events, "good_candidate", np.bool_)
    prunable = _column(events, "oracle_prunable", np.bool_)
    margin = _column(events, "margin", np.float64)
    if not np.array_equal(good, candidate <= threshold):
        raise ValueError("good-candidate labels are inconsistent")
    if not np.array_equal(prunable, candidate > threshold):
        raise ValueError("oracle-prunable labels are inconsistent")
    if not np.allclose(margin, candidate - threshold, rtol=0.0, atol=0.0):
        raise ValueError("margin values are inconsistent")

    inputs = manifest["inputs"]
    for name, artifact in inputs.items():
        path = Path(artifact["physical_path"])
        if not path.is_file():
            raise FileNotFoundError(f"input artifact moved or unavailable: {name}: {path}")
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"input artifact SHA-256 changed: {name}")
    dataset_hash = dataset_semantic_sha256_from_parts(
        manifest["semantic_config"],
        semantic_hashes,
        {name: artifact["sha256"] for name, artifact in inputs.items()},
    )
    if dataset_hash != manifest["dataset_semantic_sha256"]:
        raise ValueError("dataset semantic SHA-256 mismatch")

    indices = _geometry_indices(events.num_rows, geometry_limit)
    base = Fvecs(Path(inputs["base_vectors"]["physical_path"]))
    query = Fvecs(Path(inputs["query_vectors"]["physical_path"]))
    mapping = np.load(
        Path(inputs["internal_to_label"]["physical_path"]), mmap_mode="r", allow_pickle=False
    )
    query_mapping = manifest["semantic_config"]["query_mapping"]
    query_rows = event_query[indices] - int(query_mapping.get("offset", 0))
    c = base.take(mapping[source[indices]]).astype(np.float64)
    v = base.take(mapping[target[indices]]).astype(np.float64)
    q = query.take(query_rows).astype(np.float64)
    x = q - c
    delta = v - c
    edge_calc = np.einsum("ij,ij->i", delta, delta)
    x_calc = np.einsum("ij,ij->i", x, x)
    y_calc = np.einsum("ij,ij->i", x, delta)
    qv = q - v
    candidate_calc = np.einsum("ij,ij->i", qv, qv)
    current_trace = _column(events, "current_squared_distance", np.float64)[indices]
    x_stored = _column(events, "x_squared_norm", np.float64)[indices]
    y_stored = _column(events, "y_exact", np.float64)[indices]
    source_label = _column(events, "source_external_label", np.int64)[indices]
    target_label = _column(events, "target_external_label", np.int64)[indices]
    geometry_errors = {
        "edge_squared_length": float(np.max(np.abs(edge_calc - event_edge_squared[indices]))),
        "x_squared_norm": float(np.max(np.abs(x_calc - x_stored))),
        "y_exact": float(np.max(np.abs(y_calc - y_stored))),
        "current_squared_distance": float(np.max(np.abs(x_calc - current_trace))),
        "candidate_squared_distance": float(np.max(np.abs(candidate_calc - candidate[indices]))),
        "identity": float(np.max(np.abs((x_calc + edge_calc - 2.0 * y_calc) - candidate_calc))),
    }
    if max(geometry_errors.values()) > tolerance:
        raise ValueError(f"deep geometry validation exceeded tolerance: {geometry_errors}")
    if not np.array_equal(source_label, np.asarray(mapping[source[indices]], dtype=np.int64)):
        raise ValueError("source external labels differ from frozen mapping")
    if not np.array_equal(target_label, np.asarray(mapping[target[indices]], dtype=np.int64)):
        raise ValueError("target external labels differ from frozen mapping")
    base.close()
    query.close()
    mapping_mmap = getattr(mapping, "_mmap", None)
    if mapping_mmap is not None:
        mapping_mmap.close()

    adjacency = manifest["adjacency_validation"]
    if manifest["dataset_role"] == "formal" and adjacency["status"] != "PASS":
        raise ValueError("formal dataset lacks frozen-index adjacency validation")
    report = {
        "status": "PASS",
        "dataset_semantic_sha256": manifest["dataset_semantic_sha256"],
        "table_semantic_sha256": semantic_hashes,
        "rows": {name: table.num_rows for name, table in tables.items()},
        "geometry_records_checked": int(len(indices)),
        "geometry_max_abs_error": geometry_errors,
        "adjacency_validation": adjacency,
        "development_materialization_status": (
            "PASS" if manifest["dataset_role"] == "development" else "NOT_APPLICABLE"
        ),
        "formal_materialization_status": manifest["formal_materialization_status"],
    }
    write_json(dataset_dir / "validation_report.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--geometry-record-limit",
        type=int,
        default=4096,
        help="0 validates all events; positive values use a deterministic sample",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = validate_dataset(args.dataset_dir, args.geometry_record_limit)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
