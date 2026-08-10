#!/usr/bin/env python3
"""Build the immutable Stage 1 EQ-RCP operational dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from operational_dataset import (
    EDGE_SCHEMA,
    EVENT_SCHEMA,
    FORMAT_NAME,
    FORMAT_VERSION,
    QUERY_SCHEMA,
    Fvecs,
    NativeHnswLevel0,
    assign_query_splits,
    canonical_json_bytes,
    canonical_table_sha256,
    dataset_semantic_sha256,
    read_json,
    schema_document,
    semantic_config,
    sha256_bytes,
    sha256_file,
    table_from_arrays,
    write_json,
    write_parquet,
)


REQUIRED_RECORD_COLUMNS = [
    "query_id",
    "current_node_id",
    "candidate_id",
    "graph_layer",
    "ef_search",
    "current_squared_distance",
    "threshold",
    "shadow_exact_squared_distance",
]


def _path(config: dict[str, Any], key: str) -> Path:
    path = Path(config["paths"][key]).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"missing {key}: {path}")
    return path


def _verify_expected_hash(
    key: str, path: Path, config: dict[str, Any], actual: str
) -> None:
    expected = config.get("expected_sha256", {}).get(key)
    if expected and actual.lower() != str(expected).lower():
        raise ValueError(f"{key} SHA-256 mismatch: {actual} != {expected} ({path})")


def _prepare_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.iterdir())
    if existing:
        names = ", ".join(sorted(item.name for item in existing)[:5])
        raise FileExistsError(f"output directory must be empty: {output_dir} ({names})")


def _compute_geometry(
    base: Fvecs,
    query: Fvecs,
    mapping: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    query_rows: np.ndarray,
    batch_size: int,
) -> dict[str, np.ndarray]:
    count = len(source)
    edge_squared = np.empty(count, dtype=np.float64)
    x_squared = np.empty(count, dtype=np.float64)
    y_exact = np.empty(count, dtype=np.float64)
    current_calc = np.empty(count, dtype=np.float64)
    candidate_calc = np.empty(count, dtype=np.float64)
    for begin in range(0, count, batch_size):
        end = min(begin + batch_size, count)
        c = base.take(mapping[source[begin:end]]).astype(np.float64)
        v = base.take(mapping[target[begin:end]]).astype(np.float64)
        q = query.take(query_rows[begin:end]).astype(np.float64)
        x = q - c
        delta = v - c
        edge_squared[begin:end] = np.einsum("ij,ij->i", delta, delta)
        x_squared[begin:end] = np.einsum("ij,ij->i", x, x)
        y_exact[begin:end] = np.einsum("ij,ij->i", x, delta)
        current_calc[begin:end] = x_squared[begin:end]
        qv = q - v
        candidate_calc[begin:end] = np.einsum("ij,ij->i", qv, qv)
    return {
        "edge_squared": edge_squared,
        "x_squared": x_squared,
        "y_exact": y_exact,
        "current_calc": current_calc,
        "candidate_calc": candidate_calc,
    }


def build_dataset(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("format") != FORMAT_NAME or config.get("version") != FORMAT_VERSION:
        raise ValueError("unsupported operational dataset config format/version")
    if config.get("dataset_role") not in {"development", "formal"}:
        raise ValueError("dataset_role must be development or formal")
    _prepare_output(output_dir)

    records_path = _path(config, "records")
    base_path = _path(config, "base_vectors")
    query_path = _path(config, "query_vectors")
    mapping_path = _path(config, "internal_to_label")
    mapping_manifest_path = _path(config, "mapping_manifest")
    index_value = config["paths"].get("index")
    index_path = Path(index_value).expanduser() if index_value else None
    if index_path is not None and not index_path.is_file():
        raise FileNotFoundError(f"configured index does not exist: {index_path}")

    input_paths = {
        "records": records_path,
        "base_vectors": base_path,
        "query_vectors": query_path,
        "internal_to_label": mapping_path,
        "mapping_manifest": mapping_manifest_path,
    }
    if index_path is not None:
        input_paths["index"] = index_path
    input_artifacts: dict[str, Any] = {}
    for key, path in input_paths.items():
        digest = sha256_file(path)
        _verify_expected_hash(key, path, config, digest)
        input_artifacts[key] = {
            "physical_path": str(path.resolve()),
            "byte_size": path.stat().st_size,
            "sha256": digest,
            "logical_id": config["logical_artifacts"][key],
        }

    mapping_manifest = read_json(mapping_manifest_path)
    expected_index_hash = config.get("expected_sha256", {}).get("index")
    mapped_index_hash = mapping_manifest.get("index_sha256")
    if expected_index_hash and mapped_index_hash != expected_index_hash:
        raise ValueError(
            "mapping manifest belongs to a different index: "
            f"{mapped_index_hash} != {expected_index_hash}"
        )

    base = Fvecs(base_path)
    queries = Fvecs(query_path)
    if base.dim != queries.dim:
        raise ValueError(f"base/query dimension mismatch: {base.dim} != {queries.dim}")
    mapping = np.load(mapping_path, mmap_mode="r", allow_pickle=False)
    mapping_count = mapping_manifest.get(
        "current_count", mapping_manifest.get("count")
    )
    if mapping.ndim != 1 or len(mapping) != mapping_count:
        raise ValueError("mapping shape does not match mapping manifest")
    if len(mapping) and (mapping.min() < 0 or mapping.max() >= base.count):
        raise ValueError("mapping contains labels outside the base-vector range")

    record_table = pq.read_table(records_path, columns=REQUIRED_RECORD_COLUMNS)
    record_count = record_table.num_rows
    expected_rows = config.get("validation", {}).get("expected_rows")
    if expected_rows is not None and record_count != int(expected_rows):
        raise ValueError(f"record count mismatch: {record_count} != {expected_rows}")
    arrays = {
        name: np.asarray(record_table[name].combine_chunks().to_numpy(zero_copy_only=False))
        for name in REQUIRED_RECORD_COLUMNS
    }
    query_id = arrays["query_id"].astype(np.int64, copy=False)
    source = arrays["current_node_id"].astype(np.int64, copy=False)
    target = arrays["candidate_id"].astype(np.int64, copy=False)
    layer = arrays["graph_layer"].astype(np.int32, copy=False)
    ef_search = arrays["ef_search"].astype(np.int32, copy=False)
    current_trace = arrays["current_squared_distance"].astype(np.float64)
    threshold = arrays["threshold"].astype(np.float64)
    candidate_trace = arrays["shadow_exact_squared_distance"].astype(np.float64)
    if record_count == 0:
        raise ValueError("source trace contains no records")
    if source.min() < 0 or target.min() < 0 or source.max() >= len(mapping) or target.max() >= len(mapping):
        raise ValueError("trace internal node id is outside mapping range")
    if np.any(layer != 0):
        raise ValueError("Stage 1 currently requires level-0 operational events")

    query_mapping = config["query_mapping"]
    if query_mapping.get("mode") != "query_id_is_row":
        raise ValueError("only query_id_is_row query mapping is supported")
    query_rows = query_id - int(query_mapping.get("offset", 0))
    if query_rows.min() < 0 or query_rows.max() >= queries.count:
        raise ValueError("query id maps outside the query-vector file")

    batch_size = int(config.get("build", {}).get("batch_size", 4096))
    geometry = _compute_geometry(
        base, queries, mapping, source, target, query_rows, batch_size
    )
    current_error = np.abs(geometry["current_calc"] - current_trace)
    candidate_error = np.abs(geometry["candidate_calc"] - candidate_trace)
    identity_candidate = (
        geometry["current_calc"] + geometry["edge_squared"] - 2.0 * geometry["y_exact"]
    )
    identity_error = np.abs(identity_candidate - geometry["candidate_calc"])
    tolerance = float(config.get("validation", {}).get("distance_abs_tolerance", 1e-5))
    audit = {
        "distance_abs_tolerance": tolerance,
        "max_current_distance_abs_error": float(current_error.max()),
        "max_candidate_distance_abs_error": float(candidate_error.max()),
        "max_identity_abs_error": float(identity_error.max()),
    }
    if max(audit[key] for key in audit if key.startswith("max_")) > tolerance:
        raise ValueError(f"geometry validation exceeded tolerance: {audit}")

    edge_keys = np.empty(
        record_count,
        dtype=[("source", "<i8"), ("target", "<i8"), ("layer", "<i4")],
    )
    edge_keys["source"] = source
    edge_keys["target"] = target
    edge_keys["layer"] = layer
    unique_edges, first, inverse, visit_count = np.unique(
        edge_keys, return_index=True, return_inverse=True, return_counts=True
    )
    edge_source = unique_edges["source"].astype(np.int64)
    edge_target = unique_edges["target"].astype(np.int64)
    edge_layer = unique_edges["layer"].astype(np.int32)
    edge_squared = geometry["edge_squared"][first]
    if not np.allclose(
        edge_squared[inverse], geometry["edge_squared"], rtol=0.0, atol=tolerance
    ):
        raise ValueError("same directed edge produced inconsistent geometry")

    neighbor_slot = np.full(len(unique_edges), -1, dtype=np.int32)
    adjacency_status = "PENDING_INDEX_UNAVAILABLE"
    if index_path is not None:
        index = NativeHnswLevel0(index_path)
        if index.header.current_count != len(mapping):
            raise ValueError("index count differs from internal-to-label mapping")
        neighbor_slot = index.neighbor_slots(edge_source, edge_target)
        index.close()
        missing_count = int(np.count_nonzero(neighbor_slot < 0))
        if missing_count:
            raise ValueError(f"{missing_count} traced edges are absent from frozen index")
        adjacency_status = "PASS"
    require_adjacency = bool(
        config.get("validation", {}).get("require_index_adjacency_validation", False)
    )
    if require_adjacency and adjacency_status != "PASS":
        raise ValueError("formal adjacency validation is required but index is unavailable")

    source_labels = np.asarray(mapping[source], dtype=np.int64)
    target_labels = np.asarray(mapping[target], dtype=np.int64)
    threshold_valid = np.isfinite(threshold) & (threshold >= 0.0)
    if not np.all(threshold_valid):
        raise ValueError("trace contains invalid pruning thresholds")
    good_candidate = candidate_trace <= threshold
    oracle_prunable = candidate_trace > threshold
    margin = candidate_trace - threshold
    unique_query, query_inverse, query_event_count = np.unique(
        query_id, return_inverse=True, return_counts=True
    )
    split_map = assign_query_splits(
        unique_query,
        int(config["split"]["seed"]),
        config["split"]["fractions"],
    )
    split_names = [split_map[int(value)] for value in unique_query]
    per_query_weight = 1.0 / query_event_count[query_inverse].astype(np.float64)

    query_table = table_from_arrays(
        QUERY_SCHEMA,
        {
            "query_id": unique_query,
            "split": split_names,
            "dataset_role": [config["dataset_role"]] * len(unique_query),
            "event_count": query_event_count,
        },
    )
    edge_table = table_from_arrays(
        EDGE_SCHEMA,
        {
            "edge_id": np.arange(len(unique_edges), dtype=np.int64),
            "source_internal_id": edge_source,
            "target_internal_id": edge_target,
            "source_external_label": np.asarray(mapping[edge_source], dtype=np.int64),
            "target_external_label": np.asarray(mapping[edge_target], dtype=np.int64),
            "graph_layer": edge_layer,
            "neighbor_slot": neighbor_slot,
            "edge_squared_length": edge_squared,
            "edge_length": np.sqrt(edge_squared),
            "visit_count": visit_count.astype(np.int64),
        },
    )
    event_table = table_from_arrays(
        EVENT_SCHEMA,
        {
            "event_id": np.arange(record_count, dtype=np.int64),
            "query_id": query_id,
            "edge_id": inverse.astype(np.int64),
            "source_internal_id": source,
            "target_internal_id": target,
            "source_external_label": source_labels,
            "target_external_label": target_labels,
            "graph_layer": layer,
            "ef_search": ef_search,
            "current_squared_distance": current_trace,
            "candidate_squared_distance": candidate_trace,
            "threshold": threshold,
            "edge_squared_length": geometry["edge_squared"],
            "x_squared_norm": geometry["x_squared"],
            "y_exact": geometry["y_exact"],
            "margin": margin,
            "threshold_valid": threshold_valid,
            "good_candidate": good_candidate,
            "oracle_prunable": oracle_prunable,
            "raw_event_weight": np.ones(record_count, dtype=np.float64),
            "per_query_weight": per_query_weight,
        },
    )

    table_hashes = {
        "queries": canonical_table_sha256(query_table, QUERY_SCHEMA),
        "edges": canonical_table_sha256(edge_table, EDGE_SCHEMA),
        "events": canonical_table_sha256(event_table, EVENT_SCHEMA),
    }
    outputs = {
        "queries": write_parquet(query_table, output_dir / "queries.parquet"),
        "edges": write_parquet(edge_table, output_dir / "edges.parquet"),
        "events": write_parquet(event_table, output_dir / "events.parquet"),
    }
    for key in outputs:
        outputs[key]["semantic_sha256"] = table_hashes[key]

    split_counts = {
        name: int(split_names.count(name)) for name in sorted(set(split_names))
    }
    split_event_counts = {
        name: int(query_event_count[np.asarray(split_names) == name].sum())
        for name in sorted(set(split_names))
    }
    split_manifest = {
        "algorithm": "sha256(seed:query_id)-uint64le-v1",
        "seed": int(config["split"]["seed"]),
        "fractions": config["split"]["fractions"],
        "query_counts": split_counts,
        "event_counts": split_event_counts,
        "assignment_sha256": sha256_bytes(
            canonical_json_bytes(
                [[int(qid), split_map[int(qid)]] for qid in unique_query]
            )
        ),
    }
    write_json(output_dir / "split_manifest.json", split_manifest)
    write_json(output_dir / "schema.json", schema_document())

    oracle_fraction = float(np.mean(oracle_prunable))
    expected_oracle = config.get("validation", {}).get("expected_oracle_prunable_fraction")
    oracle_tolerance = float(config.get("validation", {}).get("oracle_fraction_tolerance", 1e-12))
    if expected_oracle is not None and abs(oracle_fraction - float(expected_oracle)) > oracle_tolerance:
        raise ValueError(
            f"oracle-prunable fraction mismatch: {oracle_fraction} != {expected_oracle}"
        )
    formal_status = "PASS" if config["dataset_role"] == "formal" and adjacency_status == "PASS" else "PENDING"
    manifest = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "dataset_id": config["dataset_id"],
        "dataset_role": config["dataset_role"],
        "development_only": config["dataset_role"] != "formal",
        "formal_materialization_status": formal_status,
        "semantic_config": semantic_config(config),
        "semantic_config_sha256": sha256_bytes(canonical_json_bytes(semantic_config(config))),
        "dataset_semantic_sha256": dataset_semantic_sha256(
            config,
            table_hashes,
            {key: value["sha256"] for key, value in input_artifacts.items()},
        ),
        "inputs": input_artifacts,
        "outputs": outputs,
        "counts": {
            "queries": query_table.num_rows,
            "directed_edges": edge_table.num_rows,
            "events": event_table.num_rows,
            "oracle_prunable": int(np.count_nonzero(oracle_prunable)),
            "good_candidate": int(np.count_nonzero(good_candidate)),
        },
        "statistics": {"oracle_prunable_fraction": oracle_fraction},
        "geometry_audit": audit,
        "adjacency_validation": {
            "status": adjacency_status,
            "required": require_adjacency,
            "checked_edges": int(len(unique_edges)) if adjacency_status == "PASS" else 0,
            "edge_id_scope": (
                "frozen_index_neighbor_slot"
                if adjacency_status == "PASS"
                else "development_trace_directed_pair"
            ),
        },
        "split_manifest": "split_manifest.json",
        "schema": "schema.json",
        "limitations": [
            "Delta vectors are reconstructed from frozen base vectors and are not duplicated.",
            "Source trace lacks queue sizes, expansion index, and search progress.",
        ],
    }
    if config["dataset_role"] != "formal":
        manifest["limitations"].append(
            "Development materialization is not formal pruning evidence and its edge ids are index-specific."
        )
    write_json(output_dir / "dataset_manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_dataset(args.config, args.output_dir)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": "PASS",
        "dataset_semantic_sha256": manifest["dataset_semantic_sha256"],
        "formal_materialization_status": manifest["formal_materialization_status"],
        "output_dir": str(args.output_dir.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
