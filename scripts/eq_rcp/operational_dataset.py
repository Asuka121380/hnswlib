#!/usr/bin/env python3
"""Shared Stage 1 primitives for the EQ-RCP operational dataset."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


FORMAT_NAME = "eq_rcp_operational_dataset"
FORMAT_VERSION = 1
HNSW_HEADER = struct.Struct("<6QiI3QdQ")


QUERY_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.int64(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("dataset_role", pa.string(), nullable=False),
        pa.field("event_count", pa.int64(), nullable=False),
    ]
)

EDGE_SCHEMA = pa.schema(
    [
        pa.field("edge_id", pa.int64(), nullable=False),
        pa.field("source_internal_id", pa.int64(), nullable=False),
        pa.field("target_internal_id", pa.int64(), nullable=False),
        pa.field("source_external_label", pa.int64(), nullable=False),
        pa.field("target_external_label", pa.int64(), nullable=False),
        pa.field("graph_layer", pa.int32(), nullable=False),
        pa.field("neighbor_slot", pa.int32(), nullable=False),
        pa.field("edge_squared_length", pa.float64(), nullable=False),
        pa.field("edge_length", pa.float64(), nullable=False),
        pa.field("visit_count", pa.int64(), nullable=False),
    ]
)

EVENT_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.int64(), nullable=False),
        pa.field("query_id", pa.int64(), nullable=False),
        pa.field("edge_id", pa.int64(), nullable=False),
        pa.field("source_internal_id", pa.int64(), nullable=False),
        pa.field("target_internal_id", pa.int64(), nullable=False),
        pa.field("source_external_label", pa.int64(), nullable=False),
        pa.field("target_external_label", pa.int64(), nullable=False),
        pa.field("graph_layer", pa.int32(), nullable=False),
        pa.field("ef_search", pa.int32(), nullable=False),
        pa.field("current_squared_distance", pa.float64(), nullable=False),
        pa.field("candidate_squared_distance", pa.float64(), nullable=False),
        pa.field("threshold", pa.float64(), nullable=False),
        pa.field("edge_squared_length", pa.float64(), nullable=False),
        pa.field("x_squared_norm", pa.float64(), nullable=False),
        pa.field("y_exact", pa.float64(), nullable=False),
        pa.field("margin", pa.float64(), nullable=False),
        pa.field("threshold_valid", pa.bool_(), nullable=False),
        pa.field("good_candidate", pa.bool_(), nullable=False),
        pa.field("oracle_prunable", pa.bool_(), nullable=False),
        pa.field("raw_event_weight", pa.float64(), nullable=False),
        pa.field("per_query_weight", pa.float64(), nullable=False),
    ]
)

SCHEMA_ROLES = {
    "edge_encoder_inputs": [
        "source_external_label",
        "target_external_label",
        "graph_layer",
        "edge_squared_length",
    ],
    "online_decoder_inputs": [
        "query_id",
        "current_squared_distance",
        "threshold",
        "ef_search",
    ],
    "label_only": [
        "candidate_squared_distance",
        "y_exact",
        "margin",
        "good_candidate",
        "oracle_prunable",
    ],
    "weight_only": ["raw_event_weight", "per_query_weight"],
    "unavailable_in_source_trace": [
        "expansion_index",
        "candidate_queue_size",
        "result_queue_size",
        "search_progress",
    ],
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash_piece(digest: Any, value: bytes) -> None:
    digest.update(struct.pack("<Q", len(value)))
    digest.update(value)


def canonical_table_sha256(table: pa.Table, schema: pa.Schema) -> str:
    """Hash logical table values independent of Parquet bytes and physical paths."""

    table = table.select(schema.names).combine_chunks()
    if table.schema != schema:
        table = table.cast(schema)
    digest = hashlib.sha256()
    _hash_piece(digest, FORMAT_NAME.encode())
    digest.update(struct.pack("<Q", FORMAT_VERSION))
    digest.update(struct.pack("<Q", table.num_rows))
    for field in schema:
        _hash_piece(digest, field.name.encode("utf-8"))
        _hash_piece(digest, str(field.type).encode("ascii"))
        column = table[field.name]
        if column.null_count:
            raise ValueError(f"canonical hash does not permit nulls: {field.name}")
        if pa.types.is_string(field.type):
            for value in column.to_pylist():
                _hash_piece(digest, value.encode("utf-8"))
            continue
        if pa.types.is_boolean(field.type):
            values = np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.uint8)
        elif pa.types.is_int16(field.type):
            values = np.asarray(column.to_numpy(zero_copy_only=False), dtype="<i2")
        elif pa.types.is_int32(field.type):
            values = np.asarray(column.to_numpy(zero_copy_only=False), dtype="<i4")
        elif pa.types.is_int64(field.type):
            values = np.asarray(column.to_numpy(zero_copy_only=False), dtype="<i8")
        elif pa.types.is_float64(field.type):
            values = np.asarray(column.to_numpy(zero_copy_only=False), dtype="<f8")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"canonical hash requires finite values: {field.name}")
        else:
            raise TypeError(f"unsupported canonical hash type: {field.type}")
        data = np.ascontiguousarray(values).tobytes(order="C")
        _hash_piece(digest, data)
    return digest.hexdigest()


def table_from_arrays(schema: pa.Schema, values: dict[str, Any]) -> pa.Table:
    arrays = [pa.array(values[field.name], type=field.type) for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def write_parquet(table: pa.Table, path: Path) -> dict[str, Any]:
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
    )
    return {
        "relative_path": path.name,
        "rows": table.num_rows,
        "columns": table.num_columns,
        "byte_size": path.stat().st_size,
        "file_sha256": sha256_file(path),
    }


class Fvecs:
    def __init__(self, path: Path):
        self.path = path
        if path.stat().st_size < 4:
            raise ValueError(f"empty fvecs file: {path}")
        with path.open("rb") as handle:
            self.dim = struct.unpack("<i", handle.read(4))[0]
        if self.dim <= 0:
            raise ValueError(f"invalid fvecs dimension {self.dim}: {path}")
        self.record_bytes = 4 + 4 * self.dim
        if path.stat().st_size % self.record_bytes:
            raise ValueError(f"invalid fvecs byte size: {path}")
        self.count = path.stat().st_size // self.record_bytes
        self._raw = np.memmap(path, mode="r", dtype=np.uint8)

    def take(self, rows: np.ndarray | Sequence[int]) -> np.ndarray:
        row_array = np.asarray(rows, dtype=np.int64)
        if row_array.size and (row_array.min() < 0 or row_array.max() >= self.count):
            raise IndexError(f"fvecs row out of range [0, {self.count})")
        output = np.empty((row_array.size, self.dim), dtype=np.float32)
        for offset, row in enumerate(row_array):
            start = int(row) * self.record_bytes + 4
            output[offset] = np.frombuffer(
                self._raw[start : start + self.dim * 4], dtype="<f4", count=self.dim
            )
        return output

    def close(self) -> None:
        mmap = getattr(self._raw, "_mmap", None)
        if mmap is not None:
            mmap.close()


def query_bucket(seed: int, query_id: int) -> float:
    raw = hashlib.sha256(f"{seed}:{query_id}".encode("utf-8")).digest()[:8]
    return int.from_bytes(raw, byteorder="little", signed=False) / float(1 << 64)


def assign_query_splits(
    query_ids: Iterable[int], seed: int, fractions: Sequence[dict[str, Any]]
) -> dict[int, str]:
    total = sum(float(item["fraction"]) for item in fractions)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"split fractions sum to {total}, expected 1")
    names = [str(item["name"]) for item in fractions]
    if len(names) != len(set(names)):
        raise ValueError("split names must be unique")
    boundaries = np.cumsum([float(item["fraction"]) for item in fractions])
    result: dict[int, str] = {}
    for query_id in query_ids:
        bucket = query_bucket(seed, int(query_id))
        index = int(np.searchsorted(boundaries, bucket, side="right"))
        result[int(query_id)] = names[min(index, len(names) - 1)]
    return result


@dataclass(frozen=True)
class HnswHeader:
    offset_level0: int
    max_elements: int
    current_count: int
    size_data_per_element: int
    label_offset: int
    offset_data: int
    max_level: int
    entrypoint: int
    max_m: int
    max_m0: int
    m: int
    mult: float
    ef_construction: int


class NativeHnswLevel0:
    """Read level-0 adjacency from the native hnswlib saveIndex format."""

    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as handle:
            raw = handle.read(HNSW_HEADER.size)
        if len(raw) != HNSW_HEADER.size:
            raise ValueError(f"truncated native HNSW header: {path}")
        self.header = HnswHeader(*HNSW_HEADER.unpack(raw))
        minimum_size = HNSW_HEADER.size + (
            self.header.current_count * self.header.size_data_per_element
        )
        if path.stat().st_size < minimum_size:
            raise ValueError(f"truncated native HNSW level-0 data: {path}")
        if self.header.current_count > self.header.max_elements:
            raise ValueError("native HNSW current_count exceeds max_elements")
        self._raw = np.memmap(path, mode="r", dtype=np.uint8)

    def neighbors(self, source_internal_id: int) -> np.ndarray:
        source = int(source_internal_id)
        if source < 0 or source >= self.header.current_count:
            raise IndexError(f"HNSW internal id out of range: {source}")
        start = (
            HNSW_HEADER.size
            + source * self.header.size_data_per_element
            + self.header.offset_level0
        )
        count = struct.unpack_from("<H", self._raw, start)[0]
        if count > self.header.max_m0:
            raise ValueError(f"invalid level-0 neighbor count {count} for node {source}")
        return np.frombuffer(self._raw, dtype="<u4", count=count, offset=start + 4)

    def neighbor_slots(
        self, source_ids: np.ndarray, target_ids: np.ndarray
    ) -> np.ndarray:
        slots = np.full(len(source_ids), -1, dtype=np.int32)
        order = np.argsort(source_ids, kind="stable")
        for position in order:
            neighbors = self.neighbors(int(source_ids[position]))
            matches = np.flatnonzero(neighbors == int(target_ids[position]))
            if matches.size:
                slots[position] = int(matches[0])
        return slots

    def close(self) -> None:
        mmap = getattr(self._raw, "_mmap", None)
        if mmap is not None:
            mmap.close()


def schema_document() -> dict[str, Any]:
    def fields(schema: pa.Schema) -> list[dict[str, Any]]:
        return [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in schema
        ]

    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "tables": {
            "queries": fields(QUERY_SCHEMA),
            "edges": fields(EDGE_SCHEMA),
            "events": fields(EVENT_SCHEMA),
        },
        "roles": SCHEMA_ROLES,
        "edge_definition": "directed (source_internal_id, target_internal_id, graph_layer)",
        "event_identity": "D(q,v) = D(q,c) + ||v-c||^2 - 2 * <q-c,v-c>",
        "neighbor_slot_sentinel": -1,
    }


def semantic_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return path-free fields that define dataset meaning."""

    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "dataset_id": config["dataset_id"],
        "dataset_role": config["dataset_role"],
        "logical_artifacts": config["logical_artifacts"],
        "query_mapping": config["query_mapping"],
        "split": config["split"],
        "edge_key": ["source_internal_id", "target_internal_id", "graph_layer"],
    }


def dataset_semantic_sha256(
    config: dict[str, Any],
    table_hashes: dict[str, str],
    input_hashes: dict[str, str],
) -> str:
    return dataset_semantic_sha256_from_parts(
        semantic_config(config), table_hashes, input_hashes
    )


def dataset_semantic_sha256_from_parts(
    semantic_configuration: dict[str, Any],
    table_hashes: dict[str, str],
    input_hashes: dict[str, str],
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "semantic_config": semantic_configuration,
                "input_sha256": input_hashes,
                "schema": schema_document(),
                "table_sha256": table_hashes,
            }
        )
    )
