#!/usr/bin/env python3
"""Self-contained tests for the Stage 1 operational dataset pipeline."""

from __future__ import annotations

import json
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "eq_rcp"
sys.path.insert(0, str(SCRIPT_DIR))

from build_operational_dataset import build_dataset  # noqa: E402
from operational_dataset import (  # noqa: E402
    EVENT_SCHEMA,
    HNSW_HEADER,
    NativeHnswLevel0,
    sha256_file,
)
from validate_operational_dataset import validate_dataset  # noqa: E402


def write_fvecs(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values, dtype="<f4")
    with path.open("wb") as handle:
        for row in values:
            handle.write(struct.pack("<i", values.shape[1]))
            handle.write(row.tobytes())


def write_fake_index(path: Path, adjacency: list[list[int]], dim: int) -> None:
    count = len(adjacency)
    max_m0 = max(4, max(len(row) for row in adjacency))
    offset_level0 = 0
    offset_data = 4 + max_m0 * 4
    label_offset = offset_data + dim * 4
    stride = label_offset + 8
    header = HNSW_HEADER.pack(
        offset_level0,
        count,
        count,
        stride,
        label_offset,
        offset_data,
        0,
        0,
        2,
        max_m0,
        2,
        1.0,
        20,
    )
    payload = bytearray(stride * count)
    for node, neighbors in enumerate(adjacency):
        start = node * stride
        struct.pack_into("<H", payload, start, len(neighbors))
        for slot, neighbor in enumerate(neighbors):
            struct.pack_into("<I", payload, start + 4 + slot * 4, neighbor)
        struct.pack_into("<Q", payload, start + label_offset, node)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(payload)
        handle.write(b"\x00\x00\x00\x00" * count)


def make_fixture(root: Path, with_index: bool = False) -> Path:
    base_values = np.asarray(
        [[0, 0], [2, 0], [0, 1], [1, 1], [3, 2]], dtype=np.float32
    )
    query_values = np.asarray(
        [[0.2, 0.1], [1.5, 0.2], [2.2, 1.8], [0.0, 1.2], [2.8, 0.1], [1.1, 1.1]],
        dtype=np.float32,
    )
    base_path = root / "base.fvecs"
    query_path = root / "query.fvecs"
    mapping_path = root / "mapping.npy"
    mapping_manifest_path = root / "mapping.manifest.json"
    records_path = root / "records.parquet"
    index_path = root / "index.bin"
    write_fvecs(base_path, base_values)
    write_fvecs(query_path, query_values)
    mapping = np.asarray([2, 0, 4, 1, 3], dtype=np.int64)
    np.save(mapping_path, mapping, allow_pickle=False)
    adjacency = [[1, 2], [2, 3], [3, 4], [4, 0], [0, 1]]
    if with_index:
        write_fake_index(index_path, adjacency, dim=2)
        index_hash = sha256_file(index_path)
    else:
        index_hash = "0" * 64
    mapping_manifest_path.write_text(
        json.dumps({"current_count": 5, "index_sha256": index_hash}), encoding="utf-8"
    )

    query_id = np.arange(6, dtype=np.int64)
    source = np.asarray([0, 1, 2, 3, 4, 0], dtype=np.int64)
    target = np.asarray([1, 2, 3, 4, 0, 2], dtype=np.int64)
    c = base_values[mapping[source]].astype(np.float64)
    v = base_values[mapping[target]].astype(np.float64)
    q = query_values.astype(np.float64)
    current = np.sum((q - c) ** 2, axis=1)
    candidate = np.sum((q - v) ** 2, axis=1)
    threshold = np.asarray([0.5, 1.0, 2.0, 0.4, 1.5, 0.8], dtype=np.float64)
    table = pa.table(
        {
            "query_id": query_id,
            "current_node_id": source,
            "candidate_id": target,
            "graph_layer": np.zeros(6, dtype=np.int32),
            "ef_search": np.full(6, 40, dtype=np.int32),
            "current_squared_distance": current,
            "threshold": threshold,
            "shadow_exact_squared_distance": candidate,
        }
    )
    pq.write_table(table, records_path)
    logical = {
        "records": "fixture/records",
        "base_vectors": "fixture/base",
        "query_vectors": "fixture/query",
        "internal_to_label": "fixture/mapping",
        "mapping_manifest": "fixture/mapping-manifest",
        "index": "fixture/index",
    }
    paths = {
        "records": str(records_path),
        "base_vectors": str(base_path),
        "query_vectors": str(query_path),
        "internal_to_label": str(mapping_path),
        "mapping_manifest": str(mapping_manifest_path),
        "index": str(index_path) if with_index else None,
    }
    config = {
        "format": "eq_rcp_operational_dataset",
        "version": 1,
        "dataset_id": "fixture-v1",
        "dataset_role": "formal" if with_index else "development",
        "logical_artifacts": logical,
        "paths": paths,
        "expected_sha256": {"index": index_hash},
        "query_mapping": {"mode": "query_id_is_row", "offset": 0},
        "split": {
            "seed": 7,
            "fractions": (
                [{"name": "all", "fraction": 1.0}]
                if with_index
                else [{"name": "development_all", "fraction": 1.0}]
            ),
        },
        "build": {"batch_size": 2},
        "validation": {
            "expected_rows": 6,
            "distance_abs_tolerance": 1e-10,
            "require_index_adjacency_validation": with_index,
        },
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def run_tests() -> None:
    with tempfile.TemporaryDirectory(prefix="eq-rcp-stage1-") as temp:
        root = Path(temp)
        config_path = make_fixture(root)
        output_a = root / "output-a"
        output_b = root / "different-physical-output-b"
        manifest_a = build_dataset(config_path, output_a)
        manifest_b = build_dataset(config_path, output_b)
        assert manifest_a["dataset_semantic_sha256"] == manifest_b["dataset_semantic_sha256"]
        assert manifest_a["development_only"] is True
        assert manifest_a["adjacency_validation"]["status"] == "PENDING_INDEX_UNAVAILABLE"
        report = validate_dataset(output_a, geometry_limit=0)
        assert report["status"] == "PASS"
        assert report["geometry_records_checked"] == 6

        # Semantic hashes detect logical table corruption even if Parquet is rewritten.
        corrupt = root / "corrupt"
        shutil.copytree(output_a, corrupt)
        events_path = corrupt / "events.parquet"
        events = pq.read_table(events_path)
        values = events["margin"].to_numpy(zero_copy_only=False).copy()
        values[0] += 1.0
        events = events.set_column(
            events.schema.get_field_index("margin"),
            EVENT_SCHEMA.field("margin"),
            pa.array(values, type=pa.float64()),
        )
        pq.write_table(events, events_path)
        try:
            validate_dataset(corrupt, geometry_limit=0)
        except ValueError as exc:
            assert "SHA-256" in str(exc)
        else:
            raise AssertionError("corrupted event table unexpectedly passed")

    with tempfile.TemporaryDirectory(prefix="eq-rcp-stage1-index-") as temp:
        root = Path(temp)
        config_path = make_fixture(root, with_index=True)
        index = NativeHnswLevel0(root / "index.bin")
        assert index.neighbors(0).tolist() == [1, 2]
        index.close()
        output = root / "formal-output"
        manifest = build_dataset(config_path, output)
        assert manifest["adjacency_validation"]["status"] == "PASS"
        assert manifest["formal_materialization_status"] == "PASS"
        report = validate_dataset(output, geometry_limit=0)
        assert report["formal_materialization_status"] == "PASS"


if __name__ == "__main__":
    run_tests()
    print("EQ-RCP Stage 1 dataset tests: PASS")
