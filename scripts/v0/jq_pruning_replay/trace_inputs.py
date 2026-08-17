#!/usr/bin/env python3
"""Readers and frozen-input validation for JQ pruning replay."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np


REQUIRED_SHADOW_COLUMNS = {
    "query_id",
    "current_node_id",
    "candidate_id",
    "bound_status",
    "current_squared_distance",
    "threshold",
    "lower_bound",
    "shadow_exact_squared_distance",
    "would_prune",
    "lower_bound_valid",
    "lower_bound_violation",
    "false_prune",
}


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ShadowRecords:
    query_id: np.ndarray
    current_node_id: np.ndarray
    candidate_id: np.ndarray
    bound_status: np.ndarray
    current_squared_distance: np.ndarray
    threshold: np.ndarray
    recorded_lower_bound: np.ndarray
    shadow_exact_squared_distance: np.ndarray
    recorded_would_prune: np.ndarray
    recorded_valid: np.ndarray
    recorded_violation: np.ndarray
    recorded_false_prune: np.ndarray

    @property
    def count(self) -> int:
        return int(self.query_id.size)

    def validate_shapes(self) -> None:
        expected = self.count
        for name, value in self.__dict__.items():
            if value.ndim != 1 or value.size != expected:
                raise ValueError(f"shadow column {name} has inconsistent shape")

    def frozen_statistics(self) -> dict[str, int | float]:
        valid = self.recorded_valid
        valid_count = int(np.count_nonzero(valid))
        prune_count = int(np.count_nonzero(self.recorded_would_prune & valid))
        oracle = self.shadow_exact_squared_distance > self.threshold
        return {
            "row_count": self.count,
            "valid_count": valid_count,
            "would_prune_count": prune_count,
            "would_prune_rate": prune_count / valid_count if valid_count else 0.0,
            "oracle_count": int(np.count_nonzero(oracle & valid)),
            "oracle_rate": float(np.mean(oracle[valid])) if valid_count else 0.0,
            "lower_bound_violation_count": int(np.count_nonzero(self.recorded_violation)),
            "false_prune_count": int(np.count_nonzero(self.recorded_false_prune)),
        }


def _open_csv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def load_shadow_records(path: Path, *, max_rows: int | None = None) -> ShadowRecords:
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive")
    columns: dict[str, list] = {
        "query_id": [],
        "current_node_id": [],
        "candidate_id": [],
        "bound_status": [],
        "current_squared_distance": [],
        "threshold": [],
        "recorded_lower_bound": [],
        "shadow_exact_squared_distance": [],
        "recorded_would_prune": [],
        "recorded_valid": [],
        "recorded_violation": [],
        "recorded_false_prune": [],
    }
    with _open_csv(path) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("shadow CSV has no header")
        missing = REQUIRED_SHADOW_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"shadow CSV is missing columns: {sorted(missing)}")
        for row_index, row in enumerate(reader):
            if max_rows is not None and row_index >= max_rows:
                break
            columns["query_id"].append(int(row["query_id"]))
            columns["current_node_id"].append(int(row["current_node_id"]))
            columns["candidate_id"].append(int(row["candidate_id"]))
            columns["bound_status"].append(int(row["bound_status"]))
            columns["current_squared_distance"].append(
                float(row["current_squared_distance"])
            )
            columns["threshold"].append(float(row["threshold"]))
            columns["recorded_lower_bound"].append(float(row["lower_bound"]))
            columns["shadow_exact_squared_distance"].append(
                float(row["shadow_exact_squared_distance"])
            )
            columns["recorded_would_prune"].append(row["would_prune"] == "1")
            columns["recorded_valid"].append(row["lower_bound_valid"] == "1")
            columns["recorded_violation"].append(row["lower_bound_violation"] == "1")
            columns["recorded_false_prune"].append(row["false_prune"] == "1")
    records = ShadowRecords(
        query_id=np.asarray(columns["query_id"], dtype=np.uint64),
        current_node_id=np.asarray(columns["current_node_id"], dtype=np.uint64),
        candidate_id=np.asarray(columns["candidate_id"], dtype=np.uint64),
        bound_status=np.asarray(columns["bound_status"], dtype=np.uint8),
        current_squared_distance=np.asarray(
            columns["current_squared_distance"], dtype=np.float64
        ),
        threshold=np.asarray(columns["threshold"], dtype=np.float64),
        recorded_lower_bound=np.asarray(columns["recorded_lower_bound"], dtype=np.float64),
        shadow_exact_squared_distance=np.asarray(
            columns["shadow_exact_squared_distance"], dtype=np.float64
        ),
        recorded_would_prune=np.asarray(
            columns["recorded_would_prune"], dtype=np.bool_
        ),
        recorded_valid=np.asarray(columns["recorded_valid"], dtype=np.bool_),
        recorded_violation=np.asarray(
            columns["recorded_violation"], dtype=np.bool_
        ),
        recorded_false_prune=np.asarray(
            columns["recorded_false_prune"], dtype=np.bool_
        ),
    )
    records.validate_shapes()
    if records.count == 0:
        raise ValueError("shadow CSV contains no records")
    return records


def validate_frozen_shadow_contract(
    records: ShadowRecords,
    metadata_path: Path,
) -> dict:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "dataset": "gist1m",
        "dimension": 960,
        "query_count": 1000,
        "k": 10,
        "ef_search": 200,
        "shadow_sample_modulus": 16,
        "shadow_sample_remainder": 0,
    }
    failures = [
        f"metadata {key}={metadata.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    stats = records.frozen_statistics()
    count_expectations = {
        "row_count": 211_992,
        "valid_count": 211_836,
        "would_prune_count": 20,
        "lower_bound_violation_count": 0,
        "false_prune_count": 0,
    }
    failures.extend(
        f"shadow {key}={stats[key]!r}, expected {value!r}"
        for key, value in count_expectations.items()
        if stats[key] != value
    )
    recomputed = records.recorded_lower_bound > records.threshold
    if not np.array_equal(recomputed & records.recorded_valid, records.recorded_would_prune):
        failures.append("recorded would_prune disagrees with lower_bound > threshold")
    if failures:
        raise ValueError("frozen shadow contract failed: " + "; ".join(failures))
    return {"metadata": metadata, "statistics": stats}


class FvecsMemmap:
    """Read fixed-dimension fvecs without loading the whole file."""

    def __init__(self, path: Path, *, expected_dimension: int | None = None) -> None:
        self.path = path.resolve()
        if self.path.stat().st_size < 8:
            raise ValueError(f"fvecs file is too small: {self.path}")
        first = np.memmap(self.path, dtype="<i4", mode="r", shape=(1,))
        dimension = int(first[0])
        if dimension <= 0:
            raise ValueError("fvecs dimension must be positive")
        if expected_dimension is not None and dimension != expected_dimension:
            raise ValueError(
                f"fvecs dimension {dimension} != expected {expected_dimension}"
            )
        record_bytes = 4 * (dimension + 1)
        file_bytes = self.path.stat().st_size
        if file_bytes % record_bytes != 0:
            raise ValueError("fvecs file size is not a whole number of records")
        self.dimension = dimension
        self.count = file_bytes // record_bytes
        self._words = np.memmap(
            self.path,
            dtype="<i4",
            mode="r",
            shape=(self.count, dimension + 1),
        )
        sample = np.unique(
            np.linspace(0, self.count - 1, min(self.count, 1024), dtype=np.int64)
        )
        if not np.all(self._words[sample, 0] == dimension):
            raise ValueError("fvecs contains inconsistent record dimensions")

    def take(self, rows: np.ndarray) -> np.ndarray:
        indices = np.asarray(rows, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("fvecs row indices must be 1-D")
        if indices.size and (indices.min() < 0 or indices.max() >= self.count):
            raise IndexError("fvecs row index is out of range")
        words = np.ascontiguousarray(self._words[indices, 1:])
        return words.view("<f4").reshape(indices.size, self.dimension)

    def close(self) -> None:
        mmap = getattr(self._words, "_mmap", None)
        if mmap is not None:
            mmap.close()

    def __enter__(self) -> "FvecsMemmap":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def load_internal_to_label(path: Path, *, expected_count: int) -> np.ndarray:
    # The mapping is only 8 MB for GIST1M. Loading it eagerly avoids a Windows
    # file-handle lifetime that would otherwise block fixture cleanup.
    mapping = np.load(path, allow_pickle=False)
    if mapping.ndim != 1 or mapping.size != expected_count:
        raise ValueError("internal-to-label mapping has the wrong shape")
    if not np.issubdtype(mapping.dtype, np.integer):
        raise ValueError("internal-to-label mapping must contain integers")
    if mapping.size and (mapping.min() < 0 or mapping.max() >= expected_count):
        raise ValueError("internal-to-label mapping contains out-of-range labels")
    if np.unique(np.asarray(mapping)).size != expected_count:
        raise ValueError("internal-to-label mapping is not one-to-one")
    return mapping


def iter_slices(count: int, batch_size: int) -> Iterator[slice]:
    if count < 0 or batch_size <= 0:
        raise ValueError("invalid slice iteration arguments")
    for start in range(0, count, batch_size):
        yield slice(start, min(count, start + batch_size))
