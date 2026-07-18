#!/usr/bin/env python3
"""Validate a flat real-dataset manifest and vector-file record structure."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
from pathlib import Path
import struct


SUPPORTED_FORMATS = {"fvecs": 4, "bvecs": 1, "ivecs": 4}


def inspect(path: Path, vector_format: str, expected_dimension: int, expected_count: int) -> None:
    if not path.is_file():
        raise RuntimeError(f"Missing file: {path}")
    if vector_format not in SUPPORTED_FORMATS:
        raise RuntimeError(f"Unsupported vector format {vector_format!r} for {path}")
    component_bytes = SUPPORTED_FORMATS[vector_format]
    size = path.stat().st_size
    if size < 4:
        raise RuntimeError(f"Empty file: {path}")
    record_bytes = 4 + expected_dimension * component_bytes
    if size % record_bytes:
        raise RuntimeError(f"Malformed file size: {path}")
    count = size // record_bytes
    if count != expected_count:
        raise RuntimeError(f"Count mismatch for {path}: {count} != {expected_count}")
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        for row in range(expected_count):
            dimension = struct.unpack_from("<i", data, row * record_bytes)[0]
            if dimension != expected_dimension:
                raise RuntimeError(
                    f"Dimension mismatch for {path} at record {row}: "
                    f"{dimension} != {expected_dimension}"
                )


def validate_ground_truth_labels(path: Path, ground_truth_k: int, n_query: int, n_base: int) -> None:
    record_bytes = 4 + ground_truth_k * 4
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        for row in range(n_query):
            offset = row * record_bytes + 4
            labels = struct.unpack_from(f"<{ground_truth_k}i", data, offset)
            for column, label in enumerate(labels):
                if label < 0 or label >= n_base:
                    raise RuntimeError(
                        f"Ground-truth label out of range at query {row}, column {column}: {label}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.dataset_config.read_text(encoding="utf-8"))
    if config.get("distance_kind") != "squared_l2_float32":
        raise RuntimeError(f"Unsupported distance_kind: {config.get('distance_kind')!r}")
    inspect(Path(config["base_path"]), config["base_format"], config["dimension"], config["n_base"])
    inspect(Path(config["query_path"]), config["query_format"], config["dimension"], config["n_query"])
    inspect(
        Path(config["ground_truth_path"]),
        config["ground_truth_format"],
        config["ground_truth_k"],
        config["n_query"],
    )
    validate_ground_truth_labels(
        Path(config["ground_truth_path"]),
        config["ground_truth_k"],
        config["n_query"],
        config["n_base"],
    )
    checksums = config.get("checksums", {})
    for key, path_key in (
        ("base_sha256", "base_path"),
        ("query_sha256", "query_path"),
        ("ground_truth_sha256", "ground_truth_path"),
    ):
        if key not in checksums:
            continue
        digest = hashlib.sha256()
        with Path(config[path_key]).open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != checksums[key]:
            raise RuntimeError(f"Checksum mismatch for {config[path_key]}")
    print("validate_dataset_ok")


if __name__ == "__main__":
    main()
