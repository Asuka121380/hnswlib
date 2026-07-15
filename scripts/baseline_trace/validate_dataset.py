#!/usr/bin/env python3
"""Validate a flat real-dataset manifest and vector-file record structure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct


def inspect(path: Path, component_bytes: int, expected_dimension: int, expected_count: int) -> None:
    if not path.is_file():
        raise RuntimeError(f"Missing file: {path}")
    with path.open("rb") as handle:
        raw = handle.read(4)
    if len(raw) != 4:
        raise RuntimeError(f"Empty file: {path}")
    dimension = struct.unpack("<i", raw)[0]
    if dimension != expected_dimension:
        raise RuntimeError(f"Dimension mismatch for {path}: {dimension} != {expected_dimension}")
    record_bytes = 4 + dimension * component_bytes
    size = path.stat().st_size
    if size % record_bytes:
        raise RuntimeError(f"Malformed file size: {path}")
    count = size // record_bytes
    if count != expected_count:
        raise RuntimeError(f"Count mismatch for {path}: {count} != {expected_count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.dataset_config.read_text(encoding="utf-8"))
    base_bytes = 4 if config["base_format"] == "fvecs" else 1
    query_bytes = 4 if config["query_format"] == "fvecs" else 1
    inspect(Path(config["base_path"]), base_bytes, config["dimension"], config["n_base"])
    inspect(Path(config["query_path"]), query_bytes, config["dimension"], config["n_query"])
    inspect(Path(config["ground_truth_path"]), 4, config["ground_truth_k"], config["n_query"])
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
