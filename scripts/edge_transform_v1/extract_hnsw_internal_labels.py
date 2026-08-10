#!/usr/bin/env python3
"""Extract internal-id -> external-label mapping from a native hnswlib index."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


HEADER = struct.Struct("<6QiI3QdQ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(index_path: Path) -> tuple[np.ndarray, dict[str, int | float]]:
    with index_path.open("rb") as stream:
        raw = stream.read(HEADER.size)
    if len(raw) != HEADER.size:
        raise ValueError("index is shorter than the native hnswlib header")
    values = HEADER.unpack(raw)
    names = (
        "offset_level0", "max_elements", "cur_element_count", "size_data_per_element",
        "label_offset", "offset_data", "max_level", "enterpoint_node", "max_m",
        "max_m0", "m", "mult", "ef_construction",
    )
    header = dict(zip(names, values))
    count = int(header["cur_element_count"])
    stride = int(header["size_data_per_element"])
    label_offset = int(header["label_offset"])
    if count <= 0 or stride <= 0 or label_offset < 0 or label_offset + 8 > stride:
        raise ValueError(f"implausible hnswlib header: {header}")
    level0_end = HEADER.size + count * stride
    if level0_end > index_path.stat().st_size:
        raise ValueError("truncated level-0 index data")
    raw_index = np.memmap(index_path, dtype=np.uint8, mode="r")
    labels_view = np.ndarray(
        shape=(count,), dtype="<u8", buffer=raw_index,
        offset=HEADER.size + label_offset, strides=(stride,),
    )
    return np.asarray(labels_view, dtype=np.uint64).copy(), header


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--require-permutation", action="store_true")
    args = parser.parse_args()
    labels, header = extract(args.index)
    if args.expected_count is not None and labels.size != args.expected_count:
        raise SystemExit(f"count mismatch: index={labels.size}, expected={args.expected_count}")
    if args.require_permutation:
        expected = labels.size if args.expected_count is None else args.expected_count
        if np.any(labels >= expected) or np.unique(labels).size != labels.size:
            raise SystemExit("labels are not a permutation of [0, expected_count)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, labels, allow_pickle=False)
    output_path = args.output if args.output.suffix == ".npy" else args.output.with_suffix(args.output.suffix + ".npy")
    report = {
        "format": "hnswlib_internal_to_external_label",
        "format_version": 1,
        "index": str(args.index),
        "index_bytes": args.index.stat().st_size,
        "index_sha256": sha256_file(args.index),
        "mapping": str(output_path),
        "mapping_sha256": sha256_file(output_path),
        "count": int(labels.size),
        "min_label": int(labels.min()),
        "max_label": int(labels.max()),
        "unique_labels": int(np.unique(labels).size),
        "header": header,
    }
    manifest = output_path.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
