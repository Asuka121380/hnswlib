#!/usr/bin/env python3
"""Create the canonical tiny sidecar-v1 fixture for cross-language tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


MAGIC = b"HNSWV0PQ"
CHECKSUM_MAGIC = b"V0SUMS01"
VERSION = 1
HEADER_SIZE = 256
CHECKSUM_SIZE = 176


def _sha(value: bytes) -> bytes:
    return hashlib.sha256(value).digest()


def create_fixture(path: Path) -> None:
    dimension = 4
    pq_m = 2
    nbits = 1
    ksub = 2
    dsub = 2
    code_size = 2
    scalar_offset = 8
    edge_stride = 40
    node_count = 3
    edge_count = 3

    training = json.dumps(
        {"quantizer": "synthetic", "seed": 17},
        separators=(",", ":"),
    ).encode("utf-8")
    codebook = struct.pack(
        "<8f",
        0.0,
        0.5,
        1.0,
        1.5,
        -1.0,
        -0.5,
        2.0,
        2.5,
    )
    node_offsets = struct.pack("<4Q", 0, 2, 2, 3)

    record_values = (
        (bytes((0, 1)), 0, 1.25, 0.125, -2.5, 0.0),
        (bytes((1, 0)), 1, 2.5, 0.25, 4.0, 0.001),
        (bytes((0, 0)), 3, 0.0, 0.0, 0.0, 0.0),
    )
    records = bytearray()
    for code, flags, length, error, anchor, padding in record_values:
        prefix = code + bytes((flags,))
        prefix += bytes(scalar_offset - len(prefix))
        records += prefix + struct.pack("<dddd", length, error, anchor, padding)
    edge_records = bytes(records)

    training_offset = HEADER_SIZE
    codebook_offset = training_offset + len(training)
    offsets_offset = codebook_offset + len(codebook)
    edges_offset = offsets_offset + len(node_offsets)
    checksums_offset = edges_offset + len(edge_records)

    header = bytearray()
    header += MAGIC
    header += struct.pack("<II", VERSION, HEADER_SIZE)
    header += bytes((1, 1, 1, 1))
    header += struct.pack(
        "<7I",
        dimension,
        pq_m,
        nbits,
        ksub,
        dsub,
        code_size,
        edge_stride,
    )
    header += struct.pack("<2Q", node_count, edge_count)
    header += struct.pack(
        "<10Q",
        training_offset,
        len(training),
        codebook_offset,
        len(codebook),
        offsets_offset,
        len(node_offsets),
        edges_offset,
        len(edge_records),
        checksums_offset,
        CHECKSUM_SIZE,
    )
    header += _sha(b"synthetic-base-index")
    header += _sha(b"synthetic-adjacency")
    header += _sha(codebook)
    header += bytes(HEADER_SIZE - len(header))
    assert len(header) == HEADER_SIZE

    checksum_section = bytearray()
    checksum_section += CHECKSUM_MAGIC
    checksum_section += struct.pack("<II", VERSION, 5)
    checksum_section += _sha(bytes(header))
    checksum_section += _sha(training)
    checksum_section += _sha(codebook)
    checksum_section += _sha(node_offsets)
    checksum_section += _sha(edge_records)
    assert len(checksum_section) == CHECKSUM_SIZE

    path.write_bytes(
        bytes(header)
        + training
        + codebook
        + node_offsets
        + edge_records
        + bytes(checksum_section)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    create_fixture(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
