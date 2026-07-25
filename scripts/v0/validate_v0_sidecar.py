#!/usr/bin/env python3
"""Independent V0 sidecar-v1 structural and checksum validator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any


SIDECAR_MAGIC = b"HNSWV0PQ"
CHECKSUM_MAGIC = b"V0SUMS01"
VERSION = 1
HEADER_SIZE = 256
CHECKSUM_SIZE = 176


class ValidationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _u64(data: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", data, offset)[0]


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _section(data: bytes, offset: int, size: int, name: str) -> bytes:
    _require(offset <= len(data), f"{name} offset is out of bounds")
    _require(size <= len(data) - offset, f"{name} size is out of bounds")
    return data[offset : offset + size]


def validate(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    _require(len(data) >= HEADER_SIZE, "file is shorter than fixed header")
    _require(data[:8] == SIDECAR_MAGIC, "invalid sidecar magic")

    version = _u32(data, 8)
    header_size = _u32(data, 12)
    _require(version == VERSION, "unsupported sidecar version")
    _require(header_size == HEADER_SIZE, "unsupported header size")
    _require(data[16:20] == b"\x01\x01\x01\x01", "unsupported representation")

    dimension = _u32(data, 20)
    pq_m = _u32(data, 24)
    nbits = _u32(data, 28)
    ksub = _u32(data, 32)
    dsub = _u32(data, 36)
    code_size = _u32(data, 40)
    edge_stride = _u32(data, 44)
    node_count = _u64(data, 48)
    edge_count = _u64(data, 56)

    training_offset, training_size = _u64(data, 64), _u64(data, 72)
    codebook_offset, codebook_size = _u64(data, 80), _u64(data, 88)
    offsets_offset, offsets_size = _u64(data, 96), _u64(data, 104)
    edges_offset, edges_size = _u64(data, 112), _u64(data, 120)
    checksums_offset, checksums_size = _u64(data, 128), _u64(data, 136)

    base_hash = data[144:176]
    adjacency_hash = data[176:208]
    stored_codebook_hash = data[208:240]
    _require(data[240:256] == bytes(16), "reserved header bytes are non-zero")

    _require(dimension > 0 and pq_m > 0 and dsub > 0, "invalid PQ dimensions")
    _require(1 <= nbits <= 8, "invalid PQ bit width")
    _require(ksub == 1 << nbits, "ksub does not match nbits")
    _require(dimension == pq_m * dsub, "dimension does not equal M*dsub")
    _require(code_size == pq_m, "v1 code size must equal M")
    scalar_offset = (code_size + 1 + 7) & ~7
    _require(edge_stride == scalar_offset + 32, "invalid edge-record stride")

    _require(codebook_size == pq_m * ksub * dsub * 4, "invalid codebook size")
    _require(offsets_size == (node_count + 1) * 8, "invalid node-offset size")
    _require(edges_size == edge_count * edge_stride, "invalid edge-record size")
    _require(checksums_size == CHECKSUM_SIZE, "invalid checksum size")

    expected = HEADER_SIZE
    _require(training_offset == expected, "non-canonical training offset")
    expected += training_size
    _require(codebook_offset == expected, "non-canonical codebook offset")
    expected += codebook_size
    _require(offsets_offset == expected, "non-canonical node-offset offset")
    expected += offsets_size
    _require(edges_offset == expected, "non-canonical edge-record offset")
    expected += edges_size
    _require(checksums_offset == expected, "non-canonical checksum offset")
    expected += checksums_size
    _require(expected == len(data), "file size does not match canonical layout")

    training = _section(data, training_offset, training_size, "training metadata")
    codebook = _section(data, codebook_offset, codebook_size, "codebook")
    node_offsets = _section(data, offsets_offset, offsets_size, "node offsets")
    edge_records = _section(data, edges_offset, edges_size, "edge records")
    checksums = _section(data, checksums_offset, checksums_size, "checksums")

    _require(checksums[:8] == CHECKSUM_MAGIC, "invalid checksum magic")
    _require(_u32(checksums, 8) == VERSION, "invalid checksum version")
    _require(_u32(checksums, 12) == 5, "invalid checksum count")
    expected_hashes = (
        checksums[16:48],
        checksums[48:80],
        checksums[80:112],
        checksums[112:144],
        checksums[144:176],
    )
    actual_hashes = (
        _sha256(data[:HEADER_SIZE]),
        _sha256(training),
        _sha256(codebook),
        _sha256(node_offsets),
        _sha256(edge_records),
    )
    _require(expected_hashes == actual_hashes, "section checksum mismatch")
    _require(stored_codebook_hash == actual_hashes[2], "codebook hash mismatch")

    try:
        training_text = training.decode("utf-8")
        training_json = json.loads(training_text) if training_text else None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid training metadata JSON: {exc}") from exc

    offsets = struct.unpack(f"<{node_count + 1}Q", node_offsets)
    _require(offsets[0] == 0, "node offsets do not start at zero")
    _require(offsets[-1] == edge_count, "node offsets do not end at edge count")
    _require(
        all(left <= right for left, right in zip(offsets, offsets[1:])),
        "node offsets are not monotonic",
    )

    centroid_count = pq_m * ksub * dsub
    centroids = struct.unpack(f"<{centroid_count}f", codebook)
    _require(all(math.isfinite(value) for value in centroids), "non-finite centroid")

    allowed_flags = 0x07
    max_direction_error = 0.0
    exact_only_count = 0
    for edge in range(edge_count):
        start = edge * edge_stride
        record = edge_records[start : start + edge_stride]
        _require(all(code < ksub for code in record[:code_size]), "invalid PQ code")
        flags = record[code_size]
        _require(flags & ~allowed_flags == 0, "unknown edge flags")
        _require(
            record[code_size + 1 : scalar_offset] == bytes(scalar_offset - code_size - 1),
            "non-zero edge-record padding",
        )
        edge_length, direction_error, anchor, numeric_padding = struct.unpack_from(
            "<dddd", record, scalar_offset
        )
        _require(math.isfinite(edge_length) and edge_length >= 0, "invalid edge length")
        _require(
            math.isfinite(direction_error) and direction_error >= 0,
            "invalid direction error",
        )
        _require(math.isfinite(anchor), "invalid anchor projection")
        _require(
            math.isfinite(numeric_padding) and numeric_padding >= 0,
            "invalid numeric padding",
        )
        max_direction_error = max(max_direction_error, direction_error)
        exact_only_count += int(bool(flags & 0x01))

    return {
        "path": str(path),
        "format_version": version,
        "dimension": dimension,
        "node_count": node_count,
        "directed_edge_count": edge_count,
        "pq_m": pq_m,
        "pq_nbits": nbits,
        "pq_ksub": ksub,
        "pq_dsub": dsub,
        "pq_code_size": code_size,
        "edge_record_stride": edge_stride,
        "base_index_sha256": base_hash.hex(),
        "adjacency_sha256": adjacency_hash.hex(),
        "codebook_sha256": stored_codebook_hash.hex(),
        "exact_only_count": exact_only_count,
        "max_direction_error": max_direction_error,
        "training_metadata": training_json,
        "status": "valid",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    try:
        summary = validate(args.sidecar)
    except (OSError, ValidationError) as exc:
        parser.error(str(exc))

    if args.as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "valid "
            f"version={summary['format_version']} "
            f"nodes={summary['node_count']} "
            f"edges={summary['directed_edge_count']} "
            f"M={summary['pq_m']} "
            f"nbits={summary['pq_nbits']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
