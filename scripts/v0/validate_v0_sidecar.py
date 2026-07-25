#!/usr/bin/env python3
"""Independent V0 sidecar-v1 structural and checksum validator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import struct
from pathlib import Path
from typing import Any

from v0pq_format import V0PQFormatError, read_v0pq


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


def _require_section(
    data: mmap.mmap, offset: int, size: int, name: str
) -> None:
    _require(offset <= len(data), f"{name} offset is out of bounds")
    _require(size <= len(data) - offset, f"{name} size is out of bounds")


def _sha256_section(data: mmap.mmap, offset: int, size: int) -> bytes:
    digest = hashlib.sha256()
    end = offset + size
    while offset < end:
        next_offset = min(offset + 1024 * 1024, end)
        digest.update(data[offset:next_offset])
        offset = next_offset
    return digest.digest()


def _parse_sha256(value: str, field: str) -> bytes:
    _require(len(value) == 64, f"{field} must contain 64 hexadecimal digits")
    try:
        parsed = bytes.fromhex(value)
    except ValueError as exc:
        raise ValidationError(f"{field} is not hexadecimal") from exc
    _require(len(parsed) == 32, f"{field} must contain a SHA-256 digest")
    return parsed


def _validate_mapped(
    data: mmap.mmap,
    path: Path,
    *,
    expected_index_sha256: str | None = None,
    expected_adjacency_sha256: str | None = None,
    expected_codebook_sha256: str | None = None,
    expected_node_count: int | None = None,
    expected_edge_count: int | None = None,
    expected_zero_length_count: int | None = None,
) -> dict[str, Any]:
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

    _require_section(
        data, training_offset, training_size, "training metadata"
    )
    _require_section(data, codebook_offset, codebook_size, "codebook")
    _require_section(data, offsets_offset, offsets_size, "node offsets")
    _require_section(data, edges_offset, edges_size, "edge records")
    _require_section(data, checksums_offset, checksums_size, "checksums")

    _require(
        data[checksums_offset : checksums_offset + 8] == CHECKSUM_MAGIC,
        "invalid checksum magic",
    )
    _require(
        _u32(data, checksums_offset + 8) == VERSION,
        "invalid checksum version",
    )
    _require(
        _u32(data, checksums_offset + 12) == 5,
        "invalid checksum count",
    )
    expected_hashes = (
        data[checksums_offset + 16 : checksums_offset + 48],
        data[checksums_offset + 48 : checksums_offset + 80],
        data[checksums_offset + 80 : checksums_offset + 112],
        data[checksums_offset + 112 : checksums_offset + 144],
        data[checksums_offset + 144 : checksums_offset + 176],
    )
    actual_hashes = (
        _sha256_section(data, 0, HEADER_SIZE),
        _sha256_section(data, training_offset, training_size),
        _sha256_section(data, codebook_offset, codebook_size),
        _sha256_section(data, offsets_offset, offsets_size),
        _sha256_section(data, edges_offset, edges_size),
    )
    _require(expected_hashes == actual_hashes, "section checksum mismatch")
    _require(stored_codebook_hash == actual_hashes[2], "codebook hash mismatch")

    try:
        training_text = data[
            training_offset : training_offset + training_size
        ].decode("utf-8")
        training_json = json.loads(training_text) if training_text else None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid training metadata JSON: {exc}") from exc

    previous_offset = _u64(data, offsets_offset)
    _require(previous_offset == 0, "node offsets do not start at zero")
    for node in range(1, node_count + 1):
        current_offset = _u64(data, offsets_offset + node * 8)
        _require(
            previous_offset <= current_offset,
            "node offsets are not monotonic",
        )
        previous_offset = current_offset
    _require(previous_offset == edge_count, "node offsets do not end at edge count")

    centroid_count = pq_m * ksub * dsub
    for centroid in range(centroid_count):
        value = struct.unpack_from(
            "<f", data, codebook_offset + centroid * 4
        )[0]
        _require(math.isfinite(value), "non-finite centroid")

    allowed_flags = 0x07
    max_direction_error = 0.0
    direction_error_sum = 0.0
    exact_only_count = 0
    zero_length_count = 0
    for edge in range(edge_count):
        start = edges_offset + edge * edge_stride
        _require(
            all(data[start + index] < ksub for index in range(code_size)),
            "invalid PQ code",
        )
        flags = data[start + code_size]
        _require(flags & ~allowed_flags == 0, "unknown edge flags")
        _require(
            data[
                start + code_size + 1 : start + scalar_offset
            ]
            == bytes(scalar_offset - code_size - 1),
            "non-zero edge-record padding",
        )
        edge_length, direction_error, anchor, numeric_padding = struct.unpack_from(
            "<dddd", data, start + scalar_offset
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
        direction_error_sum += direction_error
        exact_only_count += int(bool(flags & 0x01))
        is_zero_length = bool(flags & 0x02)
        zero_length_count += int(is_zero_length)
        if is_zero_length:
            _require(
                bool(flags & 0x01),
                "zero-length edge is not marked exact-only",
            )
            _require(
                all(
                    data[start + index] == 0
                    for index in range(code_size)
                ),
                "zero-length edge code is not zero-filled",
            )
            _require(
                edge_length == 0.0
                and direction_error == 0.0
                and anchor == 0.0
                and numeric_padding == 0.0,
                "zero-length edge metadata is not all-zero",
            )
        else:
            _require(edge_length > 0.0, "non-zero edge has zero length")

    if expected_index_sha256 is not None:
        _require(
            base_hash == _parse_sha256(
                expected_index_sha256, "expected_index_sha256"
            ),
            "base-index SHA-256 does not match expectation",
        )
    if expected_adjacency_sha256 is not None:
        _require(
            adjacency_hash
            == _parse_sha256(
                expected_adjacency_sha256,
                "expected_adjacency_sha256",
            ),
            "adjacency SHA-256 does not match expectation",
        )
    if expected_codebook_sha256 is not None:
        _require(
            stored_codebook_hash
            == _parse_sha256(
                expected_codebook_sha256,
                "expected_codebook_sha256",
            ),
            "codebook SHA-256 does not match expectation",
        )
    if expected_node_count is not None:
        _require(
            node_count == expected_node_count,
            "node count does not match expectation",
        )
    if expected_edge_count is not None:
        _require(
            edge_count == expected_edge_count,
            "edge count does not match expectation",
        )
    if expected_zero_length_count is not None:
        _require(
            zero_length_count == expected_zero_length_count,
            "zero-length edge count does not match expectation",
        )

    result = {
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
        "zero_length_edge_count": zero_length_count,
        "mean_direction_error": (
            direction_error_sum / (edge_count - zero_length_count)
            if edge_count != zero_length_count
            else 0.0
        ),
        "max_direction_error": max_direction_error,
        "training_metadata": training_json,
        "status": "valid",
    }
    return result


def validate(
    path: Path,
    *,
    expected_v0pq: Path | None = None,
    expected_index_sha256: str | None = None,
    expected_adjacency_sha256: str | None = None,
    expected_codebook_sha256: str | None = None,
    expected_node_count: int | None = None,
    expected_edge_count: int | None = None,
    expected_zero_length_count: int | None = None,
) -> dict[str, Any]:
    _require(path.stat().st_size > 0, "sidecar file is empty")
    with path.open("rb") as handle:
        data = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            result = _validate_mapped(
                data,
                path,
                expected_index_sha256=expected_index_sha256,
                expected_adjacency_sha256=expected_adjacency_sha256,
                expected_codebook_sha256=expected_codebook_sha256,
                expected_node_count=expected_node_count,
                expected_edge_count=expected_edge_count,
                expected_zero_length_count=expected_zero_length_count,
            )
        finally:
            data.close()
    if expected_v0pq is not None:
        codebook = read_v0pq(expected_v0pq)
        _require(
            result["dimension"] == codebook.header.dimension
            and result["pq_m"] == codebook.header.M
            and result["pq_nbits"] == codebook.header.nbits
            and result["pq_ksub"] == codebook.header.ksub
            and result["pq_dsub"] == codebook.header.dsub,
            "sidecar PQ configuration does not match expected V0PQ",
        )
        _require(
            result["codebook_sha256"]
            == codebook.header.centroids_sha256,
            "sidecar centroids do not match expected V0PQ",
        )
        result["expected_v0pq"] = str(expected_v0pq)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--expected-v0pq", type=Path)
    parser.add_argument("--expected-index-sha256")
    parser.add_argument("--expected-adjacency-sha256")
    parser.add_argument("--expected-codebook-sha256")
    parser.add_argument("--expected-node-count", type=int)
    parser.add_argument("--expected-edge-count", type=int)
    parser.add_argument("--expected-zero-length-count", type=int)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    try:
        summary = validate(
            args.sidecar,
            expected_v0pq=args.expected_v0pq,
            expected_index_sha256=args.expected_index_sha256,
            expected_adjacency_sha256=args.expected_adjacency_sha256,
            expected_codebook_sha256=args.expected_codebook_sha256,
            expected_node_count=args.expected_node_count,
            expected_edge_count=args.expected_edge_count,
            expected_zero_length_count=args.expected_zero_length_count,
        )
    except (OSError, ValidationError, V0PQFormatError) as exc:
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
