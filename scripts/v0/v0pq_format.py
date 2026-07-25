#!/usr/bin/env python3
"""Versioned, dependency-free V0 PQ codebook format."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MAGIC = b"V0PQBOOK"
CHECKSUM_MAGIC = b"V0PQSUM1"
VERSION = 1
HEADER_SIZE = 256
CHECKSUM_SIZE = 112
CHECKSUM_COUNT = 3


class V0PQFormatError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V0PQFormatError(message)


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_sha256(value: str | bytes, field: str) -> bytes:
    if isinstance(value, bytes):
        _require(len(value) == 32, f"{field} must contain 32 bytes")
        return value
    _require(len(value) == 64, f"{field} must contain 64 hexadecimal digits")
    try:
        parsed = bytes.fromhex(value)
    except ValueError as exc:
        raise V0PQFormatError(f"{field} is not hexadecimal") from exc
    _require(len(parsed) == 32, f"{field} must contain a SHA-256 digest")
    return parsed


@dataclass(frozen=True)
class V0PQHeader:
    dimension: int
    M: int
    nbits: int
    ksub: int
    dsub: int
    code_size: int
    centroid_count: int
    training_metadata_offset: int
    training_metadata_size: int
    centroids_offset: int
    centroids_size: int
    checksums_offset: int
    checksums_size: int
    source_manifest_sha256: str
    source_directions_sha256: str
    centroids_sha256: str


@dataclass(frozen=True)
class V0PQCodebook:
    header: V0PQHeader
    training_metadata: dict[str, Any]
    centroids: tuple[float, ...]

    def centroid(self, m: int, code: int, coordinate: int) -> float:
        header = self.header
        if not 0 <= m < header.M:
            raise IndexError("subquantizer is out of range")
        if not 0 <= code < header.ksub:
            raise IndexError("centroid code is out of range")
        if not 0 <= coordinate < header.dsub:
            raise IndexError("centroid coordinate is out of range")
        index = (m * header.ksub + code) * header.dsub + coordinate
        return self.centroids[index]


def _validate_configuration(
    dimension: int,
    M: int,
    nbits: int,
    ksub: int,
    dsub: int,
    code_size: int,
    centroid_count: int,
) -> None:
    _require(dimension > 0, "dimension must be positive")
    _require(M > 0, "M must be positive")
    _require(1 <= nbits <= 8, "v1 nbits must be between 1 and 8")
    _require(ksub == 1 << nbits, "ksub does not match nbits")
    _require(dsub > 0 and dimension == M * dsub, "dimension must equal M*dsub")
    _require(code_size == M, "v1 code_size must equal M")
    _require(
        centroid_count == M * ksub * dsub,
        "centroid count does not match M*ksub*dsub",
    )


def _encode_header(
    *,
    dimension: int,
    M: int,
    nbits: int,
    ksub: int,
    dsub: int,
    centroid_count: int,
    metadata_offset: int,
    metadata_size: int,
    centroids_offset: int,
    centroids_size: int,
    checksums_offset: int,
    manifest_sha256: bytes,
    directions_sha256: bytes,
    centroids_sha256: bytes,
) -> bytes:
    header = bytearray()
    header += MAGIC
    header += struct.pack("<II", VERSION, HEADER_SIZE)
    header += bytes((1, 1, 1, 0))  # little, float32, standard PQ, reserved
    header += struct.pack("<6I", dimension, M, nbits, ksub, dsub, M)
    header += struct.pack("<Q", centroid_count)
    header += struct.pack(
        "<6Q",
        metadata_offset,
        metadata_size,
        centroids_offset,
        centroids_size,
        checksums_offset,
        CHECKSUM_SIZE,
    )
    header += manifest_sha256
    header += directions_sha256
    header += centroids_sha256
    _require(len(header) <= HEADER_SIZE, "internal V0PQ header overflow")
    header += bytes(HEADER_SIZE - len(header))
    return bytes(header)


def write_v0pq(
    path: Path,
    *,
    dimension: int,
    M: int,
    nbits: int,
    centroids: Iterable[float],
    training_metadata: dict[str, Any],
    source_manifest_sha256: str | bytes,
    source_directions_sha256: str | bytes,
) -> V0PQHeader:
    path = Path(path)
    _require(not path.exists(), f"refusing to overwrite existing codebook: {path}")
    partial = Path(str(path) + ".partial")
    _require(not partial.exists(), f"partial codebook already exists: {partial}")

    ksub = 1 << nbits
    _require(M > 0 and dimension % M == 0, "dimension must be divisible by M")
    dsub = dimension // M
    values = tuple(float(value) for value in centroids)
    _validate_configuration(
        dimension, M, nbits, ksub, dsub, M, len(values)
    )
    _require(all(math.isfinite(value) for value in values), "non-finite centroid")
    _require(isinstance(training_metadata, dict), "training metadata must be an object")

    metadata_bytes = json.dumps(
        training_metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    centroids_bytes = struct.pack(f"<{len(values)}f", *values)
    manifest_digest = parse_sha256(
        source_manifest_sha256, "source_manifest_sha256"
    )
    directions_digest = parse_sha256(
        source_directions_sha256, "source_directions_sha256"
    )
    centroid_digest = _sha256(centroids_bytes)

    metadata_offset = HEADER_SIZE
    centroids_offset = metadata_offset + len(metadata_bytes)
    checksums_offset = centroids_offset + len(centroids_bytes)
    header_bytes = _encode_header(
        dimension=dimension,
        M=M,
        nbits=nbits,
        ksub=ksub,
        dsub=dsub,
        centroid_count=len(values),
        metadata_offset=metadata_offset,
        metadata_size=len(metadata_bytes),
        centroids_offset=centroids_offset,
        centroids_size=len(centroids_bytes),
        checksums_offset=checksums_offset,
        manifest_sha256=manifest_digest,
        directions_sha256=directions_digest,
        centroids_sha256=centroid_digest,
    )

    checksum_bytes = bytearray()
    checksum_bytes += CHECKSUM_MAGIC
    checksum_bytes += struct.pack("<II", VERSION, CHECKSUM_COUNT)
    checksum_bytes += _sha256(header_bytes)
    checksum_bytes += _sha256(metadata_bytes)
    checksum_bytes += centroid_digest
    _require(len(checksum_bytes) == CHECKSUM_SIZE, "internal checksum size mismatch")

    try:
        with partial.open("xb") as handle:
            handle.write(header_bytes)
            handle.write(metadata_bytes)
            handle.write(centroids_bytes)
            handle.write(checksum_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        partial.replace(path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return read_v0pq(path).header


def _section(data: bytes, offset: int, size: int, name: str) -> bytes:
    _require(offset <= len(data), f"{name} offset is out of bounds")
    _require(size <= len(data) - offset, f"{name} size is out of bounds")
    return data[offset : offset + size]


def read_v0pq(path: Path) -> V0PQCodebook:
    path = Path(path)
    data = path.read_bytes()
    _require(len(data) >= HEADER_SIZE, "V0PQ file is shorter than its header")
    _require(data[:8] == MAGIC, "invalid V0PQ magic")
    version, header_size = struct.unpack_from("<II", data, 8)
    _require(version == VERSION, "unsupported V0PQ version")
    _require(header_size == HEADER_SIZE, "unsupported V0PQ header size")
    _require(data[16:20] == bytes((1, 1, 1, 0)), "unsupported representation")

    dimension, M, nbits, ksub, dsub, code_size = struct.unpack_from(
        "<6I", data, 20
    )
    centroid_count = struct.unpack_from("<Q", data, 44)[0]
    (
        metadata_offset,
        metadata_size,
        centroids_offset,
        centroids_size,
        checksums_offset,
        checksums_size,
    ) = struct.unpack_from("<6Q", data, 52)
    manifest_digest = data[100:132]
    directions_digest = data[132:164]
    stored_centroid_digest = data[164:196]
    _require(data[196:HEADER_SIZE] == bytes(HEADER_SIZE - 196), "reserved bytes")

    _validate_configuration(
        dimension, M, nbits, ksub, dsub, code_size, centroid_count
    )
    _require(centroids_size == centroid_count * 4, "centroid section size mismatch")
    _require(checksums_size == CHECKSUM_SIZE, "checksum section size mismatch")
    expected = HEADER_SIZE
    _require(metadata_offset == expected, "non-canonical metadata offset")
    expected += metadata_size
    _require(centroids_offset == expected, "non-canonical centroid offset")
    expected += centroids_size
    _require(checksums_offset == expected, "non-canonical checksum offset")
    expected += checksums_size
    _require(expected == len(data), "file size does not match canonical layout")

    metadata_bytes = _section(data, metadata_offset, metadata_size, "metadata")
    centroids_bytes = _section(data, centroids_offset, centroids_size, "centroids")
    checksums = _section(data, checksums_offset, checksums_size, "checksums")
    _require(checksums[:8] == CHECKSUM_MAGIC, "invalid checksum magic")
    checksum_version, checksum_count = struct.unpack_from("<II", checksums, 8)
    _require(checksum_version == VERSION, "invalid checksum version")
    _require(checksum_count == CHECKSUM_COUNT, "invalid checksum count")
    _require(checksums[16:48] == _sha256(data[:HEADER_SIZE]), "header checksum")
    _require(checksums[48:80] == _sha256(metadata_bytes), "metadata checksum")
    actual_centroid_digest = _sha256(centroids_bytes)
    _require(checksums[80:112] == actual_centroid_digest, "centroid checksum")
    _require(stored_centroid_digest == actual_centroid_digest, "stored centroid hash")

    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V0PQFormatError(f"invalid training metadata: {exc}") from exc
    _require(isinstance(metadata, dict), "training metadata must be a JSON object")
    centroids = struct.unpack(f"<{centroid_count}f", centroids_bytes)
    _require(all(math.isfinite(value) for value in centroids), "non-finite centroid")

    header = V0PQHeader(
        dimension=dimension,
        M=M,
        nbits=nbits,
        ksub=ksub,
        dsub=dsub,
        code_size=code_size,
        centroid_count=centroid_count,
        training_metadata_offset=metadata_offset,
        training_metadata_size=metadata_size,
        centroids_offset=centroids_offset,
        centroids_size=centroids_size,
        checksums_offset=checksums_offset,
        checksums_size=checksums_size,
        source_manifest_sha256=manifest_digest.hex(),
        source_directions_sha256=directions_digest.hex(),
        centroids_sha256=stored_centroid_digest.hex(),
    )
    return V0PQCodebook(header=header, training_metadata=metadata, centroids=centroids)
