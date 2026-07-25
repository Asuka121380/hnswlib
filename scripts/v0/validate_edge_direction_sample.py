#!/usr/bin/env python3
"""Validate a V0 edge-direction sample manifest and raw float32 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import struct
from pathlib import Path
from typing import Any


class ValidationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _resolve_directions(
    manifest_path: Path,
    manifest: dict[str, Any],
    override: Path | None,
) -> Path:
    if override is not None:
        return override
    configured = Path(manifest["directions_path"])
    if configured.is_absolute() or configured.exists():
        return configured
    beside_manifest = manifest_path.parent / configured
    if beside_manifest.exists():
        return beside_manifest
    return configured


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def validate(
    manifest_path: Path,
    directions_override: Path | None = None,
    norm_tolerance: float = 5e-5,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        manifest.get("format") == "hnswlib_v0_edge_direction_sample",
        "unexpected sample format",
    )
    _require(manifest.get("format_version") == 1, "unsupported format version")
    _require(
        manifest.get("sampling_algorithm")
        == "reservoir_algorithm_r_splitmix64_v1",
        "unexpected sampling algorithm",
    )
    _require(
        manifest.get("traversal_order") == "source_id_then_layer0_slot",
        "unexpected graph traversal order",
    )
    _require(
        manifest.get("sampling_population")
        == "all_nonzero_layer0_directed_edges",
        "unexpected sampling population",
    )
    _require(manifest.get("metric") == "squared_l2", "unexpected metric")
    _require(
        manifest.get("sampled_object") == "unit_edge_direction",
        "unexpected sampled object",
    )
    _require(
        manifest.get("vector_source") == "embedded_hnsw_level0_data",
        "unexpected vector source",
    )
    _require(manifest.get("dtype") == "float32", "unexpected sample dtype")
    _require(manifest.get("byte_order") == "little", "unexpected byte order")
    _require(manifest.get("layout") == "row_major", "unexpected matrix layout")

    dimension = int(manifest["dimension"])
    node_count = int(manifest["node_count"])
    requested = int(manifest["requested_sample_count"])
    produced = int(manifest["produced_sample_count"])
    directed_edges = int(manifest["directed_edge_count"])
    valid_edges = int(manifest["valid_edge_count"])
    zero_edges = int(manifest["zero_length_edge_count"])
    declared_bytes = int(manifest["directions_bytes"])
    _require(dimension > 0, "dimension must be positive")
    _require(node_count >= 0, "node count must be non-negative")
    _require(requested > 0, "requested sample count must be positive")
    _require(produced == min(requested, valid_edges), "sample count mismatch")
    _require(directed_edges == valid_edges + zero_edges, "edge accounting mismatch")
    _require(declared_bytes == produced * dimension * 4, "byte count mismatch")

    directions_path = _resolve_directions(
        manifest_path, manifest, directions_override
    )
    _require(directions_path.is_file(), "direction matrix does not exist")
    _require(directions_path.stat().st_size == declared_bytes, "file size mismatch")
    _require(
        _file_sha256(directions_path) == manifest["directions_sha256"],
        "direction matrix SHA-256 mismatch",
    )

    min_norm = math.inf
    max_norm = 0.0
    if declared_bytes:
        with directions_path.open("rb") as handle:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                coordinate = 0
                squared_norm = 0.0
                row_count = 0
                for (value,) in struct.iter_unpack("<f", mapped):
                    _require(math.isfinite(value), "non-finite sampled direction")
                    squared_norm += value * value
                    coordinate += 1
                    if coordinate == dimension:
                        norm = math.sqrt(squared_norm)
                        _require(
                            abs(norm - 1.0) <= norm_tolerance,
                            f"sampled direction norm {norm} exceeds tolerance",
                        )
                        min_norm = min(min_norm, norm)
                        max_norm = max(max_norm, norm)
                        coordinate = 0
                        squared_norm = 0.0
                        row_count += 1
                _require(coordinate == 0, "partial direction row")
                _require(row_count == produced, "direction row count mismatch")
    else:
        min_norm = 0.0

    return {
        "status": "valid",
        "manifest": str(manifest_path),
        "directions": str(directions_path),
        "dimension": dimension,
        "node_count": node_count,
        "sample_count": produced,
        "directed_edge_count": directed_edges,
        "valid_edge_count": valid_edges,
        "zero_length_edge_count": zero_edges,
        "min_direction_norm": min_norm,
        "max_direction_norm": max_norm,
        "directions_sha256": manifest["directions_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--directions", type=Path)
    parser.add_argument("--norm-tolerance", type=float, default=5e-5)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    try:
        summary = validate(
            args.manifest,
            directions_override=args.directions,
            norm_tolerance=args.norm_tolerance,
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError, ValidationError) as exc:
        parser.error(str(exc))

    if args.as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "valid "
            f"samples={summary['sample_count']} "
            f"dimension={summary['dimension']} "
            f"edges={summary['directed_edge_count']} "
            f"sha256={summary['directions_sha256']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
