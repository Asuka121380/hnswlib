#!/usr/bin/env python3
"""Independently validate and summarize a V0PQ codebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v0pq_format import V0PQFormatError, read_v0pq, sha256_file


REQUIRED_METADATA = {
    "quantizer_name",
    "faiss_version",
    "faiss_backend",
    "dimension",
    "M_pq",
    "nbits",
    "training_seed",
    "split_seed",
    "split_algorithm",
    "iterations",
    "nredo",
    "min_points_per_centroid",
    "max_points_per_centroid",
    "training_count",
    "validation_count",
    "source_manifest_sha256",
    "source_directions_sha256",
    "centroid_layout",
    "creation_timestamp_utc",
    "producer_git_commit",
}


def validate_codebook(
    codebook_path: Path,
    expected_manifest: Path | None = None,
    expected_directions: Path | None = None,
) -> dict[str, object]:
    codebook = read_v0pq(codebook_path)
    header = codebook.header
    metadata = codebook.training_metadata
    missing = sorted(REQUIRED_METADATA - set(metadata))
    if missing:
        raise V0PQFormatError(f"missing training metadata: {', '.join(missing)}")
    if metadata["dimension"] != header.dimension:
        raise V0PQFormatError("metadata dimension disagrees with header")
    if metadata["M_pq"] != header.M or metadata["nbits"] != header.nbits:
        raise V0PQFormatError("metadata PQ configuration disagrees with header")
    if metadata["quantizer_name"] != "faiss_product_quantizer":
        raise V0PQFormatError("codebook was not produced by the V0 Faiss PQ trainer")
    if metadata["source_manifest_sha256"] != header.source_manifest_sha256:
        raise V0PQFormatError("metadata manifest digest disagrees with header")
    if metadata["source_directions_sha256"] != header.source_directions_sha256:
        raise V0PQFormatError("metadata directions digest disagrees with header")
    if expected_manifest and sha256_file(expected_manifest) != header.source_manifest_sha256:
        raise V0PQFormatError("expected manifest SHA-256 mismatch")
    if expected_directions and sha256_file(expected_directions) != header.source_directions_sha256:
        raise V0PQFormatError("expected directions SHA-256 mismatch")
    return {
        "status": "valid",
        "path": str(codebook_path),
        "file_sha256": sha256_file(codebook_path),
        "dimension": header.dimension,
        "M_pq": header.M,
        "nbits": header.nbits,
        "ksub": header.ksub,
        "dsub": header.dsub,
        "code_size": header.code_size,
        "centroid_count": header.centroid_count,
        "codebook_bytes": codebook_path.stat().st_size,
        "source_manifest_sha256": header.source_manifest_sha256,
        "source_directions_sha256": header.source_directions_sha256,
        "centroids_sha256": header.centroids_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--expected-manifest", type=Path)
    parser.add_argument("--expected-directions", type=Path)
    args = parser.parse_args()
    try:
        result = validate_codebook(
            args.codebook, args.expected_manifest, args.expected_directions
        )
    except (OSError, ValueError, V0PQFormatError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
