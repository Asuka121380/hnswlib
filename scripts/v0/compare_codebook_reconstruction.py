#!/usr/bin/env python3
"""Check custom V0PQ encode/decode and optional Faiss equivalence."""

from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import struct
from pathlib import Path
from typing import Iterable

from v0pq_format import V0PQCodebook, read_v0pq
from validate_edge_direction_sample import validate


def _row_indices(count: int, requested: int) -> list[int]:
    if requested >= count:
        return list(range(count))
    if requested == 1:
        return [0]
    return [(i * (count - 1)) // (requested - 1) for i in range(requested)]


def _encode(codebook: V0PQCodebook, vector: tuple[float, ...]) -> bytes:
    h = codebook.header
    output = bytearray(h.M)
    for m in range(h.M):
        begin = m * h.dsub
        best_code = 0
        best_distance = math.inf
        for code in range(h.ksub):
            distance = 0.0
            for coordinate in range(h.dsub):
                delta = vector[begin + coordinate] - codebook.centroid(m, code, coordinate)
                distance += delta * delta
            if distance < best_distance:
                best_code, best_distance = code, distance
        output[m] = best_code
    return bytes(output)


def _decode(codebook: V0PQCodebook, codes: bytes) -> tuple[float, ...]:
    h = codebook.header
    values: list[float] = []
    for m, code in enumerate(codes):
        values.extend(codebook.centroid(m, code, c) for c in range(h.dsub))
    return tuple(values)


def _dot(left: Iterable[float], right: Iterable[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right))


def _faiss_check(codebook: V0PQCodebook, vectors: list[tuple[float, ...]]) -> float:
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("--faiss-check requires NumPy and faiss-cpu") from exc
    h = codebook.header
    pq = faiss.ProductQuantizer(h.dimension, h.M, h.nbits)
    faiss.copy_array_to_vector(np.asarray(codebook.centroids, dtype=np.float32), pq.centroids)
    matrix = np.asarray(vectors, dtype=np.float32)
    faiss_decoded = pq.decode(pq.compute_codes(matrix))
    maximum = 0.0
    for vector, decoded in zip(vectors, faiss_decoded):
        custom = _decode(codebook, _encode(codebook, vector))
        maximum = max(maximum, max(abs(a - float(b)) for a, b in zip(custom, decoded)))
    return maximum


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample-rows", type=int, default=256)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--faiss-check", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    try:
        if args.output_metrics.exists():
            raise RuntimeError(f"refusing to overwrite output: {args.output_metrics}")
        sample = validate(args.manifest, directions_override=args.directions)
        codebook = read_v0pq(args.codebook)
        if sample["dimension"] != codebook.header.dimension:
            raise RuntimeError("sample and codebook dimensions differ")
        if sample["directions_sha256"] != codebook.header.source_directions_sha256:
            raise RuntimeError("sample and codebook direction digests differ")
        count = int(sample["sample_count"])
        if count <= 0:
            raise RuntimeError("reconstruction check requires a non-empty sample")
        if args.sample_rows <= 0:
            raise RuntimeError("sample-rows must be positive")
        if args.tolerance < 0.0:
            raise RuntimeError("tolerance must be non-negative")
        indices = _row_indices(count, min(args.sample_rows, count))
        row_size = codebook.header.dimension * 4
        vectors: list[tuple[float, ...]] = []
        errors: list[float] = []
        max_lut_difference = 0.0
        with args.directions.open("rb") as handle:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                for row in indices:
                    vector = struct.unpack_from(
                        f"<{codebook.header.dimension}f", mapped, row * row_size
                    )
                    codes = _encode(codebook, vector)
                    reconstructed = _decode(codebook, codes)
                    errors.append(math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(vector, reconstructed))))
                    lut_sum = 0.0
                    for m, code in enumerate(codes):
                        begin = m * codebook.header.dsub
                        centroid = (
                            codebook.centroid(m, code, c)
                            for c in range(codebook.header.dsub)
                        )
                        lut_sum += _dot(vector[begin : begin + codebook.header.dsub], centroid)
                    max_lut_difference = max(max_lut_difference, abs(lut_sum - _dot(vector, reconstructed)))
                    vectors.append(vector)
        faiss_difference = _faiss_check(codebook, vectors) if args.faiss_check else None
        if max_lut_difference > args.tolerance:
            raise RuntimeError(
                f"custom LUT identity difference {max_lut_difference} exceeds tolerance"
            )
        if faiss_difference is not None and faiss_difference > args.tolerance:
            raise RuntimeError(
                f"Faiss/custom decode difference {faiss_difference} exceeds tolerance"
            )
        metrics = {
            "format": "hnswlib_v0_codebook_reconstruction_check",
            "format_version": 1,
            "status": "valid",
            "checked_rows": len(indices),
            "dimension": codebook.header.dimension,
            "M_pq": codebook.header.M,
            "nbits": codebook.header.nbits,
            "mean_reconstruction_error": math.fsum(errors) / len(errors),
            "max_reconstruction_error": max(errors),
            "max_custom_lut_identity_difference": max_lut_difference,
            "max_faiss_decode_difference": faiss_difference,
            "faiss_check_performed": args.faiss_check,
            "acceptance_tolerance": args.tolerance,
        }
        partial = Path(str(args.output_metrics) + ".partial")
        args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
        with partial.open("x", encoding="utf-8", newline="\n") as output:
            json.dump(metrics, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        partial.replace(args.output_metrics)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
