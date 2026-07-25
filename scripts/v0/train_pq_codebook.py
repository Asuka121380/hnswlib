#!/usr/bin/env python3
"""Train a Faiss ProductQuantizer and export the stable V0PQ format."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from v0pq_format import sha256_file, write_v0pq
from validate_edge_direction_sample import ValidationError, validate


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _refuse_existing(*paths: Path | None) -> None:
    for path in paths:
        if path is not None and path.exists():
            raise RuntimeError(f"refusing to overwrite existing output: {path}")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    partial = Path(str(path) + ".partial")
    _require(not partial.exists(), f"partial output already exists: {partial}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        partial.replace(path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _input_contract(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    sample = validate(args.manifest, directions_override=args.directions)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    dimension = int(sample["dimension"])
    count = int(sample["sample_count"])
    _require(dimension % args.M_pq == 0, "sample dimension must be divisible by M-pq")
    _require(args.nbits == 8, "V0 pilot contract fixes nbits=8")
    _require(0.0 < args.validation_fraction < 1.0, "validation fraction must be in (0,1)")
    validation_count = max(1, int(count * args.validation_fraction))
    training_count = count - validation_count
    _require(training_count > 0, "sample must leave at least one training row")
    _require(
        training_count >= (1 << args.nbits) * args.min_points_per_centroid,
        "training split is too small for min-points-per-centroid",
    )
    manifest_sha = sha256_file(args.manifest)
    _require(
        manifest["directions_sha256"] == sample["directions_sha256"],
        "manifest direction digest disagrees with validated sample",
    )
    return sample, {
        "format": "hnswlib_v0_pq_training_contract",
        "format_version": 1,
        "dimension": dimension,
        "sample_count": count,
        "training_count": training_count,
        "validation_count": validation_count,
        "M_pq": args.M_pq,
        "nbits": args.nbits,
        "ksub": 1 << args.nbits,
        "dsub": dimension // args.M_pq,
        "code_size": args.M_pq,
        "training_seed": args.seed,
        "split_seed": args.split_seed,
        "split_algorithm": "numpy_default_rng_permutation_v1",
        "iterations": args.iterations,
        "nredo": args.nredo,
        "min_points_per_centroid": args.min_points_per_centroid,
        "max_points_per_centroid": args.max_points_per_centroid,
        "threads": args.threads,
        "validation_fraction": args.validation_fraction,
        "source_manifest_path": str(args.manifest.resolve()),
        "source_directions_path": str(args.directions.resolve()),
        "source_manifest_sha256": manifest_sha,
        "source_directions_sha256": sample["directions_sha256"],
        "source_index_sha256": manifest.get("index_sha256"),
        "source_adjacency_sha256": manifest.get("adjacency_sha256"),
        "sampling_algorithm": manifest["sampling_algorithm"],
        "sampling_seed": manifest.get("seed"),
    }


def _error_summary(np: Any, errors: Any, residuals: Any) -> dict[str, float]:
    return {
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "p90_error": float(np.quantile(errors, 0.90)),
        "p95_error": float(np.quantile(errors, 0.95)),
        "p99_error": float(np.quantile(errors, 0.99)),
        "max_error": float(np.max(errors)),
        "reconstruction_mse": float(np.mean(residuals * residuals)),
    }


def _train(args: argparse.Namespace, contract: dict[str, Any]) -> dict[str, Any]:
    try:
        import numpy as np
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "real PQ training requires NumPy and faiss-cpu; "
            "use --check-input-only on a local non-Faiss environment"
        ) from exc

    dimension = contract["dimension"]
    count = contract["sample_count"]
    matrix = np.memmap(
        args.directions, dtype="<f4", mode="r", shape=(count, dimension)
    )
    rng = np.random.default_rng(args.split_seed)
    permutation = rng.permutation(count)
    validation_indices = permutation[: contract["validation_count"]]
    training_indices = permutation[contract["validation_count"] :]
    training = np.ascontiguousarray(matrix[training_indices], dtype=np.float32)
    validation = np.ascontiguousarray(matrix[validation_indices], dtype=np.float32)

    faiss.omp_set_num_threads(args.threads)
    pq = faiss.ProductQuantizer(dimension, args.M_pq, args.nbits)
    pq.cp.seed = args.seed
    pq.cp.niter = args.iterations
    pq.cp.nredo = args.nredo
    pq.cp.min_points_per_centroid = args.min_points_per_centroid
    pq.cp.max_points_per_centroid = args.max_points_per_centroid

    started = time.perf_counter()
    pq.train(training)
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    codes = pq.compute_codes(validation)
    reconstructed = pq.decode(codes)
    validation_seconds = time.perf_counter() - started
    residuals = validation.astype(np.float64) - reconstructed.astype(np.float64)
    errors = np.linalg.norm(residuals, axis=1)
    centroids = faiss.vector_to_array(pq.centroids).astype("<f4", copy=False)

    errors_path = args.output_errors
    if errors_path is not None:
        partial = Path(str(errors_path) + ".partial")
        errors_path.parent.mkdir(parents=True, exist_ok=True)
        _require(not partial.exists(), f"partial output already exists: {partial}")
        with partial.open("xb") as handle:
            handle.write(errors.astype("<f4", copy=False).tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        partial.replace(errors_path)

    metadata = dict(contract)
    metadata.update(
        {
            "quantizer_name": "faiss_product_quantizer",
            "faiss_version": getattr(faiss, "__version__", "unknown"),
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "faiss_backend": "cpu",
            "centroid_layout": "M_ksub_dsub_row_major_float32",
            "creation_timestamp_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "producer_git_commit": args.producer_git_commit,
        }
    )
    write_v0pq(
        args.output_codebook,
        dimension=dimension,
        M=args.M_pq,
        nbits=args.nbits,
        centroids=centroids,
        training_metadata=metadata,
        source_manifest_sha256=contract["source_manifest_sha256"],
        source_directions_sha256=contract["source_directions_sha256"],
    )
    metrics = {
        "format": "hnswlib_v0_pq_training_metrics",
        "format_version": 1,
        "status": "trained",
        **metadata,
        **_error_summary(np, errors, residuals),
        "training_seconds": training_seconds,
        "validation_seconds": validation_seconds,
        "codebook_path": str(args.output_codebook.resolve()),
        "codebook_sha256": sha256_file(args.output_codebook),
        "validation_errors_path": (
            str(errors_path.resolve()) if errors_path is not None else None
        ),
        "validation_errors_sha256": (
            sha256_file(errors_path) if errors_path is not None else None
        ),
        "codebook_bytes": args.output_codebook.stat().st_size,
        "per_vector_code_bytes": args.M_pq,
        "per_query_lut_bytes_float32": args.M_pq * (1 << args.nbits) * 4,
    }
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--M-pq", dest="M_pq", type=int, required=True)
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split-seed", type=int, default=29)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--nredo", type=int, default=1)
    parser.add_argument("--min-points-per-centroid", type=int, default=39)
    parser.add_argument("--max-points-per-centroid", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--output-codebook", type=Path)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--output-errors", type=Path)
    parser.add_argument(
        "--producer-git-commit",
        default=os.environ.get("V0_PRODUCER_GIT_COMMIT", "unknown"),
    )
    parser.add_argument("--check-input-only", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        _require(args.M_pq > 0, "M-pq must be positive")
        _require(args.iterations > 0 and args.nredo > 0, "training counts must be positive")
        _require(args.threads > 0, "threads must be positive")
        _refuse_existing(args.output_metrics, args.output_codebook, args.output_errors)
        _, contract = _input_contract(args)
        if args.check_input_only:
            metrics = {"status": "input-valid", **contract}
        else:
            _require(args.output_codebook is not None, "--output-codebook is required")
            metrics = _train(args, contract)
        _write_json_atomic(args.output_metrics, metrics)
    except (OSError, ValueError, KeyError, RuntimeError, ValidationError) as exc:
        parser.error(str(exc))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
