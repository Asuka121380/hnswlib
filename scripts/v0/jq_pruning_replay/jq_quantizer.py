#!/usr/bin/env python3
"""A small, deterministic, primary-level JQ implementation.

The implementation is a behavioral port of the primary level in JHQ commit
1636e197a36871db4a79d66d22f9f9dd0fa51e31.  It deliberately excludes the
Faiss index, exhaustive/IVF search, and all residual JHQ levels.

The public JHQ implementation uses Faiss' Gaussian RNG, LAPACK QR, and
libstdc++'s normal_distribution.  NumPy's MT19937 and QR backends are not
bitwise identical to those components.  Persisted rotation/centroid artifacts
therefore define the reproducible experiment, while tests verify the same
algorithmic construction, shapes, nearest-centroid behavior, and error bounds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


OFFICIAL_JHQ_COMMIT = "1636e197a36871db4a79d66d22f9f9dd0fa51e31"
OFFICIAL_JHQ_URL = "https://github.com/jiabhan/JHQ"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(repr(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    partial = Path(str(path) + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
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


@dataclass(frozen=True)
class JQConfig:
    dimension: int
    subspaces: int
    bits: int = 8
    rotation_seed: int = 1234
    initializer_seed: int = 42
    use_analytical_init: bool = True
    use_kmeans_refinement: bool = False

    def validate(self) -> None:
        _require(self.dimension > 0, "dimension must be positive")
        _require(self.subspaces > 0, "subspaces must be positive")
        _require(
            self.dimension % self.subspaces == 0,
            "dimension must be divisible by subspaces",
        )
        _require(self.bits == 8, "the replay contract fixes primary codes at 8 bits")
        _require(self.use_analytical_init, "analytical initialization is required")
        _require(
            not self.use_kmeans_refinement,
            "k-means refinement is outside the single-level JQ contract",
        )

    @property
    def subvector_dimension(self) -> int:
        return self.dimension // self.subspaces

    @property
    def centroid_count(self) -> int:
        return 1 << self.bits

    @property
    def code_size(self) -> int:
        return self.subspaces


def generate_qr_rotation(dimension: int, seed: int) -> np.ndarray:
    """Generate a deterministic float32 Gaussian QR rotation.

    The Q column signs are canonicalized.  This makes the persisted artifact
    stable when LAPACK returns an otherwise equivalent sign-flipped basis.
    """

    _require(dimension > 0, "dimension must be positive")
    rng = np.random.Generator(np.random.MT19937(seed))
    gaussian = rng.standard_normal((dimension, dimension), dtype=np.float32)
    q, r = np.linalg.qr(gaussian.astype(np.float64), mode="reduced")
    diagonal = np.diag(r)
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    q *= signs[np.newaxis, :]
    result = np.ascontiguousarray(q, dtype=np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("QR rotation contains non-finite values")
    return result


def _official_erfinv_approx(value: np.ndarray) -> np.ndarray:
    """Vector form of jhq_internal::erfinv_approx from the pinned commit."""

    x = np.asarray(value, dtype=np.float32)
    result = np.empty_like(x)
    outside = np.abs(x) >= np.float32(1.0)
    result[outside] = np.where(x[outside] > 0, np.float32(10.0), np.float32(-10.0))
    inside = ~outside
    if np.any(inside):
        y = x[inside]
        a = np.float32(0.147)
        pi = np.float32(math.pi)
        ln1mx2 = np.log(np.float32(1.0) - y * y).astype(np.float32)
        term1 = np.float32(2.0) / (pi * a) + ln1mx2 / np.float32(2.0)
        term2 = ln1mx2 / a
        magnitude = np.sqrt(np.sqrt(term1 * term1 - term2) - term1)
        result[inside] = np.where(y > 0, magnitude, -magnitude)
    return result


def analytical_gaussian_init(
    data: np.ndarray,
    centroid_count: int,
    *,
    seed: int = 42,
    use_jl_transform: bool = True,
) -> np.ndarray:
    """Port IndexJHQ::analytical_gaussian_init for one subspace.

    This follows the pinned source's robust variance, norm-quantile radii,
    random direction Gram-Schmidt, dimension adjustment, and small noise.  The
    RNG backend is intentionally declared in the artifact manifest because it
    is behaviorally, not bitwise, compatible with std::normal_distribution.
    """

    matrix = np.ascontiguousarray(data, dtype=np.float32)
    _require(matrix.ndim == 2, "initializer data must be a 2-D matrix")
    n, dim = matrix.shape
    _require(n > 1, "initializer needs at least two rows")
    _require(dim > 0, "initializer dimension must be positive")
    _require(centroid_count > 1, "centroid_count must be greater than one")

    mean = np.mean(matrix, axis=0, dtype=np.float32).astype(np.float32)
    centered = matrix - mean
    variance = (
        np.sum(centered * centered, axis=0, dtype=np.float32)
        / np.float32(n - 1)
    ).astype(np.float32)

    sorted_variance = np.sort(variance.copy())
    if dim <= 8:
        robust_variance = np.float32(sorted_variance[dim // 2])
    else:
        start = dim // 4
        end = 3 * dim // 4
        trimmed_mean = np.mean(sorted_variance[start:end], dtype=np.float32)
        if use_jl_transform and dim > 32:
            robust_variance = np.float32(
                np.float32(0.8) * sorted_variance[dim // 2]
                + np.float32(0.2) * trimmed_mean
            )
        else:
            robust_variance = np.float32(trimmed_mean)

    std_scale = np.float32(math.sqrt(float(max(robust_variance, np.float32(1e-10)))))
    if not np.isfinite(std_scale) or std_scale <= 0.0:
        raise RuntimeError("analytical initializer has invalid scale")

    k = centroid_count
    if k == 2:
        gaussian_quantiles = np.asarray([-0.6745, 0.6745], dtype=np.float32)
    elif k <= 8:
        table: dict[int, Iterable[float]] = {
            3: (-1.0364, 0.0, 1.0364),
            4: (-1.2816, -0.5244, 0.5244, 1.2816),
            5: (-1.4652, -0.7416, 0.0, 0.7416, 1.4652),
            6: (-1.5982, -0.9082, -0.3584, 0.3584, 0.9082, 1.5982),
            7: (-1.7109, -1.0364, -0.5244, 0.0, 0.5244, 1.0364, 1.7109),
            8: (-1.8119, -1.1503, -0.6745, -0.2533, 0.2533, 0.6745, 1.1503, 1.8119),
        }
        gaussian_quantiles = np.asarray(table[k], dtype=np.float32)
        if use_jl_transform and dim > 64:
            adjustment = min(1.0 + 0.03 * math.log(float(dim) / 64.0), 1.15)
            gaussian_quantiles *= np.float32(adjustment)
    else:
        sample_size = min(n, 2000)
        stride = max(1, n // sample_size)
        # This deliberately matches the pinned source's loop over i <
        # sample_size with step=stride, including its small sample for n>>2000.
        sampled_indices = np.arange(0, sample_size, stride, dtype=np.int64)
        sampled = matrix[sampled_indices].astype(np.float32) - mean
        norms = np.sqrt(
            np.sum(sampled * sampled, axis=1, dtype=np.float32),
            dtype=np.float32,
        )
        norms.sort()
        quantiles = (
            np.arange(1, k + 1, dtype=np.float32) / np.float32(k + 1)
        ).astype(np.float32)
        norm_indices = np.asarray(
            quantiles * np.float32(max(0, norms.size - 1)), dtype=np.int64
        )
        norm_quantiles = norms[norm_indices]
        fallback = np.float32(math.sqrt(2.0)) * _official_erfinv_approx(
            np.float32(2.0) * quantiles - np.float32(1.0)
        )
        gaussian_quantiles = np.where(
            norm_quantiles > std_scale * np.float32(0.1),
            norm_quantiles / std_scale,
            fallback,
        ).astype(np.float32)

    rng = np.random.Generator(np.random.MT19937(seed))
    directions = np.empty((k, dim), dtype=np.float32)
    ortho_epsilon = np.float32(1e-8)
    for index in range(k):
        direction = rng.standard_normal(dim, dtype=np.float32)
        basis_count = min(index, dim)
        for _ in range(2):
            for basis_index in range(basis_count):
                basis = directions[basis_index]
                dot = np.sum(direction * basis, dtype=np.float32)
                direction -= np.float32(dot) * basis
        norm_sq = np.sum(direction * direction, dtype=np.float32)
        if not np.isfinite(norm_sq) or norm_sq < ortho_epsilon:
            direction = rng.standard_normal(dim, dtype=np.float32)
            norm_sq = np.sum(direction * direction, dtype=np.float32)
        direction *= np.float32(1.0 / math.sqrt(float(max(norm_sq, ortho_epsilon))))
        directions[index] = direction

    radii = np.abs(gaussian_quantiles).astype(np.float32) * std_scale
    noise = (
        rng.standard_normal((k, dim), dtype=np.float32)
        * np.float32(std_scale * np.float32(0.01))
    ).astype(np.float32)
    adjustment = np.sqrt(
        variance / np.float32(max(float(robust_variance), 1e-30)),
        dtype=np.float32,
    )
    adjustment = np.clip(adjustment, np.float32(0.7), np.float32(1.5))
    centroids = (
        mean[np.newaxis, :]
        + radii[:, np.newaxis] * directions * adjustment[np.newaxis, :]
        + noise
    ).astype(np.float32)
    if centroids.shape != (k, dim) or not np.isfinite(centroids).all():
        raise RuntimeError("analytical initializer produced invalid centroids")
    return np.ascontiguousarray(centroids)


class JQQuantizer:
    """Single-level primary JQ quantizer for edge directions."""

    def __init__(self, config: JQConfig, rotation: np.ndarray | None = None) -> None:
        config.validate()
        self.config = config
        self.rotation = (
            generate_qr_rotation(config.dimension, config.rotation_seed)
            if rotation is None
            else np.ascontiguousarray(rotation, dtype=np.float32)
        )
        _require(
            self.rotation.shape == (config.dimension, config.dimension),
            "rotation has the wrong shape",
        )
        _require(np.isfinite(self.rotation).all(), "rotation must be finite")
        self.centroids: np.ndarray | None = None
        gram = self.rotation.astype(np.float64).T @ self.rotation.astype(np.float64)
        defect = gram - np.eye(config.dimension, dtype=np.float64)
        self.orthogonality_defect_fro = float(np.linalg.norm(defect, ord="fro"))
        self.rotation_norm_upper = math.sqrt(1.0 + self.orthogonality_defect_fro)

    def rotate(self, vectors: np.ndarray, *, batch_size: int = 4096) -> np.ndarray:
        matrix = np.asarray(vectors)
        _require(matrix.ndim == 2, "vectors must be a 2-D matrix")
        _require(matrix.shape[1] == self.config.dimension, "vector dimension mismatch")
        _require(batch_size > 0, "batch_size must be positive")
        result = np.empty(matrix.shape, dtype=np.float64)
        rotation64 = self.rotation.astype(np.float64)
        for start in range(0, matrix.shape[0], batch_size):
            end = min(matrix.shape[0], start + batch_size)
            result[start:end] = np.asarray(matrix[start:end], dtype=np.float64) @ rotation64.T
        return result

    def fit_rotated(self, rotated_directions: np.ndarray) -> "JQQuantizer":
        matrix = np.asarray(rotated_directions)
        _require(matrix.ndim == 2, "rotated directions must be 2-D")
        _require(matrix.shape[1] == self.config.dimension, "direction dimension mismatch")
        _require(matrix.shape[0] > 1, "at least two directions are required")
        ds = self.config.subvector_dimension
        tables = []
        for subspace in range(self.config.subspaces):
            block = np.ascontiguousarray(
                matrix[:, subspace * ds : (subspace + 1) * ds], dtype=np.float32
            )
            tables.append(
                analytical_gaussian_init(
                    block,
                    self.config.centroid_count,
                    seed=self.config.initializer_seed,
                )
            )
        self.centroids = np.ascontiguousarray(np.stack(tables, axis=0), dtype=np.float32)
        return self

    def _require_trained(self) -> np.ndarray:
        if self.centroids is None:
            raise RuntimeError("JQ quantizer is not fitted")
        return self.centroids

    def encode_rotated(self, vectors: np.ndarray, *, batch_size: int = 2048) -> np.ndarray:
        centroids = self._require_trained()
        matrix = np.asarray(vectors)
        _require(matrix.ndim == 2, "vectors must be 2-D")
        _require(matrix.shape[1] == self.config.dimension, "vector dimension mismatch")
        _require(batch_size > 0, "batch_size must be positive")
        codes = np.empty((matrix.shape[0], self.config.subspaces), dtype=np.uint8)
        ds = self.config.subvector_dimension
        for subspace in range(self.config.subspaces):
            table = centroids[subspace].astype(np.float64)
            centroid_norm = np.sum(table * table, axis=1, dtype=np.float64)
            begin = subspace * ds
            finish = begin + ds
            for start in range(0, matrix.shape[0], batch_size):
                end = min(matrix.shape[0], start + batch_size)
                block = np.asarray(matrix[start:end, begin:finish], dtype=np.float64)
                distance = (
                    np.sum(block * block, axis=1, dtype=np.float64)[:, np.newaxis]
                    + centroid_norm[np.newaxis, :]
                    - 2.0 * (block @ table.T)
                )
                codes[start:end, subspace] = np.argmin(distance, axis=1).astype(np.uint8)
        return codes

    def decode_rotated(self, codes: np.ndarray) -> np.ndarray:
        centroids = self._require_trained()
        values = np.asarray(codes)
        _require(values.ndim == 2, "codes must be 2-D")
        _require(values.shape[1] == self.config.subspaces, "code width mismatch")
        result = np.empty((values.shape[0], self.config.dimension), dtype=np.float32)
        ds = self.config.subvector_dimension
        rows = np.arange(values.shape[0])
        for subspace in range(self.config.subspaces):
            begin = subspace * ds
            result[:, begin : begin + ds] = centroids[subspace][
                values[:, subspace].astype(np.int64)
            ]
        return result

    def direction_error_upper(
        self,
        exact_rotated_directions: np.ndarray,
        codes: np.ndarray,
        *,
        batch_size: int = 4096,
    ) -> np.ndarray:
        matrix = np.asarray(exact_rotated_directions, dtype=np.float64)
        values = np.asarray(codes)
        _require(matrix.shape[0] == values.shape[0], "direction/code count mismatch")
        result = np.empty(matrix.shape[0], dtype=np.float64)
        for start in range(0, matrix.shape[0], batch_size):
            end = min(matrix.shape[0], start + batch_size)
            reconstruction = self.decode_rotated(values[start:end]).astype(np.float64)
            residual = matrix[start:end] - reconstruction
            transformed_error = np.sqrt(
                np.sum(residual * residual, axis=1, dtype=np.float64)
            )
            # If R is not perfectly orthogonal after float32 storage, then
            # ||u - R^T y|| <= ||R^T|| ||Ru-y|| + ||I-R^T R|| ||u||.
            conservative = (
                self.rotation_norm_upper * transformed_error
                + self.orthogonality_defect_fro
            )
            result[start:end] = np.nextafter(conservative, np.inf)
        return result

    def selected_inner_product_upper(
        self,
        rotated_residuals: np.ndarray,
        codes: np.ndarray,
    ) -> np.ndarray:
        """Reproduce the V0 cell-upward and sum-upward LUT semantics."""

        centroids = self._require_trained()
        residuals = np.asarray(rotated_residuals, dtype=np.float64)
        values = np.asarray(codes)
        _require(residuals.ndim == 2, "rotated residuals must be 2-D")
        _require(residuals.shape[0] == values.shape[0], "residual/code count mismatch")
        _require(residuals.shape[1] == self.config.dimension, "residual dimension mismatch")
        _require(values.shape[1] == self.config.subspaces, "code width mismatch")
        ds = self.config.subvector_dimension
        total = np.zeros(residuals.shape[0], dtype=np.float64)
        for subspace in range(self.config.subspaces):
            begin = subspace * ds
            selected = centroids[subspace][values[:, subspace].astype(np.int64)].astype(
                np.float64
            )
            cell = np.zeros(residuals.shape[0], dtype=np.float64)
            block = residuals[:, begin : begin + ds]
            for coordinate in range(ds):
                product = block[:, coordinate] * selected[:, coordinate]
                cell = np.nextafter(cell + product, np.inf)
            cell32 = cell.astype(np.float32)
            rounded_down = cell32.astype(np.float64) < cell
            if np.any(rounded_down):
                cell32[rounded_down] = np.nextafter(
                    cell32[rounded_down], np.float32(np.inf), dtype=np.float32
                )
            total = np.nextafter(total + cell32.astype(np.float64), np.inf)
        return total

    def artifact_manifest(self) -> dict[str, Any]:
        centroids = self._require_trained()
        return {
            "format": "hnswlib_jq_primary_artifact",
            "format_version": 1,
            "config": asdict(self.config),
            "subvector_dimension": self.config.subvector_dimension,
            "centroid_count": self.config.centroid_count,
            "code_size": self.config.code_size,
            "rotation_dtype": str(self.rotation.dtype),
            "centroid_dtype": str(centroids.dtype),
            "rotation_sha256": _sha256_array(self.rotation),
            "centroids_sha256": _sha256_array(centroids),
            "orthogonality_defect_fro": self.orthogonality_defect_fro,
            "rotation_norm_upper": self.rotation_norm_upper,
            "official_jhq_url": OFFICIAL_JHQ_URL,
            "official_jhq_commit": OFFICIAL_JHQ_COMMIT,
            "compatibility": "behavioral_port_not_bitwise_rng_or_lapack_compatible",
            "rng_backend": "numpy.random.MT19937",
            "qr_backend": "numpy.linalg.qr_float64_then_float32",
        }

    def save_artifacts(self, output_dir: Path) -> dict[str, Any]:
        centroids = self._require_trained()
        output_dir = output_dir.resolve()
        if output_dir.exists():
            raise FileExistsError(f"refusing to overwrite artifact directory: {output_dir}")
        output_dir.mkdir(parents=True)
        rotation_path = output_dir / "rotation.npy"
        centroids_path = output_dir / "centroids.npy"
        with rotation_path.open("xb") as handle:
            np.save(handle, self.rotation, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        with centroids_path.open("xb") as handle:
            np.save(handle, centroids, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        manifest = self.artifact_manifest()
        _atomic_json(output_dir / "jq_config.json", manifest)
        return manifest
