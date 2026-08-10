"""Small, auditable PQ/OPQ implementation for ETV1 Gate B.

It is not the eventual online encoder.  The purpose is to compare transforms
under one code budget before any hnswlib control-flow work is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def _check_training_matrix(vectors: np.ndarray, m: int, ksub: int) -> np.ndarray:
    data = np.asarray(vectors, dtype=np.float32)
    if data.ndim != 2 or data.shape[0] < ksub:
        raise ValueError("training matrix must be 2D with at least ksub rows")
    if data.shape[1] % m:
        raise ValueError("dimension must be divisible by m")
    if not np.all(np.isfinite(data)):
        raise ValueError("training matrix contains non-finite values")
    return data


@dataclass(frozen=True)
class ProductQuantizerModel:
    transform: np.ndarray
    codebooks: np.ndarray
    method: str
    seed: int

    @property
    def m(self) -> int:
        return int(self.codebooks.shape[0])

    @property
    def ksub(self) -> int:
        return int(self.codebooks.shape[1])

    @property
    def dsub(self) -> int:
        return int(self.codebooks.shape[2])

    @property
    def dimension(self) -> int:
        return self.m * self.dsub

    def transform_vectors(self, vectors: np.ndarray) -> np.ndarray:
        data = np.asarray(vectors, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != self.dimension:
            raise ValueError("vectors have the wrong shape")
        return np.asarray(data @ self.transform.T, dtype=np.float32)

    def reconstruct_transformed(self, original_vectors: np.ndarray) -> np.ndarray:
        transformed = self.transform_vectors(original_vectors)
        reconstruction = np.empty_like(transformed)
        for block in range(self.m):
            sl = slice(block * self.dsub, (block + 1) * self.dsub)
            values = transformed[:, sl]
            centroids = self.codebooks[block]
            distances = (
                np.sum(values * values, axis=1, keepdims=True)
                + np.sum(centroids * centroids, axis=1)[None, :]
                - 2.0 * values @ centroids.T
            )
            codes = np.argmin(distances, axis=1)
            reconstruction[:, sl] = centroids[codes]
        return reconstruction

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            transform=self.transform.astype(np.float64),
            codebooks=self.codebooks.astype(np.float32),
            method=np.asarray(self.method),
            seed=np.asarray(self.seed, dtype=np.int64),
        )


def _fit_codebooks(
    transformed: np.ndarray,
    m: int,
    ksub: int,
    seed: int,
    *,
    sample_weight: np.ndarray | None,
    max_iter: int,
    batch_size: int,
) -> np.ndarray:
    data = _check_training_matrix(transformed, m, ksub)
    dsub = data.shape[1] // m
    codebooks = np.empty((m, ksub, dsub), dtype=np.float32)
    for block in range(m):
        sl = slice(block * dsub, (block + 1) * dsub)
        estimator = MiniBatchKMeans(
            n_clusters=ksub,
            init="k-means++",
            n_init=1,
            max_iter=max_iter,
            batch_size=min(batch_size, data.shape[0]),
            random_state=seed + block,
            reassignment_ratio=0.01,
        )
        estimator.fit(data[:, sl], sample_weight=sample_weight)
        codebooks[block] = estimator.cluster_centers_.astype(np.float32)
    return codebooks


def fit_pq(
    vectors: np.ndarray,
    *,
    m: int,
    ksub: int = 256,
    seed: int = 20260810,
    transform: np.ndarray | None = None,
    sample_weight: Sequence[float] | None = None,
    method: str = "identity_pq",
    max_iter: int = 50,
    batch_size: int = 4096,
) -> ProductQuantizerModel:
    data = _check_training_matrix(vectors, m, ksub)
    matrix = np.eye(data.shape[1], dtype=np.float64) if transform is None else np.asarray(transform, dtype=np.float64)
    if matrix.shape != (data.shape[1], data.shape[1]):
        raise ValueError("transform has the wrong shape")
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if weights is not None:
        if weights.shape != (data.shape[0],) or np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
            raise ValueError("sample weights are invalid")
        if not np.any(weights > 0.0):
            raise ValueError("at least one sample weight must be positive")
        weights = weights / np.mean(weights)
    transformed = np.asarray(data @ matrix.T, dtype=np.float32)
    codebooks = _fit_codebooks(
        transformed,
        m,
        ksub,
        seed,
        sample_weight=weights,
        max_iter=max_iter,
        batch_size=batch_size,
    )
    return ProductQuantizerModel(matrix, codebooks, method, seed)


def random_orthogonal(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gaussian = rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return np.asarray(q * signs[None, :], dtype=np.float64)


def fit_opq(
    vectors: np.ndarray,
    *,
    m: int,
    ksub: int = 256,
    seed: int = 20260810,
    iterations: int = 3,
    sample_weight: Sequence[float] | None = None,
    method: str = "edge_opq",
    max_iter: int = 40,
    batch_size: int = 4096,
) -> ProductQuantizerModel:
    data = _check_training_matrix(vectors, m, ksub)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if weights is not None:
        if weights.shape != (data.shape[0],) or np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
            raise ValueError("sample weights are invalid")
        weights = weights / np.mean(weights)
    transform = np.eye(data.shape[1], dtype=np.float64)
    model: ProductQuantizerModel | None = None
    for iteration in range(iterations):
        model = fit_pq(
            data,
            m=m,
            ksub=ksub,
            seed=seed + iteration * 1000,
            transform=transform,
            sample_weight=weights,
            method=method,
            max_iter=max_iter,
            batch_size=batch_size,
        )
        reconstruction = model.reconstruct_transformed(data).astype(np.float64)
        weighted_reconstruction = reconstruction if weights is None else reconstruction * weights[:, None]
        cross = data.astype(np.float64).T @ weighted_reconstruction
        left, _, right_t = np.linalg.svd(cross, full_matrices=False)
        # Row convention: Y = X R^T.  Procrustes Q=R^T=U V^T.
        transform = np.asarray((left @ right_t).T, dtype=np.float64)
    return fit_pq(
        data,
        m=m,
        ksub=ksub,
        seed=seed + iterations * 1000,
        transform=transform,
        sample_weight=weights,
        method=method,
        max_iter=max_iter,
        batch_size=batch_size,
    )

