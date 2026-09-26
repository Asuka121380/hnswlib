from __future__ import annotations

import numpy as np

from .product_model import ProductCodebookModel


def _kmeans(values: np.ndarray, k: int, iterations: int, seed: int) -> np.ndarray:
    if values.shape[0] < k:
        raise ValueError(f"training sample {values.shape[0]} is smaller than centroid count {k}")
    rng = np.random.default_rng(seed)
    centers = values[rng.choice(values.shape[0], k, replace=False)].copy()
    for _ in range(iterations):
        distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assignment = distances.argmin(axis=1)
        for index in range(k):
            members = values[assignment == index]
            if members.size:
                centers[index] = members.mean(axis=0)
    return np.ascontiguousarray(centers, dtype="<f4")


def train(sample: np.ndarray, m: int, nbits: int, trainer: dict) -> ProductCodebookModel:
    dimension = int(sample.shape[1])
    if dimension % m:
        raise ValueError("dimension must be divisible by m")
    dsub, ksub = dimension // m, 1 << nbits
    iterations = int(trainer.get("iterations", 12))
    seed = int(trainer.get("seed", 20260924))
    codebooks = np.empty((m, ksub, dsub), dtype="<f4")
    for sub in range(m):
        codebooks[sub] = _kmeans(
            sample[:, sub * dsub:(sub + 1) * dsub], ksub, iterations, seed + sub)

    def encode(values: np.ndarray) -> np.ndarray:
        codes = np.empty((values.shape[0], m), dtype=np.uint16)
        for sub in range(m):
            part = values[:, sub * dsub:(sub + 1) * dsub]
            distances = ((part[:, None, :] - codebooks[sub][None, :, :]) ** 2).sum(axis=2)
            codes[:, sub] = distances.argmin(axis=1)
        return codes

    return ProductCodebookModel(
        backend="pq_packed", implementation="numpy_reference_pq",
        provider="numpy_reference", codebooks=codebooks, nbits=nbits,
        encoder=encode, dependency_versions={"numpy": np.__version__})
