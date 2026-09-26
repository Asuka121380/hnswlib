from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Encoder = Callable[[np.ndarray], np.ndarray]


@dataclass
class ProductCodebookModel:
    backend: str
    implementation: str
    provider: str
    codebooks: np.ndarray
    nbits: int
    encoder: Encoder
    rotation: np.ndarray | None = None
    dependency_versions: dict[str, str] | None = None

    @property
    def m(self) -> int:
        return int(self.codebooks.shape[0])

    @property
    def dimension(self) -> int:
        return self.m * int(self.codebooks.shape[2])

    def transform(self, values: np.ndarray) -> np.ndarray:
        source = np.ascontiguousarray(values, dtype=np.float32)
        if self.rotation is None:
            return source
        return np.ascontiguousarray(source @ self.rotation.T, dtype=np.float32)

    def encode_batch(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        transformed = self.transform(values)
        codes = np.asarray(self.encoder(transformed), dtype=np.uint16)
        if codes.shape != (transformed.shape[0], self.m):
            raise RuntimeError("trainer returned an invalid product-code shape")
        if np.any(codes >= (1 << self.nbits)):
            raise RuntimeError("trainer returned a code outside the configured bit width")
        reconstructed = np.empty_like(transformed, dtype=np.float32)
        dsub = self.dimension // self.m
        rows = np.arange(transformed.shape[0])
        for sub in range(self.m):
            reconstructed[:, sub * dsub:(sub + 1) * dsub] = self.codebooks[sub, codes[:, sub]]
        return codes, transformed, reconstructed
