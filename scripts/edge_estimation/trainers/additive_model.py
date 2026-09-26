from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ProductResidualModel:
    backend: str
    implementation: str
    provider: str
    codebooks: np.ndarray
    nbits: int
    nsplits: int
    stages_per_split: int
    encoder: Callable[[np.ndarray], np.ndarray]
    rotation: None = None
    dependency_versions: dict[str, str] | None = None

    @property
    def m(self) -> int:
        return self.nsplits * self.stages_per_split

    @property
    def dimension(self) -> int:
        return self.nsplits * int(self.codebooks.shape[2])

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(values, dtype=np.float32)

    def encode_batch(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        transformed = self.transform(values)
        codes = np.asarray(self.encoder(transformed), dtype=np.uint16)
        if codes.shape != (transformed.shape[0], self.m):
            raise RuntimeError("PRQ trainer returned an invalid code shape")
        reconstructed = np.zeros_like(transformed, dtype=np.float32)
        dsub = self.dimension // self.nsplits
        for split in range(self.nsplits):
            destination = reconstructed[:, split * dsub:(split + 1) * dsub]
            for stage in range(self.stages_per_split):
                index = split * self.stages_per_split + stage
                destination += self.codebooks[index, codes[:, index]]
        return codes, transformed, reconstructed
