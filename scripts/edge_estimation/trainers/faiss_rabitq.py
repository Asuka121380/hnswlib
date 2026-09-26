from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .faiss_pq import _configure_threads, _faiss
from .jq_behavioral_reference import generate_qr_rotation


@dataclass
class RaBitQModel:
    backend: str
    implementation: str
    provider: str
    dimension: int
    rotation: np.ndarray
    quantizer: object
    code_size: int
    dependency_versions: dict[str, str]
    nbits: int = 1
    native_code_bytes: bool = True

    @property
    def m(self) -> int:
        return self.dimension

    @property
    def codebooks(self) -> np.ndarray:
        return np.empty((0,), dtype=np.float32)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(values, dtype=np.float32) @ self.rotation.T,
                                    dtype=np.float32)

    def encode_batch(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        transformed = self.transform(values)
        codes = np.asarray(self.quantizer.compute_codes(transformed), dtype=np.uint8)
        reconstructed = np.asarray(self.quantizer.decode(codes), dtype=np.float32)
        if codes.shape != (values.shape[0], self.code_size):
            raise RuntimeError("Faiss RaBitQ emitted an unexpected code size")
        return codes, transformed, reconstructed


def train(sample: np.ndarray, trainer: dict) -> RaBitQModel:
    faiss = _faiss()
    _configure_threads(faiss, trainer)
    dimension = int(sample.shape[1])
    rotation = generate_qr_rotation(
        dimension, int(trainer.get("rotation_seed", trainer.get("seed", 1234))))
    quantizer = faiss.RaBitQuantizer(dimension, faiss.METRIC_INNER_PRODUCT, 1)
    quantizer.train(np.ascontiguousarray(sample, dtype=np.float32))
    expected = (dimension + 7) // 8 + 8
    if int(quantizer.code_size) != expected:
        raise RuntimeError("Faiss RaBitQ 1-bit code size contract changed")
    model = RaBitQModel(
        backend="rabitq", implementation="faiss_rabitq_1bit_qb0_zero_centroid",
        provider="faiss", dimension=dimension,
        rotation=np.ascontiguousarray(rotation, dtype="<f4"), quantizer=quantizer,
        code_size=expected,
        dependency_versions={"faiss": faiss.__version__, "numpy": np.__version__})
    probe = np.ascontiguousarray(sample[:min(8, sample.shape[0])], dtype=np.float32)
    codes, transformed, reconstructed = model.encode_batch(probe)
    query = np.ascontiguousarray(transformed[0], dtype=np.float32)
    computer = quantizer.get_distance_computer(0, None, False)
    computer.set_query(faiss.swig_ptr(query))
    for row in range(codes.shape[0]):
        upstream = float(computer.distance_to_code(faiss.swig_ptr(codes[row])))
        decoded_dot = float(np.dot(query.astype(np.float64),
                                   reconstructed[row].astype(np.float64)))
        if not np.isclose(upstream, decoded_dot, rtol=2e-5, atol=2e-6):
            raise RuntimeError("RaBitQ qb=0 IP estimator is not decode-dot linear")
    return model
