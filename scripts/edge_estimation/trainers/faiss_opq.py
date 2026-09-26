from __future__ import annotations

import numpy as np

from .faiss_pq import _configure_threads, _faiss
from .product_model import ProductCodebookModel


def train(sample: np.ndarray, m: int, nbits: int, trainer: dict) -> ProductCodebookModel:
    faiss = _faiss()
    _configure_threads(faiss, trainer)
    values = np.ascontiguousarray(sample, dtype=np.float32)
    dimension = int(values.shape[1])
    pq = faiss.ProductQuantizer(dimension, m, nbits)
    pq.cp.seed = int(trainer.get("seed", 20260924))
    pq.cp.nredo = int(trainer.get("redos", 1))
    opq = faiss.OPQMatrix(dimension, m)
    opq.pq = pq
    opq.niter = int(trainer.get("outer_iterations", 25))
    opq.niter_pq_0 = int(trainer.get("initial_pq_iterations", trainer.get("iterations", 25)))
    opq.niter_pq = int(trainer.get("iterations", 4))
    opq.verbose = bool(trainer.get("verbose", False))
    opq.train(values)
    if bool(opq.have_bias):
        raise RuntimeError("OPQ artifact v1 does not support a non-zero bias")
    rotation = np.asarray(faiss.vector_to_array(opq.A), dtype=np.float32).reshape(
        dimension, dimension)
    trained_pq = opq.pq
    codebooks = np.asarray(faiss.vector_to_array(trained_pq.centroids), dtype=np.float32).reshape(
        m, 1 << nbits, dimension // m)

    def encode(transformed: np.ndarray) -> np.ndarray:
        raw = trained_pq.compute_codes(np.ascontiguousarray(transformed, dtype=np.float32))
        return np.asarray(faiss.unpack_bitstrings(raw, m, nbits), dtype=np.uint16)

    model = ProductCodebookModel(
        backend="opq", implementation="faiss_opq_product_quantizer", provider="faiss",
        codebooks=np.ascontiguousarray(codebooks, dtype="<f4"), nbits=nbits,
        encoder=encode, rotation=np.ascontiguousarray(rotation, dtype="<f4"),
        dependency_versions={"faiss": faiss.__version__, "numpy": np.__version__})
    probe = values[:min(64, values.shape[0])]
    transformed = model.transform(probe)
    upstream_transform = opq.apply_py(probe)
    if not np.allclose(transformed, upstream_transform, rtol=1e-5, atol=1e-6):
        raise RuntimeError("OPQ rotation layout parity failed")
    codes, _, reconstructed = model.encode_batch(probe)
    upstream_decode = trained_pq.decode(faiss.pack_bitstrings(codes, nbits))
    if not np.allclose(reconstructed, upstream_decode, rtol=1e-5, atol=1e-6):
        raise RuntimeError("OPQ PQ canonical code/decode parity failed")
    return model
