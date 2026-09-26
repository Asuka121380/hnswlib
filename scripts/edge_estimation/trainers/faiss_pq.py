from __future__ import annotations

import numpy as np

from .product_model import ProductCodebookModel


def _faiss() -> object:
    try:
        import faiss
    except ImportError as error:
        raise RuntimeError("Faiss trainer requested but the faiss module is unavailable") from error
    return faiss


def _configure_threads(faiss: object, trainer: dict) -> int:
    threads = int(trainer.get("threads", 1))
    if threads <= 0:
        raise ValueError("trainer threads must be positive")
    faiss.omp_set_num_threads(threads)
    return threads


def train(sample: np.ndarray, m: int, nbits: int, trainer: dict) -> ProductCodebookModel:
    faiss = _faiss()
    _configure_threads(faiss, trainer)
    values = np.ascontiguousarray(sample, dtype=np.float32)
    dimension = int(values.shape[1])
    pq = faiss.ProductQuantizer(dimension, m, nbits)
    pq.cp.niter = int(trainer.get("iterations", 25))
    pq.cp.seed = int(trainer.get("seed", 20260924))
    pq.cp.nredo = int(trainer.get("redos", 1))
    pq.verbose = bool(trainer.get("verbose", False))
    pq.train(values)
    codebooks = np.asarray(faiss.vector_to_array(pq.centroids), dtype=np.float32).reshape(
        m, 1 << nbits, dimension // m)

    def encode(transformed: np.ndarray) -> np.ndarray:
        raw = pq.compute_codes(np.ascontiguousarray(transformed, dtype=np.float32))
        codes = faiss.unpack_bitstrings(raw, m, nbits)
        return np.asarray(codes, dtype=np.uint16)

    model = ProductCodebookModel(
        backend="pq_packed", implementation="faiss_product_quantizer",
        provider="faiss", codebooks=np.ascontiguousarray(codebooks, dtype="<f4"),
        nbits=nbits, encoder=encode,
        dependency_versions={"faiss": faiss.__version__, "numpy": np.__version__})
    probe = values[:min(64, values.shape[0])]
    codes, _, reconstructed = model.encode_batch(probe)
    raw = faiss.pack_bitstrings(codes, nbits)
    upstream = pq.decode(raw)
    if not np.allclose(reconstructed, upstream, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Faiss PQ canonical code/decode parity failed")
    return model
