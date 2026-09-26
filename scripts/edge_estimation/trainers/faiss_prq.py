from __future__ import annotations

import numpy as np

from .additive_model import ProductResidualModel
from .faiss_pq import _configure_threads, _faiss


def train(sample: np.ndarray, nsplits: int, stages: int, nbits: int,
          trainer: dict) -> ProductResidualModel:
    faiss = _faiss()
    _configure_threads(faiss, trainer)
    values = np.ascontiguousarray(sample, dtype=np.float32)
    dimension = int(values.shape[1])
    prq = faiss.ProductResidualQuantizer(
        dimension, nsplits, stages, nbits, faiss.AdditiveQuantizer.ST_LUT_nonorm)
    seed = int(trainer.get("seed", 20260924))
    for split in range(nsplits):
        residual = faiss.downcast_AdditiveQuantizer(prq.subquantizer(split))
        residual.cp.niter = int(trainer.get("iterations", 25))
        residual.cp.seed = seed + split
        residual.cp.nredo = int(trainer.get("redos", 1))
        residual.max_beam_size = int(trainer.get("beam_size", 5))
    prq.verbose = bool(trainer.get("verbose", False))
    prq.train(values)
    if int(prq.norm_bits) != 0:
        raise RuntimeError("PRQ ST_LUT_nonorm unexpectedly emitted norm bits")
    code_count = nsplits * stages
    ksub = 1 << nbits
    dsub = dimension // nsplits
    expected_code_size = (code_count * nbits + 7) // 8
    if int(prq.code_size) != expected_code_size:
        raise RuntimeError("PRQ code_size does not match the requested payload")
    offsets = np.asarray(faiss.vector_to_array(prq.codebook_offsets), dtype=np.int64)
    expected_offsets = np.arange(code_count + 1, dtype=np.int64) * ksub
    if not np.array_equal(offsets, expected_offsets):
        raise RuntimeError("PRQ codebook offsets are not canonical split/stage order")
    codebooks = np.asarray(faiss.vector_to_array(prq.codebooks), dtype=np.float32).reshape(
        code_count, ksub, dsub)

    def encode(transformed: np.ndarray) -> np.ndarray:
        raw = prq.compute_codes(np.ascontiguousarray(transformed, dtype=np.float32))
        return np.asarray(faiss.unpack_bitstrings(raw, code_count, nbits), dtype=np.uint16)

    model = ProductResidualModel(
        backend="prq", implementation="faiss_product_residual_quantizer",
        provider="faiss", codebooks=np.ascontiguousarray(codebooks, dtype="<f4"),
        nbits=nbits, nsplits=nsplits, stages_per_split=stages, encoder=encode,
        dependency_versions={"faiss": faiss.__version__, "numpy": np.__version__})
    probe = values[:min(64, values.shape[0])]
    codes, _, reconstructed = model.encode_batch(probe)
    upstream = prq.decode(faiss.pack_bitstrings(codes, nbits))
    if not np.allclose(reconstructed, upstream, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Faiss PRQ split/stage decode parity failed")
    return model
