from __future__ import annotations

import numpy as np

from .jq_behavioral_reference import JQConfig, JQQuantizer, OFFICIAL_JHQ_COMMIT
from .product_model import ProductCodebookModel


def train(sample: np.ndarray, m: int, nbits: int, trainer: dict) -> ProductCodebookModel:
    if nbits != 8:
        raise ValueError("the single-level JQ behavioral contract requires 8-bit codes")
    config = JQConfig(
        dimension=int(sample.shape[1]), subspaces=m, bits=nbits,
        rotation_seed=int(trainer.get("rotation_seed", trainer.get("seed", 1234))),
        initializer_seed=int(trainer.get("initializer_seed", 42)),
        use_analytical_init=True, use_kmeans_refinement=False)
    quantizer = JQQuantizer(config)
    quantizer.fit_rotated(quantizer.rotate(sample))

    def encode(transformed: np.ndarray) -> np.ndarray:
        return quantizer.encode_rotated(transformed,
                                        batch_size=int(trainer.get("encode_batch_size", 2048)))

    return ProductCodebookModel(
        backend="jq", implementation="legacy_jq_behavioral_port",
        provider="vldb26_jhq_behavioral_port",
        codebooks=np.ascontiguousarray(quantizer.centroids, dtype="<f4"),
        nbits=8, encoder=encode,
        rotation=np.ascontiguousarray(quantizer.rotation, dtype="<f4"),
        dependency_versions={"upstream_commit_claim": OFFICIAL_JHQ_COMMIT,
                             "oracle_status": "behavioral_port_not_author_binary"})
