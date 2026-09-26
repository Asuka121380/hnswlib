from __future__ import annotations

from typing import Any

from .product_model import ProductCodebookModel


def train_product_model(codec: dict[str, Any], trainer: dict[str, Any],
                        sample: object) -> object:
    kind = str(codec.get("kind"))
    provider = str(trainer.get("provider", "numpy_reference"))
    m = int(codec.get("m", 32 if kind == "opq" else 0))
    nbits = int(codec.get("nbits", 8))
    if kind == "pq_packed" and provider == "numpy_reference":
        from .numpy_pq import train
        return train(sample, m, nbits, trainer)
    if kind == "pq_packed" and provider == "faiss":
        from .faiss_pq import train
        return train(sample, m, nbits, trainer)
    if kind == "opq" and provider == "faiss":
        from .faiss_opq import train
        return train(sample, m, nbits, trainer)
    if kind == "jq" and provider == "behavioral_port":
        from .jq import train
        return train(sample, m, nbits, trainer)
    if kind == "prq" and provider == "faiss":
        from .faiss_prq import train
        return train(sample, int(codec.get("nsplits", 16)),
                     int(codec.get("stages_per_split", 2)), nbits, trainer)
    if kind == "rabitq" and provider == "faiss":
        from .faiss_rabitq import train
        return train(sample, trainer)
    raise RuntimeError(f"unsupported trainer combination: backend={kind}, provider={provider}")


__all__ = ["ProductCodebookModel", "train_product_model"]
