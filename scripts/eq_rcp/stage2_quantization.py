#!/usr/bin/env python3
"""Stage 2 quantizers and evaluation primitives for EQ-RCP."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier

from operational_dataset import Fvecs, canonical_json_bytes, read_json, sha256_bytes


STAGE2_FORMAT = "eq_rcp_flat_oae_ceiling"
STAGE2_VERSION = 1


def _array(table: Any, name: str, dtype: Any | None = None) -> np.ndarray:
    values = np.asarray(table[name].combine_chunks().to_numpy(zero_copy_only=False))
    return values.astype(dtype, copy=False) if dtype is not None else values


def stable_rank(seed: int, value: int) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()


class Stage1OperationalData:
    """Read the immutable Stage 1 dataset and reconstruct Delta/X on demand."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir.resolve()
        self.manifest = read_json(self.dataset_dir / "dataset_manifest.json")
        self.queries = pq.read_table(self.dataset_dir / "queries.parquet")
        self.edges = pq.read_table(self.dataset_dir / "edges.parquet")
        self.events = pq.read_table(self.dataset_dir / "events.parquet")
        self.query_ids = _array(self.queries, "query_id", np.int64)
        self.query_splits = np.asarray(self.queries["split"].to_pylist(), dtype=object)
        self.event_query = _array(self.events, "query_id", np.int64)
        self.event_edge = _array(self.events, "edge_id", np.int64)
        self.event_source_label = _array(
            self.events, "source_external_label", np.int64
        )
        self.edge_source_label = _array(
            self.edges, "source_external_label", np.int64
        )
        self.edge_target_label = _array(
            self.edges, "target_external_label", np.int64
        )
        self.edge_squared_length = _array(
            self.edges, "edge_squared_length", np.float64
        )
        inputs = self.manifest["inputs"]
        self.base = Fvecs(Path(inputs["base_vectors"]["physical_path"]))
        self.query = Fvecs(Path(inputs["query_vectors"]["physical_path"]))
        if self.base.dim != self.query.dim:
            raise ValueError("Stage 1 base/query dimensions differ")
        self.dim = self.base.dim
        self.query_offset = int(
            self.manifest["semantic_config"]["query_mapping"].get("offset", 0)
        )

    def split_query_ids(
        self, split: str, maximum_queries: int | None, seed: int
    ) -> np.ndarray:
        candidates = self.query_ids[self.query_splits == split]
        if candidates.size == 0:
            raise ValueError(f"Stage 1 split is empty or missing: {split}")
        ordered = sorted(candidates.tolist(), key=lambda value: stable_rank(seed, int(value)))
        if maximum_queries is not None:
            ordered = ordered[: int(maximum_queries)]
        return np.asarray(sorted(ordered), dtype=np.int64)

    def event_indices(self, query_ids: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.isin(self.event_query, query_ids)).astype(np.int64)

    def edge_vectors(self, edge_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(edge_ids, dtype=np.int64)
        source = self.base.take(self.edge_source_label[ids]).astype(np.float32)
        target = self.base.take(self.edge_target_label[ids]).astype(np.float32)
        return target - source

    def event_x(self, event_indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(event_indices, dtype=np.int64)
        qrows = self.event_query[indices] - self.query_offset
        query = self.query.take(qrows).astype(np.float32)
        source = self.base.take(self.event_source_label[indices]).astype(np.float32)
        return query - source

    def event_values(self, event_indices: np.ndarray) -> dict[str, np.ndarray]:
        indices = np.asarray(event_indices, dtype=np.int64)
        names = [
            "query_id",
            "edge_id",
            "current_squared_distance",
            "candidate_squared_distance",
            "threshold",
            "edge_squared_length",
            "y_exact",
            "margin",
            "good_candidate",
            "oracle_prunable",
            "raw_event_weight",
            "per_query_weight",
        ]
        return {name: _array(self.events, name)[indices] for name in names}

    def close(self) -> None:
        self.base.close()
        self.query.close()


@dataclass
class ProductQuantizer:
    subquantizers: int
    centroids: int
    centers: np.ndarray

    @property
    def dim(self) -> int:
        return int(self.centers.shape[0] * self.centers.shape[2])

    @property
    def code_bytes(self) -> int:
        if self.centroids > 256:
            raise ValueError("Stage 2 only supports one-byte subcodes")
        return self.subquantizers

    @classmethod
    def fit(
        cls,
        vectors: np.ndarray,
        sample_weight: np.ndarray,
        subquantizers: int,
        centroids: int,
        seed: int,
        max_iter: int,
        batch_size: int,
    ) -> "ProductQuantizer":
        vectors = np.asarray(vectors, dtype=np.float32)
        weights = np.asarray(sample_weight, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[1] % subquantizers:
            raise ValueError("dimension must be divisible by subquantizer count")
        if len(vectors) < centroids:
            raise ValueError(
                f"need at least {centroids} training edges, got {len(vectors)}"
            )
        if len(weights) != len(vectors) or np.any(weights <= 0):
            raise ValueError("PQ sample weights must be positive and edge-aligned")
        block_dim = vectors.shape[1] // subquantizers
        centers = np.empty(
            (subquantizers, centroids, block_dim), dtype=np.float32
        )
        for block in range(subquantizers):
            begin = block * block_dim
            end = begin + block_dim
            trainer = MiniBatchKMeans(
                n_clusters=centroids,
                random_state=seed + 104729 * block,
                batch_size=min(batch_size, len(vectors)),
                max_iter=max_iter,
                n_init=1,
                max_no_improvement=max(3, max_iter // 3),
                reassignment_ratio=0.0,
                init="k-means++",
            )
            trainer.fit(vectors[:, begin:end], sample_weight=weights)
            centers[block] = trainer.cluster_centers_.astype(np.float32)
        return cls(subquantizers, centroids, centers)

    def encode(self, vectors: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.dim:
            raise ValueError("PQ encode dimension mismatch")
        dtype = np.uint8 if self.centroids <= 256 else np.uint16
        codes = np.empty((len(vectors), self.subquantizers), dtype=dtype)
        block_dim = self.dim // self.subquantizers
        for block in range(self.subquantizers):
            begin = block * block_dim
            end = begin + block_dim
            center = self.centers[block]
            center_norm = np.einsum("ij,ij->i", center, center)
            for start in range(0, len(vectors), chunk_size):
                stop = min(start + chunk_size, len(vectors))
                value = vectors[start:stop, begin:end]
                distance = (
                    np.einsum("ij,ij->i", value, value)[:, None]
                    + center_norm[None, :]
                    - 2.0 * value @ center.T
                )
                codes[start:stop, block] = np.argmin(distance, axis=1)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        codes = np.asarray(codes)
        if codes.ndim != 2 or codes.shape[1] != self.subquantizers:
            raise ValueError("PQ code shape mismatch")
        block_dim = self.dim // self.subquantizers
        output = np.empty((len(codes), self.dim), dtype=np.float32)
        for block in range(self.subquantizers):
            begin = block * block_dim
            output[:, begin : begin + block_dim] = self.centers[block][
                codes[:, block].astype(np.int64)
            ]
        return output


class MetricTransform:
    """Regularized uncentered operational metric with exact inverse."""

    def __init__(
        self,
        kind: str,
        dim: int,
        block_size: int,
        bases: np.ndarray,
        scales: np.ndarray,
        shrinkage: float,
        ridge: float,
    ):
        self.kind = kind
        self.dim = dim
        self.block_size = block_size
        self.bases = bases
        self.scales = scales
        self.shrinkage = shrinkage
        self.ridge = ridge

    @staticmethod
    def _regularized_eigh(
        covariance: np.ndarray, shrinkage: float, ridge: float
    ) -> tuple[np.ndarray, np.ndarray]:
        dim = covariance.shape[0]
        average = max(float(np.trace(covariance)) / dim, 1e-12)
        regularized = (
            (1.0 - shrinkage) * covariance
            + shrinkage * average * np.eye(dim)
            + ridge * average * np.eye(dim)
        )
        eigenvalues, basis = np.linalg.eigh(regularized)
        floor = average * max(ridge, 1e-8)
        return basis.astype(np.float32), np.sqrt(np.maximum(eigenvalues, floor)).astype(
            np.float32
        )

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        weights: np.ndarray,
        kind: str,
        block_size: int,
        shrinkage: float,
        ridge: float,
    ) -> "MetricTransform":
        x = np.asarray(x, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if len(x) != len(weights) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("invalid metric weights")
        weights = weights / weights.sum()
        dim = x.shape[1]
        if kind == "diagonal":
            diagonal = np.einsum("i,ij,ij->j", weights, x, x)
            average = max(float(diagonal.mean()), 1e-12)
            value = (1.0 - shrinkage) * diagonal + (shrinkage + ridge) * average
            return cls(
                kind,
                dim,
                1,
                np.empty((0, 0, 0), dtype=np.float32),
                np.sqrt(np.maximum(value, average * 1e-8)).astype(np.float32),
                shrinkage,
                ridge,
            )
        if kind == "block":
            if dim % block_size:
                raise ValueError("metric block size must divide dimension")
            count = dim // block_size
            bases = np.empty((count, block_size, block_size), dtype=np.float32)
            scales = np.empty((count, block_size), dtype=np.float32)
            for block in range(count):
                begin = block * block_size
                xb = x[:, begin : begin + block_size]
                covariance = xb.T @ (weights[:, None] * xb)
                bases[block], scales[block] = cls._regularized_eigh(
                    covariance, shrinkage, ridge
                )
            return cls(kind, dim, block_size, bases, scales, shrinkage, ridge)
        if kind == "full":
            covariance = x.T @ (weights[:, None] * x)
            basis, scale = cls._regularized_eigh(covariance, shrinkage, ridge)
            return cls(
                kind,
                dim,
                dim,
                basis[None, :, :],
                scale[None, :],
                shrinkage,
                ridge,
            )
        raise ValueError(f"unsupported OAE metric kind: {kind}")

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        value = np.asarray(vectors, dtype=np.float32)
        if self.kind == "diagonal":
            return value * self.scales
        output = np.empty_like(value)
        for block, basis in enumerate(self.bases):
            begin = block * self.block_size
            end = begin + self.block_size
            output[:, begin:end] = (value[:, begin:end] @ basis) * self.scales[block]
        return output

    def inverse(self, transformed: np.ndarray) -> np.ndarray:
        value = np.asarray(transformed, dtype=np.float32)
        if self.kind == "diagonal":
            return value / self.scales
        output = np.empty_like(value)
        for block, basis in enumerate(self.bases):
            begin = block * self.block_size
            end = begin + self.block_size
            output[:, begin:end] = (value[:, begin:end] / self.scales[block]) @ basis.T
        return output

    def transform_query(self, x: np.ndarray) -> np.ndarray:
        """Return L^{-T}X so transformed-space dot products preserve X^T Delta."""
        value = np.asarray(x, dtype=np.float32)
        if self.kind == "diagonal":
            return value / self.scales
        output = np.empty_like(value)
        for block, basis in enumerate(self.bases):
            begin = block * self.block_size
            end = begin + self.block_size
            output[:, begin:end] = (value[:, begin:end] @ basis) / self.scales[block]
        return output


class EdgeQuantizer:
    name: str

    def reconstruct(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def artifact_metadata(self) -> dict[str, Any]:
        raise NotImplementedError


class FlatPQQuantizer(EdgeQuantizer):
    def __init__(self, name: str, pq: ProductQuantizer):
        self.name = name
        self.pq = pq

    @classmethod
    def fit(cls, name: str, vectors: np.ndarray, weights: np.ndarray, spec: dict[str, Any], seed: int) -> "FlatPQQuantizer":
        return cls(
            name,
            ProductQuantizer.fit(
                vectors,
                weights,
                int(spec["subquantizers"]),
                int(spec["centroids"]),
                seed,
                int(spec["max_iter"]),
                int(spec["batch_size"]),
            ),
        )

    def reconstruct(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        codes = self.pq.encode(vectors)
        return self.pq.decode(codes), codes

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        return {"centers": self.pq.centers}

    def artifact_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "pq",
            "subquantizers": self.pq.subquantizers,
            "centroids": self.pq.centroids,
            "code_bytes": self.pq.code_bytes,
        }


class OPQQuantizer(EdgeQuantizer):
    def __init__(self, pq: ProductQuantizer, rotation: np.ndarray, iterations: int):
        self.name = "direct_delta_opq"
        self.pq = pq
        self.rotation = rotation.astype(np.float32)
        self.iterations = iterations

    @classmethod
    def fit(cls, vectors: np.ndarray, weights: np.ndarray, spec: dict[str, Any], seed: int) -> "OPQQuantizer":
        vectors = np.asarray(vectors, dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float64)
        rotation = np.eye(vectors.shape[1], dtype=np.float32)
        iterations = int(spec.get("opq_iterations", 1))
        pq: ProductQuantizer | None = None
        for iteration in range(iterations + 1):
            rotated = vectors @ rotation
            pq = ProductQuantizer.fit(
                rotated,
                weights,
                int(spec["subquantizers"]),
                int(spec["centroids"]),
                seed + 1000003 * iteration,
                int(spec["max_iter"]),
                int(spec["batch_size"]),
            )
            if iteration == iterations:
                break
            reconstructed = pq.decode(pq.encode(rotated))
            cross = vectors.T @ (weights[:, None] * reconstructed)
            left, _, right = np.linalg.svd(cross, full_matrices=False)
            rotation = (left @ right).astype(np.float32)
        assert pq is not None
        return cls(pq, rotation, iterations)

    def reconstruct(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        codes = self.pq.encode(np.asarray(vectors, dtype=np.float32) @ self.rotation)
        return self.pq.decode(codes) @ self.rotation.T, codes

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        return {"centers": self.pq.centers, "rotation": self.rotation}

    def artifact_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "opq",
            "subquantizers": self.pq.subquantizers,
            "centroids": self.pq.centroids,
            "code_bytes": self.pq.code_bytes,
            "iterations": self.iterations,
        }


class OAEQuantizer(EdgeQuantizer):
    def __init__(self, metric: MetricTransform, pq: ProductQuantizer):
        self.metric = metric
        self.pq = pq
        self.name = f"oae_{metric.kind}"

    @classmethod
    def fit(
        cls,
        vectors: np.ndarray,
        edge_weights: np.ndarray,
        metric_x: np.ndarray,
        metric_weights: np.ndarray,
        metric_kind: str,
        spec: dict[str, Any],
        seed: int,
    ) -> "OAEQuantizer":
        metric = MetricTransform.fit(
            metric_x,
            metric_weights,
            metric_kind,
            int(spec["metric_block_size"]),
            float(spec["metric_shrinkage"]),
            float(spec["metric_ridge"]),
        )
        transformed = metric.transform(vectors)
        pq = ProductQuantizer.fit(
            transformed,
            edge_weights,
            int(spec["subquantizers"]),
            int(spec["centroids"]),
            seed,
            int(spec["max_iter"]),
            int(spec["batch_size"]),
        )
        return cls(metric, pq)

    def reconstruct(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        transformed = self.metric.transform(vectors)
        codes = self.pq.encode(transformed)
        return self.metric.inverse(self.pq.decode(codes)), codes

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        return {
            "centers": self.pq.centers,
            "metric_bases": self.metric.bases,
            "metric_scales": self.metric.scales,
        }

    def artifact_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "oae",
            "metric_kind": self.metric.kind,
            "metric_block_size": self.metric.block_size,
            "metric_shrinkage": self.metric.shrinkage,
            "metric_ridge": self.metric.ridge,
            "subquantizers": self.pq.subquantizers,
            "centroids": self.pq.centroids,
            "code_bytes": self.pq.code_bytes,
        }


class DirectionPQQuantizer(EdgeQuantizer):
    def __init__(self, pq: ProductQuantizer):
        self.name = "direction_pq_reference"
        self.pq = pq

    @classmethod
    def fit(cls, vectors: np.ndarray, weights: np.ndarray, spec: dict[str, Any], seed: int) -> "DirectionPQQuantizer":
        lengths = np.linalg.norm(vectors, axis=1)
        directions = vectors / np.maximum(lengths[:, None], 1e-12)
        pq = ProductQuantizer.fit(
            directions,
            weights,
            int(spec["subquantizers"]),
            int(spec["centroids"]),
            seed,
            int(spec["max_iter"]),
            int(spec["batch_size"]),
        )
        return cls(pq)

    def reconstruct(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lengths = np.linalg.norm(vectors, axis=1)
        directions = vectors / np.maximum(lengths[:, None], 1e-12)
        codes = self.pq.encode(directions)
        reconstructed = self.pq.decode(codes)
        reconstructed /= np.maximum(np.linalg.norm(reconstructed, axis=1)[:, None], 1e-12)
        return reconstructed * lengths[:, None], codes

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        return {"centers": self.pq.centers}

    def artifact_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "direction_pq_with_exact_length",
            "subquantizers": self.pq.subquantizers,
            "centroids": self.pq.centroids,
            "code_bytes": self.pq.code_bytes,
        }


def edge_training_set(
    edge_ids: np.ndarray, event_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique, inverse = np.unique(edge_ids.astype(np.int64), return_inverse=True)
    weights = np.bincount(inverse, weights=event_weights, minlength=len(unique))
    return unique, inverse, weights.astype(np.float64)


def smooth_threshold_weights(
    margin: np.ndarray, minimum: float, tau: float
) -> np.ndarray:
    value = np.abs(np.asarray(margin, dtype=np.float64)) / max(tau, 1e-12)
    return minimum + 1.0 / (1.0 + np.exp(np.minimum(value, 60.0)))


def fit_quantizer(
    method: str,
    train_vectors: np.ndarray,
    edge_weights: np.ndarray,
    metric_x: np.ndarray,
    metric_weights: np.ndarray,
    spec: dict[str, Any],
    seed: int,
) -> EdgeQuantizer:
    if method == "direct_delta_pq":
        return FlatPQQuantizer.fit(method, train_vectors, edge_weights, spec, seed)
    if method == "direct_delta_opq":
        return OPQQuantizer.fit(train_vectors, edge_weights, spec, seed)
    if method.startswith("oae_"):
        return OAEQuantizer.fit(
            train_vectors,
            edge_weights,
            metric_x,
            metric_weights,
            method.removeprefix("oae_"),
            spec,
            seed,
        )
    if method == "direction_pq_reference":
        return DirectionPQQuantizer.fit(train_vectors, edge_weights, spec, seed)
    raise ValueError(f"unknown Stage 2 method: {method}")


def save_quantizer(model: EdgeQuantizer, path: Path, extra: dict[str, Any]) -> dict[str, Any]:
    arrays = model.artifact_arrays()
    np.savez_compressed(path, **arrays)
    metadata = {**model.artifact_metadata(), **extra}
    metadata["model_semantic_sha256"] = model_semantic_sha256(arrays, metadata)
    metadata["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return metadata


def model_semantic_sha256(
    arrays: dict[str, np.ndarray], metadata: dict[str, Any]
) -> str:
    digest = hashlib.sha256()
    semantic_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in {"relative_path", "artifact_sha256", "model_semantic_sha256"}
    }
    encoded = canonical_json_bytes(semantic_metadata)
    digest.update(struct.pack("<Q", len(encoded)))
    digest.update(encoded)
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        dtype = value.dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(value.astype(dtype, copy=False))
        for piece in [
            name.encode("utf-8"),
            dtype.str.encode("ascii"),
            canonical_json_bytes(list(canonical.shape)),
            canonical.tobytes(order="C"),
        ]:
            digest.update(struct.pack("<Q", len(piece)))
            digest.update(piece)
    return digest.hexdigest()


def reconstruct_events(
    model: EdgeQuantizer,
    data: Stage1OperationalData,
    event_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    values = data.event_values(event_indices)
    edge_ids, inverse = np.unique(values["edge_id"].astype(np.int64), return_inverse=True)
    delta = data.edge_vectors(edge_ids)
    reconstructed, codes = model.reconstruct(delta)
    x = data.event_x(event_indices)
    event_reconstruction = reconstructed[inverse]
    y_hat = np.einsum("ij,ij->i", x.astype(np.float64), event_reconstruction.astype(np.float64))
    projection_error = values["y_exact"].astype(np.float64) - y_hat
    distance_hat = (
        values["current_squared_distance"].astype(np.float64)
        + values["edge_squared_length"].astype(np.float64)
        - 2.0 * y_hat
    )
    edge_mse = np.einsum(
        "ij,ij->i",
        delta.astype(np.float64) - reconstructed.astype(np.float64),
        delta.astype(np.float64) - reconstructed.astype(np.float64),
    )
    subquantizers = codes.shape[1]
    if x.shape[1] % subquantizers:
        raise ValueError("code blocks do not divide event X dimension")
    block_dim = x.shape[1] // subquantizers
    block_projection = np.empty((len(x), subquantizers), dtype=np.float32)
    block_x_norm = np.empty((len(x), subquantizers), dtype=np.float32)
    for block in range(subquantizers):
        begin = block * block_dim
        end = begin + block_dim
        xb = x[:, begin:end]
        rb = event_reconstruction[:, begin:end]
        block_projection[:, block] = np.einsum("ij,ij->i", xb, rb)
        block_x_norm[:, block] = np.einsum("ij,ij->i", xb, xb)
    return {
        **values,
        "y_hat": y_hat,
        "projection_error": projection_error,
        "distance_hat": distance_hat,
        "estimated_margin": distance_hat - values["threshold"].astype(np.float64),
        "edge_ids": edge_ids,
        "edge_inverse": inverse,
        "edge_mse": edge_mse,
        "codes": codes,
        "event_codes": codes[inverse],
        "block_projection": block_projection,
        "block_x_norm": block_x_norm,
    }


def code_oracle_features(payload: dict[str, np.ndarray]) -> np.ndarray:
    codes = payload["event_codes"].astype(np.float32) / 255.0
    block_projection = payload["block_projection"].astype(np.float32)
    block_x_norm = np.log1p(payload["block_x_norm"].astype(np.float32))
    scalar = np.column_stack(
        [
            payload["current_squared_distance"],
            payload["edge_squared_length"],
            payload["threshold"],
            payload["distance_hat"],
            payload["estimated_margin"],
        ]
    ).astype(np.float32)
    return np.column_stack([codes, block_projection, block_x_norm, scalar])


def fit_code_oracle(
    calibration: dict[str, np.ndarray], seed: int, spec: dict[str, Any]
) -> HistGradientBoostingClassifier:
    labels = calibration["oracle_prunable"].astype(np.int32)
    if len(np.unique(labels)) != 2:
        raise ValueError("flexible code oracle requires both decision classes")
    model = HistGradientBoostingClassifier(
        learning_rate=float(spec["learning_rate"]),
        max_iter=int(spec["max_iter"]),
        max_leaf_nodes=int(spec["max_leaf_nodes"]),
        min_samples_leaf=int(spec["min_samples_leaf"]),
        l2_regularization=float(spec["l2_regularization"]),
        early_stopping=False,
        random_state=seed,
    )
    model.fit(code_oracle_features(calibration), labels)
    return model


def code_oracle_scores(
    model: HistGradientBoostingClassifier, payload: dict[str, np.ndarray]
) -> np.ndarray:
    return model.predict_proba(code_oracle_features(payload))[:, 1].astype(np.float64)


def conservative_cutoff(
    scores: np.ndarray, good_candidate: np.ndarray, alpha: float
) -> float:
    unsafe_scores = np.asarray(scores, dtype=np.float64)[good_candidate.astype(bool)]
    if unsafe_scores.size == 0:
        raise ValueError("cannot calibrate diagnostic cutoff without good candidates")
    return float(np.quantile(unsafe_scores, 1.0 - alpha, method="higher"))


def decision_metrics(payload: dict[str, np.ndarray], cutoff: float) -> dict[str, float]:
    decision = payload["estimated_margin"] > cutoff
    good = payload["good_candidate"].astype(bool)
    prunable = payload["oracle_prunable"].astype(bool)
    false_prune = decision & good
    return {
        "coverage": float(np.mean(decision)),
        "oracle_opportunity_recall": float(np.sum(decision & prunable) / max(np.sum(prunable), 1)),
        "false_prune_rate_over_good": float(np.sum(false_prune) / max(np.sum(good), 1)),
        "false_prune_rate_over_all": float(np.mean(false_prune)),
    }


def projection_metrics(
    payload: dict[str, np.ndarray], near_margin_limit: float
) -> dict[str, float]:
    error = np.abs(payload["projection_error"].astype(np.float64))
    squared = error * error
    near = np.abs(payload["margin"].astype(np.float64)) <= near_margin_limit
    event_edge_mse = payload["edge_mse"][payload["edge_inverse"]]
    result = {
        "projection_mse": float(np.mean(squared)),
        "projection_abs_q50": float(np.quantile(error, 0.50)),
        "projection_abs_q95": float(np.quantile(error, 0.95)),
        "projection_abs_q99": float(np.quantile(error, 0.99)),
        "projection_abs_q999": float(np.quantile(error, 0.999)),
        "reconstruction_mse_unique_edge": float(np.mean(payload["edge_mse"])),
        "reconstruction_mse_event_weighted": float(np.mean(event_edge_mse)),
        "threshold_near_fraction": float(np.mean(near)),
        "threshold_near_projection_mse": float(np.mean(squared[near])) if np.any(near) else float("nan"),
        "threshold_near_projection_abs_q99": float(np.quantile(error[near], 0.99)) if np.any(near) else float("nan"),
    }
    return result


def per_query_rows(
    method: str, seed: int, payload: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    query_ids = payload["query_id"].astype(np.int64)
    absolute = np.abs(payload["projection_error"].astype(np.float64))
    for query_id in np.unique(query_ids):
        mask = query_ids == query_id
        rows.append(
            {
                "method": method,
                "seed": int(seed),
                "query_id": int(query_id),
                "event_count": int(np.sum(mask)),
                "projection_mse": float(np.mean(absolute[mask] ** 2)),
                "projection_abs_q95": float(np.quantile(absolute[mask], 0.95)),
                "projection_abs_q99": float(np.quantile(absolute[mask], 0.99)),
            }
        )
    return rows


def paired_bootstrap_ci(
    baseline: np.ndarray,
    challenger: np.ndarray,
    seed: int,
    samples: int,
    confidence: float,
) -> dict[str, float]:
    baseline = np.asarray(baseline, dtype=np.float64)
    challenger = np.asarray(challenger, dtype=np.float64)
    if baseline.shape != challenger.shape or baseline.size == 0:
        raise ValueError("paired bootstrap inputs must be non-empty and aligned")
    differences = baseline - challenger
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = rng.integers(0, len(differences), size=len(differences))
        means[index] = float(np.mean(differences[draw]))
    tail = (1.0 - confidence) / 2.0
    return {
        "mean_baseline_minus_challenger": float(np.mean(differences)),
        "ci_lower": float(np.quantile(means, tail)),
        "ci_upper": float(np.quantile(means, 1.0 - tail)),
        "confidence": confidence,
        "samples": int(samples),
    }


def results_semantic_sha256(value: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(value))
