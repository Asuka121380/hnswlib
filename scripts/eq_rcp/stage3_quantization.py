#!/usr/bin/env python3
"""Progressive gain-shape quantization primitives for EQ-RCP Stage 3."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans


STAGE3_FORMAT = "eq_rcp_stage3_progressive_quantization"
STAGE3_VERSION = 1
EPSILON = 1e-12


def unit_vectors(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(vectors, dtype=np.float32)
    lengths = np.linalg.norm(values.astype(np.float64), axis=1).astype(np.float32)
    unit = np.zeros_like(values)
    nonzero = lengths > EPSILON
    unit[nonzero] = values[nonzero] / lengths[nonzero, None]
    return unit, lengths


def _balanced_blocks(
    dimension: int,
    block_count: int,
    importance: np.ndarray,
    strategy: str,
) -> list[np.ndarray]:
    if block_count <= 0 or block_count > dimension:
        raise ValueError("block count must be in [1, dimension]")
    importance = np.asarray(importance, dtype=np.float64)
    if importance.shape != (dimension,) or np.any(~np.isfinite(importance)):
        raise ValueError("dimension importance is invalid")
    if strategy == "uniform":
        return [piece.astype(np.int32) for piece in np.array_split(np.arange(dimension), block_count)]
    order = np.argsort(-importance, kind="stable")
    blocks: list[list[int]] = [[] for _ in range(block_count)]
    loads = np.zeros(block_count, dtype=np.float64)
    sizes = np.zeros(block_count, dtype=np.int32)
    maximum_size = int(np.ceil(dimension / block_count))
    for dim in order:
        eligible = np.flatnonzero(sizes < maximum_size)
        target = int(eligible[np.argmin(loads[eligible])])
        blocks[target].append(int(dim))
        loads[target] += max(float(importance[dim]), EPSILON)
        sizes[target] += 1
    return [np.asarray(sorted(block), dtype=np.int32) for block in blocks]


@dataclass
class VariableBlockPQ:
    """One-byte PQ with arbitrary, disjoint dimension blocks."""

    centers: list[np.ndarray]
    blocks: list[np.ndarray]

    @property
    def code_bytes(self) -> int:
        return len(self.blocks)

    @property
    def centroids(self) -> int:
        return int(self.centers[0].shape[0])

    @property
    def dimension(self) -> int:
        return int(sum(len(block) for block in self.blocks))

    @classmethod
    def fit(
        cls,
        vectors: np.ndarray,
        weights: np.ndarray,
        blocks: list[np.ndarray],
        centroids: int,
        seed: int,
        max_iter: int,
        batch_size: int,
    ) -> "VariableBlockPQ":
        values = np.asarray(vectors, dtype=np.float32)
        sample_weight = np.asarray(weights, dtype=np.float64)
        if len(values) < centroids:
            raise ValueError("insufficient vectors for Stage 3 centroids")
        if sample_weight.shape != (len(values),) or np.any(sample_weight <= 0):
            raise ValueError("Stage 3 weights must be positive and edge-aligned")
        concatenated = np.concatenate(blocks)
        if sorted(concatenated.tolist()) != list(range(values.shape[1])):
            raise ValueError("PQ blocks must partition every dimension exactly once")
        centers: list[np.ndarray] = []
        for block_index, dimensions in enumerate(blocks):
            trainer = MiniBatchKMeans(
                n_clusters=centroids,
                random_state=seed + 104729 * block_index,
                batch_size=min(batch_size, len(values)),
                max_iter=max_iter,
                n_init=1,
                max_no_improvement=max(3, max_iter // 3),
                reassignment_ratio=0.0,
                init="k-means++",
            )
            trainer.fit(values[:, dimensions], sample_weight=sample_weight)
            centers.append(trainer.cluster_centers_.astype(np.float32))
        return cls(centers, [np.asarray(block, dtype=np.int32) for block in blocks])

    def encode(self, vectors: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
        values = np.asarray(vectors, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("Stage 3 PQ encode dimension mismatch")
        codes = np.empty((len(values), self.code_bytes), dtype=np.uint8)
        for block, (dimensions, centers) in enumerate(zip(self.blocks, self.centers)):
            center_norm = np.einsum("ij,ij->i", centers, centers)
            for start in range(0, len(values), chunk_size):
                stop = min(start + chunk_size, len(values))
                chunk = values[start:stop, dimensions]
                distances = (
                    np.einsum("ij,ij->i", chunk, chunk)[:, None]
                    + center_norm[None, :]
                    - 2.0 * chunk @ centers.T
                )
                codes[start:stop, block] = np.argmin(distances, axis=1)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        values = np.asarray(codes)
        if values.ndim != 2 or values.shape[1] != self.code_bytes:
            raise ValueError("Stage 3 PQ code shape mismatch")
        output = np.zeros((len(values), self.dimension), dtype=np.float32)
        for block, (dimensions, centers) in enumerate(zip(self.blocks, self.centers)):
            output[:, dimensions] = centers[values[:, block].astype(np.int64)]
        return output

    def arrays(self, prefix: str) -> dict[str, np.ndarray]:
        offsets = [0]
        order: list[int] = []
        packed_centers: list[np.ndarray] = []
        for dimensions, centers in zip(self.blocks, self.centers):
            order.extend(dimensions.tolist())
            offsets.append(len(order))
            packed_centers.append(centers)
        maximum = max(center.shape[1] for center in packed_centers)
        padded = np.zeros((len(packed_centers), self.centroids, maximum), dtype=np.float32)
        widths = np.empty(len(packed_centers), dtype=np.int32)
        for index, center in enumerate(packed_centers):
            widths[index] = center.shape[1]
            padded[index, :, : center.shape[1]] = center
        return {
            f"{prefix}_centers": padded,
            f"{prefix}_widths": widths,
            f"{prefix}_dimension_order": np.asarray(order, dtype=np.int32),
            f"{prefix}_block_offsets": np.asarray(offsets, dtype=np.int32),
        }


@dataclass
class ProgressiveGainShapeQuantizer:
    rotation: np.ndarray
    coarse: VariableBlockPQ
    residual: VariableBlockPQ | None
    layout_name: str
    allocation_strategy: str
    frozen_flat: bool = False

    @property
    def coarse_bytes(self) -> int:
        return self.coarse.code_bytes

    @property
    def residual_bytes(self) -> int:
        return 0 if self.residual is None else self.residual.code_bytes

    @classmethod
    def from_frozen_flat(cls, artifact_path: Path) -> "ProgressiveGainShapeQuantizer":
        with np.load(artifact_path, allow_pickle=False) as archive:
            rotation = np.asarray(archive["rotation"], dtype=np.float32)
            centers = np.asarray(archive["centers"], dtype=np.float32)
        block_dim = centers.shape[2]
        blocks = [
            np.arange(index * block_dim, (index + 1) * block_dim, dtype=np.int32)
            for index in range(centers.shape[0])
        ]
        coarse = VariableBlockPQ([centers[index] for index in range(len(blocks))], blocks)
        return cls(rotation, coarse, None, "flat_32", "frozen_flat", True)

    @staticmethod
    def _pq_from_archive(archive: Any, prefix: str) -> VariableBlockPQ:
        padded = np.asarray(archive[f"{prefix}_centers"], dtype=np.float32)
        widths = np.asarray(archive[f"{prefix}_widths"], dtype=np.int32)
        order = np.asarray(archive[f"{prefix}_dimension_order"], dtype=np.int32)
        offsets = np.asarray(archive[f"{prefix}_block_offsets"], dtype=np.int32)
        blocks: list[np.ndarray] = []
        centers: list[np.ndarray] = []
        for index, width in enumerate(widths):
            blocks.append(order[offsets[index] : offsets[index + 1]].copy())
            centers.append(padded[index, :, : int(width)].copy())
        return VariableBlockPQ(centers, blocks)

    @classmethod
    def load(cls, artifact_path: Path, metadata: dict[str, Any]) -> "ProgressiveGainShapeQuantizer":
        with np.load(artifact_path, allow_pickle=False) as archive:
            rotation = np.asarray(archive["rotation"], dtype=np.float32)
            coarse = cls._pq_from_archive(archive, "coarse")
            residual = (
                cls._pq_from_archive(archive, "residual")
                if int(metadata["residual_bytes"]) > 0
                else None
            )
        return cls(
            rotation,
            coarse,
            residual,
            str(metadata["layout"]),
            str(metadata["allocation_strategy"]),
            bool(metadata.get("frozen_flat_artifact", False)),
        )

    @classmethod
    def fit(
        cls,
        edge_vectors: np.ndarray,
        edge_weights: np.ndarray,
        rotation: np.ndarray,
        coarse_bytes: int,
        residual_bytes: int,
        allocation_strategy: str,
        operational_x: np.ndarray,
        seed: int,
        spec: dict[str, Any],
        layout_name: str,
    ) -> "ProgressiveGainShapeQuantizer":
        directions, _ = unit_vectors(edge_vectors)
        rotation = np.asarray(rotation, dtype=np.float32)
        if rotation.shape != (directions.shape[1], directions.shape[1]):
            raise ValueError("frozen Stage 2.1 rotation shape mismatch")
        rotated = directions @ rotation
        x_rotated = np.asarray(operational_x, dtype=np.float32) @ rotation
        operational = np.mean(x_rotated.astype(np.float64) ** 2, axis=0)
        variance = np.var(rotated.astype(np.float64), axis=0)
        if allocation_strategy == "uniform":
            importance = np.ones(rotated.shape[1], dtype=np.float64)
            block_strategy = "uniform"
        elif allocation_strategy == "operational_rate_distortion":
            importance = operational
            block_strategy = allocation_strategy
        elif allocation_strategy == "joint_objective":
            importance = np.sqrt(np.maximum(operational * variance, EPSILON))
            block_strategy = allocation_strategy
        elif allocation_strategy == "greedy_residual":
            importance = np.maximum(variance, EPSILON)
            block_strategy = allocation_strategy
        else:
            raise ValueError(f"unsupported Stage 3 allocation strategy: {allocation_strategy}")
        coarse_blocks = _balanced_blocks(rotated.shape[1], coarse_bytes, importance, block_strategy)
        coarse = VariableBlockPQ.fit(
            rotated,
            edge_weights,
            coarse_blocks,
            int(spec["centroids"]),
            seed,
            int(spec["max_iter"]),
            int(spec["batch_size"]),
        )
        coarse_codes = coarse.encode(rotated)
        coarse_direction, _ = unit_vectors(coarse.decode(coarse_codes))
        residual = None
        if residual_bytes:
            raw_residual = rotated - coarse_direction
            radial = np.einsum("ij,ij->i", raw_residual, coarse_direction)
            tangent = raw_residual - radial[:, None] * coarse_direction
            if allocation_strategy == "greedy_residual":
                residual_importance = np.var(tangent.astype(np.float64), axis=0)
            elif allocation_strategy == "joint_objective":
                residual_importance = np.sqrt(
                    np.maximum(operational * np.var(tangent.astype(np.float64), axis=0), EPSILON)
                )
            else:
                residual_importance = importance
            residual_blocks = _balanced_blocks(
                rotated.shape[1], residual_bytes, residual_importance, block_strategy
            )
            residual = VariableBlockPQ.fit(
                tangent,
                edge_weights,
                residual_blocks,
                int(spec["centroids"]),
                seed + 2000003,
                int(spec["max_iter"]),
                int(spec["batch_size"]),
            )
        return cls(rotation, coarse, residual, layout_name, allocation_strategy)

    def encode(self, edge_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        directions, _ = unit_vectors(edge_vectors)
        rotated = directions @ self.rotation
        coarse_codes = self.coarse.encode(rotated)
        if self.residual is None:
            return coarse_codes, None
        coarse_direction, _ = unit_vectors(self.coarse.decode(coarse_codes))
        raw_residual = rotated - coarse_direction
        radial = np.einsum("ij,ij->i", raw_residual, coarse_direction)
        tangent = raw_residual - radial[:, None] * coarse_direction
        return coarse_codes, self.residual.encode(tangent)

    def decode(
        self,
        coarse_codes: np.ndarray,
        gains: np.ndarray,
        residual_codes: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        coarse_rotated, _ = unit_vectors(self.coarse.decode(coarse_codes))
        full_rotated = coarse_rotated
        if residual_codes is not None:
            if self.residual is None:
                raise ValueError("residual codes supplied to flat Stage 3 model")
            decoded_residual = self.residual.decode(residual_codes)
            radial = np.einsum("ij,ij->i", decoded_residual, coarse_rotated)
            decoded_residual = decoded_residual - radial[:, None] * coarse_rotated
            full_rotated, _ = unit_vectors(coarse_rotated + decoded_residual)
        coarse = (coarse_rotated @ self.rotation.T) * np.asarray(gains)[:, None]
        full = (full_rotated @ self.rotation.T) * np.asarray(gains)[:, None]
        zero = np.asarray(gains) <= EPSILON
        coarse[zero] = 0.0
        full[zero] = 0.0
        return coarse.astype(np.float32), full.astype(np.float32)

    def reconstruct(self, edge_vectors: np.ndarray) -> dict[str, np.ndarray]:
        _, gains = unit_vectors(edge_vectors)
        coarse_codes, residual_codes = self.encode(edge_vectors)
        coarse, full = self.decode(coarse_codes, gains, residual_codes)
        result = {
            "coarse_reconstruction": coarse,
            "full_reconstruction": full,
            "coarse_codes": coarse_codes,
            "gains": gains,
        }
        if residual_codes is not None:
            result["residual_codes"] = residual_codes
        return result

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        arrays = {"rotation": self.rotation, **self.coarse.arrays("coarse")}
        if self.residual is not None:
            arrays.update(self.residual.arrays("residual"))
        return arrays

    def metadata(self) -> dict[str, Any]:
        return {
            "format": STAGE3_FORMAT,
            "version": STAGE3_VERSION,
            "layout": self.layout_name,
            "allocation_strategy": self.allocation_strategy,
            "parameterization_contract": "unit_direction_exact_gain",
            "residual_contract": "gain_shape_tangent_direction_residual",
            "gain_source": "stage1_edge_squared_length",
            "gain_is_exact": True,
            "coarse_bytes": self.coarse_bytes,
            "residual_bytes": self.residual_bytes,
            "total_code_bytes": self.coarse_bytes + self.residual_bytes,
            "frozen_flat_artifact": self.frozen_flat,
            "zero_vector_guard": EPSILON,
        }


def projection_payload(
    model: ProgressiveGainShapeQuantizer,
    edge_vectors: np.ndarray,
    event_inverse: np.ndarray,
    x: np.ndarray,
    current_distance: np.ndarray,
    edge_squared_length: np.ndarray,
    threshold: np.ndarray,
    good_candidate: np.ndarray,
    oracle_prunable: np.ndarray,
) -> dict[str, np.ndarray]:
    reconstruction = model.reconstruct(edge_vectors)
    inverse = np.asarray(event_inverse, dtype=np.int64)
    values = np.asarray(x, dtype=np.float64)
    coarse_y = np.einsum(
        "ij,ij->i", values, reconstruction["coarse_reconstruction"][inverse]
    )
    full_y = np.einsum(
        "ij,ij->i", values, reconstruction["full_reconstruction"][inverse]
    )
    exact_y = np.einsum("ij,ij->i", values, np.asarray(edge_vectors)[inverse])
    base = np.asarray(current_distance, dtype=np.float64) + np.asarray(
        edge_squared_length, dtype=np.float64
    )
    result = {
        **reconstruction,
        "coarse_y": coarse_y,
        "full_y": full_y,
        "exact_y": exact_y,
        "coarse_projection_error": exact_y - coarse_y,
        "full_projection_error": exact_y - full_y,
        "coarse_distance": base - 2.0 * coarse_y,
        "full_distance": base - 2.0 * full_y,
        "threshold": np.asarray(threshold, dtype=np.float64),
        "good_candidate": np.asarray(good_candidate, dtype=bool),
        "oracle_prunable": np.asarray(oracle_prunable, dtype=bool),
        "event_inverse": inverse,
    }
    return result


def fit_stopping_radius(projection_error: np.ndarray, coverage: float) -> float:
    if not 0.0 < coverage <= 1.0:
        raise ValueError("stopping coverage quantile must be in (0, 1]")
    absolute = np.abs(np.asarray(projection_error, dtype=np.float64))
    if absolute.size == 0:
        raise ValueError("cannot fit stopping radius on empty errors")
    return float(np.quantile(absolute, coverage, method="higher"))


def evaluate_stopping(payload: dict[str, np.ndarray], radius: float) -> dict[str, float]:
    lower_bound = payload["coarse_distance"] - 2.0 * float(radius)
    stop = lower_bound > payload["threshold"]
    good = payload["good_candidate"]
    prunable = payload["oracle_prunable"]
    return {
        "projection_radius": float(radius),
        "coarse_only_fraction": float(np.mean(stop)),
        "coarse_oracle_opportunity_recall": float(np.sum(stop & prunable) / max(np.sum(prunable), 1)),
        "coarse_false_prune_rate_over_good": float(np.sum(stop & good) / max(np.sum(good), 1)),
    }


def projection_metrics(payload: dict[str, np.ndarray]) -> dict[str, float]:
    result: dict[str, float] = {}
    for stage in ["coarse", "full"]:
        absolute = np.abs(payload[f"{stage}_projection_error"])
        result[f"{stage}_projection_mse"] = float(np.mean(absolute * absolute))
        for quantile, suffix in [(0.95, "q95"), (0.99, "q99"), (0.999, "q999")]:
            result[f"{stage}_projection_abs_{suffix}"] = float(np.quantile(absolute, quantile))
    return result


def save_progressive_model(
    model: ProgressiveGainShapeQuantizer, path: Path, extra: dict[str, Any]
) -> dict[str, Any]:
    arrays = model.artifact_arrays()
    np.savez_compressed(path, **arrays)
    metadata = {**model.metadata(), **extra}
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    metadata["model_semantic_sha256"] = digest.hexdigest()
    metadata["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return metadata
