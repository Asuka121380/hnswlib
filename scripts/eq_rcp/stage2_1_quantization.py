#!/usr/bin/env python3
"""Stage 2.1 representation candidates and causal diagnostics for EQ-RCP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from stage2_quantization import (
    EdgeQuantizer,
    FlatPQQuantizer,
    OAEQuantizer,
    OPQQuantizer,
    ProductQuantizer,
)


STAGE21_FORMAT = "eq_rcp_stage2_1_representation_selection"
STAGE21_VERSION = 1
EPSILON = 1e-12


def _unit(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(vectors, dtype=np.float32)
    lengths = np.linalg.norm(value.astype(np.float64), axis=1).astype(np.float32)
    direction = np.zeros_like(value)
    nonzero = lengths > EPSILON
    direction[nonzero] = value[nonzero] / lengths[nonzero, None]
    return direction, lengths


class ExactLengthQuantizer(EdgeQuantizer):
    """Apply the already-available exact edge length to a base reconstruction."""

    def __init__(self, name: str, base: EdgeQuantizer):
        self.name = name
        self.base = base

    def reconstruct(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, lengths = _unit(vectors)
        decoded, codes = self.base.reconstruct(vectors)
        decoded_direction, decoded_lengths = _unit(decoded)
        output = decoded_direction * lengths[:, None]
        output[(lengths <= EPSILON) | (decoded_lengths <= EPSILON)] = 0.0
        return output, codes

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        return self.base.artifact_arrays()

    def artifact_metadata(self) -> dict[str, Any]:
        metadata = dict(self.base.artifact_metadata())
        metadata.update(
            name=self.name,
            kind="exact_length_wrapper",
            base_kind=metadata["kind"],
            length_source="stage1_edge_squared_length",
            zero_vector_guard=EPSILON,
        )
        return metadata


class UnitDirectionQuantizer(EdgeQuantizer):
    """Train PQ/OPQ on unit directions and restore exact edge length."""

    def __init__(self, name: str, base: EdgeQuantizer, optimizer: str):
        self.name = name
        self.base = base
        self.optimizer = optimizer

    @classmethod
    def fit(
        cls,
        name: str,
        vectors: np.ndarray,
        weights: np.ndarray,
        spec: dict[str, Any],
        seed: int,
        optimizer: str,
    ) -> "UnitDirectionQuantizer":
        direction, _ = _unit(vectors)
        if optimizer == "pq":
            base: EdgeQuantizer = FlatPQQuantizer.fit(
                "unit_direction_pq_base", direction, weights, spec, seed
            )
        elif optimizer == "opq":
            base = OPQQuantizer.fit(direction, weights, spec, seed)
        else:
            raise ValueError(f"unsupported direction optimizer: {optimizer}")
        return cls(name, base, optimizer)

    def reconstruct(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        direction, lengths = _unit(vectors)
        decoded, codes = self.base.reconstruct(direction)
        decoded_direction, decoded_lengths = _unit(decoded)
        output = decoded_direction * lengths[:, None]
        output[(lengths <= EPSILON) | (decoded_lengths <= EPSILON)] = 0.0
        return output, codes

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        return self.base.artifact_arrays()

    def artifact_metadata(self) -> dict[str, Any]:
        metadata = dict(self.base.artifact_metadata())
        metadata.update(
            name=self.name,
            kind="unit_direction_exact_gain",
            optimizer=self.optimizer,
            parameterization="Delta=ell*u",
            gain_source="stage1_edge_squared_length",
            zero_vector_guard=EPSILON,
        )
        return metadata


class GroupedOAEQuantizer(EdgeQuantizer):
    """Length-conditioned OAE whose encoder group is a static edge property."""

    def __init__(
        self,
        name: str,
        boundaries: np.ndarray,
        models: list[OAEQuantizer],
        metric_kind: str,
    ):
        self.name = name
        self.boundaries = np.asarray(boundaries, dtype=np.float32)
        self.models = models
        self.metric_kind = metric_kind

    @staticmethod
    def groups(lengths: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
        return np.searchsorted(boundaries, lengths, side="right").astype(np.int32)

    @classmethod
    def fit(
        cls,
        vectors: np.ndarray,
        edge_weights: np.ndarray,
        edge_event_inverse: np.ndarray,
        metric_x: np.ndarray,
        metric_weights: np.ndarray,
        spec: dict[str, Any],
        seed: int,
        metric_kind: str,
    ) -> "GroupedOAEQuantizer":
        lengths = np.linalg.norm(np.asarray(vectors, dtype=np.float64), axis=1)
        group_count = int(spec.get("length_groups", 4))
        if group_count < 1:
            raise ValueError("length_groups must be positive")
        boundaries = np.unique(
            np.quantile(lengths, np.linspace(0.0, 1.0, group_count + 1)[1:-1])
        )
        edge_group = cls.groups(lengths, boundaries)
        models: list[OAEQuantizer] = []
        for group in range(len(boundaries) + 1):
            edge_mask = edge_group == group
            event_mask = edge_mask[np.asarray(edge_event_inverse, dtype=np.int64)]
            if np.sum(edge_mask) < int(spec["centroids"]) or not np.any(event_mask):
                raise ValueError(
                    f"length group {group} has insufficient training data; "
                    "reduce length_groups or centroids"
                )
            models.append(
                OAEQuantizer.fit(
                    np.asarray(vectors)[edge_mask],
                    np.asarray(edge_weights)[edge_mask],
                    np.asarray(metric_x)[event_mask],
                    np.asarray(metric_weights)[event_mask],
                    metric_kind,
                    spec,
                    seed + 1009 * group,
                )
            )
        return cls(f"grouped_oae_{metric_kind}", boundaries, models, metric_kind)

    def reconstruct(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        value = np.asarray(vectors, dtype=np.float32)
        groups = self.groups(np.linalg.norm(value.astype(np.float64), axis=1), self.boundaries)
        output = np.empty_like(value)
        code_bytes = self.models[0].pq.code_bytes
        codes = np.empty((len(value), code_bytes), dtype=np.uint8)
        for group, model in enumerate(self.models):
            mask = groups == group
            if np.any(mask):
                output[mask], codes[mask] = model.reconstruct(value[mask])
        return output, codes

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {"length_boundaries": self.boundaries}
        for group, model in enumerate(self.models):
            for key, value in model.artifact_arrays().items():
                arrays[f"group_{group}_{key}"] = value
        return arrays

    def artifact_metadata(self) -> dict[str, Any]:
        first = self.models[0].artifact_metadata()
        return {
            "name": self.name,
            "kind": "length_grouped_oae",
            "metric_kind": self.metric_kind,
            "group_count": len(self.models),
            "group_input": "static_edge_length",
            "subquantizers": first["subquantizers"],
            "centroids": first["centroids"],
            "code_bytes": first["code_bytes"],
            "metadata_float_count": int(self.boundaries.size),
        }


@dataclass
class RefinementResult:
    model: OPQQuantizer
    accepted: bool
    initial_validation_loss: float
    refined_validation_loss: float
    iterations_completed: int


def _event_projection_loss(
    model: EdgeQuantizer,
    edge_vectors: np.ndarray,
    event_edge_inverse: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> float:
    reconstructed, _ = model.reconstruct(edge_vectors)
    estimate = np.einsum("ij,ij->i", x, reconstructed[event_edge_inverse])
    error = np.asarray(y, dtype=np.float64) - estimate.astype(np.float64)
    return float(np.average(error * error, weights=np.asarray(weights, dtype=np.float64)))


def refine_opq_exact_loss(
    initial: OPQQuantizer,
    train_edges: np.ndarray,
    train_event_inverse: np.ndarray,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weights: np.ndarray,
    validation_edges: np.ndarray,
    validation_event_inverse: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    validation_weights: np.ndarray,
    spec: dict[str, Any],
) -> RefinementResult:
    """Block coordinate WLS on OPQ centers, with immutable held-out rollback."""
    rotation = initial.rotation.copy()
    centers = initial.pq.centers.copy()
    pq = ProductQuantizer(initial.pq.subquantizers, initial.pq.centroids, centers)
    candidate = OPQQuantizer(pq, rotation, initial.iterations)
    candidate.name = "opq_operational_refined"
    initial_loss = _event_projection_loss(
        initial, validation_edges, validation_event_inverse, validation_x,
        validation_y, validation_weights,
    )
    codes = initial.pq.encode(np.asarray(train_edges, dtype=np.float32) @ rotation)
    x_rot = np.asarray(train_x, dtype=np.float64) @ rotation.astype(np.float64)
    weights = np.asarray(train_weights, dtype=np.float64)
    target = np.asarray(train_y, dtype=np.float64)
    event_codes = codes[np.asarray(train_event_inverse, dtype=np.int64)]
    block_dim = train_edges.shape[1] // initial.pq.subquantizers
    ridge = float(spec.get("refinement_ridge", 1e-3))
    completed = 0
    for iteration in range(int(spec.get("refinement_iterations", 1))):
        reconstructed_rotated = pq.decode(codes)
        prediction = np.einsum(
            "ij,ij->i", x_rot, reconstructed_rotated[train_event_inverse]
        )
        for block in range(initial.pq.subquantizers):
            begin = block * block_dim
            end = begin + block_dim
            xb = x_rot[:, begin:end]
            current = np.einsum(
                "ij,ij->i", xb, centers[block][event_codes[:, block]]
            )
            partial_target = target - (prediction - current)
            for centroid in range(initial.pq.centroids):
                mask = event_codes[:, block] == centroid
                if not np.any(mask):
                    continue
                xm = xb[mask]
                wm = weights[mask]
                gram = xm.T @ (wm[:, None] * xm)
                scale = max(float(np.trace(gram)) / max(block_dim, 1), 1e-12)
                rhs = xm.T @ (wm * partial_target[mask])
                prior = initial.pq.centers[block, centroid].astype(np.float64)
                centers[block, centroid] = np.linalg.solve(
                    gram + ridge * scale * np.eye(block_dim),
                    rhs + ridge * scale * prior,
                ).astype(np.float32)
            reconstructed_rotated = pq.decode(codes)
            prediction = np.einsum(
                "ij,ij->i", x_rot, reconstructed_rotated[train_event_inverse]
            )
        completed = iteration + 1
    refined_loss = _event_projection_loss(
        candidate, validation_edges, validation_event_inverse, validation_x,
        validation_y, validation_weights,
    )
    tolerance = float(spec.get("rollback_tolerance", 0.0))
    accepted = refined_loss <= initial_loss * (1.0 + tolerance)
    selected = candidate if accepted else initial
    selected.name = "opq_operational_refined" if accepted else "opq_refinement_rollback"
    return RefinementResult(selected, accepted, initial_loss, refined_loss, completed)


def radial_tangential_diagnostics(
    delta: np.ndarray,
    reconstructed: np.ndarray,
    event_inverse: np.ndarray,
    x: np.ndarray,
    margin: np.ndarray,
    length_bins: int,
    margin_bins: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Decompose r=Delta-Delta_hat into radial/tangential components."""
    direction, lengths = _unit(delta)
    residual = np.asarray(delta, dtype=np.float64) - np.asarray(reconstructed, dtype=np.float64)
    radial_scalar = np.einsum("ij,ij->i", residual, direction.astype(np.float64))
    radial = radial_scalar[:, None] * direction
    tangential = residual - radial
    inverse = np.asarray(event_inverse, dtype=np.int64)
    radial_projection = np.einsum("ij,ij->i", x, radial[inverse])
    tangential_projection = np.einsum("ij,ij->i", x, tangential[inverse])

    def metrics(mask: np.ndarray) -> dict[str, float]:
        rp = np.abs(radial_projection[mask])
        tp = np.abs(tangential_projection[mask])
        return {
            "count": int(np.sum(mask)),
            "radial_mse": float(np.mean(radial_projection[mask] ** 2)),
            "tangential_mse": float(np.mean(tangential_projection[mask] ** 2)),
            "radial_abs_q95": float(np.quantile(rp, 0.95)),
            "radial_abs_q99": float(np.quantile(rp, 0.99)),
            "radial_abs_q999": float(np.quantile(rp, 0.999)),
            "tangential_abs_q95": float(np.quantile(tp, 0.95)),
            "tangential_abs_q99": float(np.quantile(tp, 0.99)),
            "tangential_abs_q999": float(np.quantile(tp, 0.999)),
        }

    aggregate = metrics(np.ones(len(x), dtype=bool))
    aggregate.update(
        radial_edge_mse=float(np.mean(np.einsum("ij,ij->i", radial, radial))),
        tangential_edge_mse=float(np.mean(np.einsum("ij,ij->i", tangential, tangential))),
    )
    rows: list[dict[str, Any]] = []
    event_lengths = lengths[inverse]
    dimensions = [
        ("edge_length", event_lengths, length_bins),
        ("absolute_margin", np.abs(np.asarray(margin, dtype=np.float64)), margin_bins),
    ]
    for dimension, values, bins in dimensions:
        boundaries = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
        if len(boundaries) < 2:
            boundaries = np.asarray([values.min(), values.max() + EPSILON])
        assignments = np.minimum(
            np.searchsorted(boundaries[1:-1], values, side="right"), len(boundaries) - 2
        )
        for index in range(len(boundaries) - 1):
            mask = assignments == index
            if np.any(mask):
                rows.append(
                    {
                        "stratum": dimension,
                        "bin": index,
                        "lower": float(boundaries[index]),
                        "upper": float(boundaries[index + 1]),
                        **metrics(mask),
                    }
                )
    return aggregate, rows


def operational_loss_decomposition(
    delta: np.ndarray,
    reconstructed: np.ndarray,
    event_inverse: np.ndarray,
    x: np.ndarray,
    subquantizers: int,
) -> dict[str, float]:
    """Expose within-block and cross-block terms of the exact projection loss."""
    residual = np.asarray(delta, dtype=np.float64) - np.asarray(reconstructed, dtype=np.float64)
    event_residual = residual[np.asarray(event_inverse, dtype=np.int64)]
    value = np.asarray(x, dtype=np.float64)
    if value.shape[1] % subquantizers:
        raise ValueError("subquantizers must divide dimension for loss decomposition")
    block_dim = value.shape[1] // subquantizers
    components = np.empty((len(value), subquantizers), dtype=np.float64)
    for block in range(subquantizers):
        begin = block * block_dim
        end = begin + block_dim
        components[:, block] = np.einsum(
            "ij,ij->i", value[:, begin:end], event_residual[:, begin:end]
        )
    full = np.sum(components, axis=1) ** 2
    within = np.sum(components * components, axis=1)
    return {
        "exact_full_projection_loss": float(np.mean(full)),
        "within_block_projection_loss": float(np.mean(within)),
        "cross_block_projection_term": float(np.mean(full - within)),
    }


def fit_stage21_quantizer(
    method: str,
    train_vectors: np.ndarray,
    edge_weights: np.ndarray,
    edge_event_inverse: np.ndarray,
    metric_x: np.ndarray,
    metric_y: np.ndarray,
    metric_weights: np.ndarray,
    spec: dict[str, Any],
    seed: int,
    validation: dict[str, np.ndarray] | None = None,
) -> tuple[EdgeQuantizer, dict[str, Any]]:
    details: dict[str, Any] = {}
    if method == "direct_delta_pq":
        model = FlatPQQuantizer.fit(method, train_vectors, edge_weights, spec, seed)
    elif method == "direct_delta_opq":
        model = OPQQuantizer.fit(train_vectors, edge_weights, spec, seed)
    elif method in {"norm_corrected_pq", "norm_corrected_opq"}:
        base = (
            FlatPQQuantizer.fit("direct_delta_pq", train_vectors, edge_weights, spec, seed)
            if method.endswith("_pq")
            else OPQQuantizer.fit(train_vectors, edge_weights, spec, seed)
        )
        model = ExactLengthQuantizer(method, base)
    elif method in {"direction_pq_exact_length", "gain_shape_opq"}:
        optimizer = "pq" if method.startswith("direction_pq") else "opq"
        model = UnitDirectionQuantizer.fit(
            method, train_vectors, edge_weights, spec, seed, optimizer
        )
    elif method.startswith("grouped_oae_"):
        model = GroupedOAEQuantizer.fit(
            train_vectors, edge_weights, edge_event_inverse, metric_x,
            metric_weights, spec, seed, method.removeprefix("grouped_oae_")
        )
    elif method == "opq_operational_refined":
        if validation is None:
            raise ValueError("operational refinement requires held-out validation")
        initial = OPQQuantizer.fit(train_vectors, edge_weights, spec, seed)
        result = refine_opq_exact_loss(
            initial, train_vectors, edge_event_inverse, metric_x, metric_y,
            metric_weights, validation["edge_vectors"], validation["event_inverse"],
            validation["x"], validation["y"], validation["weights"], spec,
        )
        model = result.model
        details["refinement"] = {
            "accepted": result.accepted,
            "initial_validation_loss": result.initial_validation_loss,
            "refined_validation_loss": result.refined_validation_loss,
            "iterations_completed": result.iterations_completed,
        }
    else:
        raise ValueError(f"unknown Stage 2.1 method: {method}")
    return model, details
