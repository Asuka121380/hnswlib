#!/usr/bin/env python3
"""Core numerics for the preregistered OAE edge/query-dependence test.

The module deliberately contains no HNSW or quantizer training code.  Query
vectors are used only as response variables; predictors are frozen functions
of static edge geometry.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class Fvecs:
    """Read-only, zero-copy fvecs accessor with complete header validation."""

    def __init__(self, path: Path, dimension: int):
        self.path = Path(path)
        self.dimension = int(dimension)
        require(self.path.is_file(), f"missing fvecs file: {self.path}")
        record_bytes = 4 * (self.dimension + 1)
        require(self.path.stat().st_size % record_bytes == 0, "fvecs size is not record aligned")
        self.count = self.path.stat().st_size // record_bytes
        self._raw = np.memmap(self.path, dtype="<i4", mode="r", shape=(self.count, self.dimension + 1))
        require(bool(np.all(self._raw[:, 0] == self.dimension)), "invalid fvecs dimension header")

    def rows(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        require(indices.ndim == 1, "fvec row indices must be one dimensional")
        require(indices.size == 0 or (indices.min() >= 0 and indices.max() < self.count), "fvec row index out of range")
        # The payload bits are float32 even though the record header is int32.
        return self._raw[indices, 1:].view("<f4")

    def close(self) -> None:
        mapped = getattr(self._raw, "_mmap", None)
        if mapped is not None:
            mapped.close()


@dataclass(frozen=True)
class SearchEvents:
    query_id: np.ndarray
    current_label: np.ndarray
    neighbor_label: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.query_id)
        require(n > 0, "search-event dataset is empty")
        require(len(self.current_label) == n and len(self.neighbor_label) == n, "event columns have unequal length")
        require(bool(np.all(self.current_label != self.neighbor_label)), "self edges are not valid search events")

    @property
    def size(self) -> int:
        return len(self.query_id)

    @classmethod
    def load(cls, path: Path) -> "SearchEvents":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                np.asarray(data["query_id"], dtype=np.int64),
                np.asarray(data["current_label"], dtype=np.int64),
                np.asarray(data["neighbor_label"], dtype=np.int64),
            )


@dataclass(frozen=True)
class GroupModel:
    probe: np.ndarray
    direction_centers: np.ndarray
    length_boundaries: np.ndarray
    mu_global: np.ndarray
    mu_group: np.ndarray
    a_global: np.ndarray
    a_group: np.ndarray
    group_effective_count: np.ndarray


def fixed_probe(dimension: int, probe_count: int, seed: int) -> np.ndarray:
    require(0 < probe_count <= dimension, "probe count must be in [1, dimension]")
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((dimension, probe_count)))
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return np.asarray(q * signs, dtype=np.float32)


def unique_edge_pairs(events: SearchEvents) -> tuple[np.ndarray, np.ndarray]:
    require(events.current_label.max(initial=0) < 2**32 and events.neighbor_label.max(initial=0) < 2**32,
            "edge labels exceed the packed-key limit")
    keys = (events.current_label.astype(np.uint64) << np.uint64(32)) | events.neighbor_label.astype(np.uint64)
    unique = np.unique(keys)
    return (unique >> np.uint64(32)).astype(np.int64), (unique & np.uint64(0xFFFFFFFF)).astype(np.int64)


def edge_geometry(base: Fvecs, current: np.ndarray, neighbor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = base.rows(neighbor).astype(np.float32) - base.rows(current).astype(np.float32)
    lengths = np.linalg.norm(delta, axis=1)
    require(bool(np.all(np.isfinite(lengths))), "non-finite edge length")
    require(bool(np.all(lengths > 0.0)), "zero-length edge encountered")
    return delta / lengths[:, None], lengths


def fit_spherical_kmeans(directions: np.ndarray, clusters: int, iterations: int, seed: int) -> np.ndarray:
    require(len(directions) >= clusters, "fewer unique edges than direction clusters")
    require(iterations > 0, "k-means iterations must be positive")
    rng = np.random.default_rng(seed)
    centers = directions[rng.choice(len(directions), size=clusters, replace=False)].astype(np.float64)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    for _ in range(iterations):
        labels = np.argmax(directions @ centers.T, axis=1)
        updated = np.zeros_like(centers)
        np.add.at(updated, labels, directions)
        norms = np.linalg.norm(updated, axis=1)
        nonempty = norms > 0.0
        updated[nonempty] /= norms[nonempty, None]
        updated[~nonempty] = centers[~nonempty]
        centers = updated
    return centers.astype(np.float32)


def assign_groups(directions: np.ndarray, lengths: np.ndarray, centers: np.ndarray,
                  boundaries: np.ndarray) -> np.ndarray:
    direction_group = np.argmax(directions @ centers.T, axis=1)
    length_group = np.searchsorted(boundaries, lengths, side="right")
    return direction_group * (len(boundaries) + 1) + length_group


def _query_index_and_weight(query_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids, inverse, counts = np.unique(query_id, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    return ids, inverse, weights


def iter_event_batches(events: SearchEvents, base: Fvecs, queries: Fvecs,
                       centers: np.ndarray, boundaries: np.ndarray, batch_size: int
                       ) -> Iterator[tuple[slice, np.ndarray, np.ndarray, np.ndarray]]:
    require(batch_size > 0, "batch size must be positive")
    for start in range(0, events.size, batch_size):
        stop = min(events.size, start + batch_size)
        sl = slice(start, stop)
        current = events.current_label[sl]
        neighbor = events.neighbor_label[sl]
        c = base.rows(current).astype(np.float32)
        x = queries.rows(events.query_id[sl]).astype(np.float32) - c
        delta = base.rows(neighbor).astype(np.float32) - c
        lengths = np.linalg.norm(delta, axis=1)
        require(bool(np.all(lengths > 0.0)), "zero-length event edge")
        groups = assign_groups(delta / lengths[:, None], lengths, centers, boundaries)
        yield sl, x, delta, groups


def fit_model(events: SearchEvents, base: Fvecs, queries: Fvecs, *, probe_count: int,
              direction_clusters: int, length_bins: int, shrinkage: float,
              probe_seed: int, group_seed: int, kmeans_iterations: int,
              group_fit_edge_cap: int, batch_size: int) -> GroupModel:
    require(events.query_id.max() < queries.count, "training query id exceeds query file")
    current, neighbor = unique_edge_pairs(events)
    rng = np.random.default_rng(group_seed)
    if group_fit_edge_cap > 0 and len(current) > group_fit_edge_cap:
        chosen = np.sort(rng.choice(len(current), size=group_fit_edge_cap, replace=False))
        fit_current, fit_neighbor = current[chosen], neighbor[chosen]
    else:
        fit_current, fit_neighbor = current, neighbor
    fit_directions, _ = edge_geometry(base, fit_current, fit_neighbor)
    centers = fit_spherical_kmeans(fit_directions, direction_clusters, kmeans_iterations, group_seed)

    all_lengths: list[np.ndarray] = []
    for start in range(0, len(current), batch_size):
        _, lengths = edge_geometry(base, current[start:start + batch_size], neighbor[start:start + batch_size])
        all_lengths.append(lengths)
    lengths = np.concatenate(all_lengths)
    boundaries = np.quantile(lengths, np.arange(1, length_bins) / length_bins).astype(np.float32)
    require(bool(np.all(np.diff(boundaries) > 0.0)), "length quantiles are not strictly increasing")

    probe = fixed_probe(base.dimension, probe_count, probe_seed)
    group_count = direction_clusters * length_bins
    _, _, weights = _query_index_and_weight(events.query_id)
    y_global_sum = np.zeros(probe_count, dtype=np.float64)
    y_group_sum = np.zeros((group_count, probe_count), dtype=np.float64)
    a_global_sum = np.zeros(base.dimension, dtype=np.float64)
    a_group_sum = np.zeros((group_count, base.dimension), dtype=np.float64)
    group_weight = np.zeros(group_count, dtype=np.float64)
    total_weight = 0.0
    for sl, x, _, groups in iter_event_batches(events, base, queries, centers, boundaries, batch_size):
        w = weights[sl]
        y = np.square(x @ probe, dtype=np.float64)
        x2 = np.square(x, dtype=np.float64)
        y_global_sum += np.sum(y * w[:, None], axis=0)
        a_global_sum += np.sum(x2 * w[:, None], axis=0)
        np.add.at(y_group_sum, groups, y * w[:, None])
        np.add.at(a_group_sum, groups, x2 * w[:, None])
        np.add.at(group_weight, groups, w)
        total_weight += float(w.sum())
    mu_global = y_global_sum / total_weight
    a_global = a_global_sum / total_weight
    raw_mu = np.divide(y_group_sum, group_weight[:, None], out=np.zeros_like(y_group_sum), where=group_weight[:, None] > 0)
    raw_a = np.divide(a_group_sum, group_weight[:, None], out=np.zeros_like(a_group_sum), where=group_weight[:, None] > 0)
    lam = group_weight / (group_weight + shrinkage)
    mu_group = lam[:, None] * raw_mu + (1.0 - lam[:, None]) * mu_global
    a_group = lam[:, None] * raw_a + (1.0 - lam[:, None]) * a_global
    return GroupModel(probe, centers, boundaries, mu_global, mu_group, a_global, a_group, group_weight)


def bootstrap_mean_ci(values: np.ndarray, repetitions: int, seed: int) -> tuple[float, float]:
    require(repetitions > 0, "bootstrap repetitions must be positive")
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 128):
        count = min(128, repetitions - start)
        sample = rng.integers(0, len(values), size=(count, len(values)))
        means[start:start + count] = values[sample].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def evaluate_primary(events: SearchEvents, base: Fvecs, queries: Fvecs, model: GroupModel,
                     *, batch_size: int, bootstrap_repetitions: int, bootstrap_seed: int,
                     relative_gate: float, minimum_noninferior_probes: int,
                     maximum_query_contribution: float) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    query_ids, inverse, weights = _query_index_and_weight(events.query_id)
    qn, pn = len(query_ids), model.probe.shape[1]
    global_error = np.zeros(qn)
    conditional_error = np.zeros(qn)
    global_probe = np.zeros((qn, pn))
    conditional_probe = np.zeros((qn, pn))
    for sl, x, _, groups in iter_event_batches(events, base, queries, model.direction_centers,
                                                model.length_boundaries, batch_size):
        y = np.square(x @ model.probe, dtype=np.float64)
        eg = np.square(y - model.mu_global)
        ec = np.square(y - model.mu_group[groups])
        qi, w = inverse[sl], weights[sl]
        np.add.at(global_error, qi, eg.mean(axis=1) * w)
        np.add.at(conditional_error, qi, ec.mean(axis=1) * w)
        np.add.at(global_probe, qi, eg * w[:, None])
        np.add.at(conditional_probe, qi, ec * w[:, None])
    improvement = global_error - conditional_error
    mean_global, mean_conditional = float(global_error.mean()), float(conditional_error.mean())
    relative = (mean_global - mean_conditional) / mean_global
    ci_low, ci_high = bootstrap_mean_ci(improvement, bootstrap_repetitions, bootstrap_seed)
    noninferior = int(np.count_nonzero(conditional_probe.mean(axis=0) <= global_probe.mean(axis=0)))
    denominator = float(np.abs(improvement).sum())
    max_contribution = float(np.abs(improvement).max() / denominator) if denominator else 1.0
    checks = {
        "relative_improvement": relative >= relative_gate,
        "bootstrap_ci_lower_positive": ci_low > 0.0,
        "minimum_noninferior_probes": noninferior >= minimum_noninferior_probes,
        "maximum_single_query_contribution": max_contribution <= maximum_query_contribution,
    }
    if all(checks.values()):
        status = "PASS_EDGE_QUERY_DEPENDENCE"
    elif relative > 0.0 and ci_low <= 0.0 <= ci_high:
        status = "INCONCLUSIVE_EDGE_QUERY_DEPENDENCE"
    else:
        status = "NO_EVIDENCE_EDGE_QUERY_DEPENDENCE"
    metrics = {"query_id": query_ids, "global_error": global_error,
               "conditional_error": conditional_error, "improvement": improvement}
    report = {
        "status": status,
        "query_count": qn,
        "event_count": events.size,
        "mean_global_error": mean_global,
        "mean_conditional_error": mean_conditional,
        "relative_improvement": relative,
        "mean_improvement_ci95": [ci_low, ci_high],
        "noninferior_probe_count": noninferior,
        "probe_count": pn,
        "maximum_single_query_absolute_contribution_fraction": max_contribution,
        "criteria": checks,
    }
    return metrics, report


def reconstruct_pq(directions: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Exact standard-PQ nearest-centroid reconstruction, vectorized by subspace."""
    m, ksub, dsub = centroids.shape
    require(directions.shape[1] == m * dsub, "PQ dimension mismatch")
    output = np.empty_like(directions, dtype=np.float32)
    for sub in range(m):
        x = directions[:, sub * dsub:(sub + 1) * dsub]
        c = centroids[sub]
        distances = np.sum(x * x, axis=1)[:, None] + np.sum(c * c, axis=1)[None, :] - 2.0 * (x @ c.T)
        output[:, sub * dsub:(sub + 1) * dsub] = c[np.argmin(distances, axis=1)]
    return output


def evaluate_secondary(events: SearchEvents, base: Fvecs, queries: Fvecs, model: GroupModel,
                       centroids: np.ndarray, *, batch_size: int, bootstrap_repetitions: int,
                       bootstrap_seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    query_ids, inverse, weights = _query_index_and_weight(events.query_id)
    global_error = np.zeros(len(query_ids))
    conditional_error = np.zeros(len(query_ids))
    for sl, x, delta, groups in iter_event_batches(events, base, queries, model.direction_centers,
                                                    model.length_boundaries, batch_size):
        lengths = np.linalg.norm(delta, axis=1)
        reconstructed = reconstruct_pq(delta / lengths[:, None], centroids) * lengths[:, None]
        residual = delta - reconstructed
        observed = np.square(np.sum(x * residual, axis=1, dtype=np.float64))
        residual2 = np.square(residual, dtype=np.float64)
        predicted_global = residual2 @ model.a_global
        predicted_conditional = np.sum(residual2 * model.a_group[groups], axis=1)
        eg = np.square(observed - predicted_global)
        ec = np.square(observed - predicted_conditional)
        qi, w = inverse[sl], weights[sl]
        np.add.at(global_error, qi, eg * w)
        np.add.at(conditional_error, qi, ec * w)
    improvement = global_error - conditional_error
    mean_global, mean_conditional = float(global_error.mean()), float(conditional_error.mean())
    relative = (mean_global - mean_conditional) / mean_global
    ci_low, ci_high = bootstrap_mean_ci(improvement, bootstrap_repetitions, bootstrap_seed)
    status = "PASS_V0_RESIDUAL_RELEVANCE" if relative > 0.0 and ci_low > 0.0 else (
        "INCONCLUSIVE_V0_RESIDUAL_RELEVANCE" if relative > 0.0 and ci_low <= 0.0 <= ci_high
        else "NO_EVIDENCE_V0_RESIDUAL_RELEVANCE")
    metrics = {"query_id": query_ids, "global_mse": global_error,
               "conditional_mse": conditional_error, "improvement": improvement}
    report = {"status": status, "query_count": len(query_ids), "event_count": events.size,
              "mean_global_mse": mean_global, "mean_conditional_mse": mean_conditional,
              "relative_improvement": relative, "mean_improvement_ci95": [ci_low, ci_high]}
    return metrics, report


def write_metrics_parquet(path: Path, metrics: dict[str, np.ndarray]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValidationError("writing required Parquet outputs requires pyarrow") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(metrics), path, compression="zstd")
