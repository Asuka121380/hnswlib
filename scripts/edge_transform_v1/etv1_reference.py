"""High-precision reference geometry for Edge Transform V1.

This module is intentionally independent from hnswlib control flow.  It is the
oracle used by Phase A/B validation and must remain simpler than the eventual
C++ implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class EdgeGeometry:
    length: float
    direction: Array | None
    zero_length: bool


@dataclass(frozen=True)
class BlockSupport:
    block: slice
    query_norm: float
    true_norm_upper: float
    residual_norm_upper: float
    norm_support: float
    reconstruction_support: float
    selected_support: float


@dataclass(frozen=True)
class BoundResult:
    lower_bound: float
    exact_distance: float
    support_upper: float
    transform_defect_padding: float
    block_supports: tuple[BlockSupport, ...]
    valid: bool
    status: str

    @property
    def violation(self) -> float:
        return self.lower_bound - self.exact_distance


def _vector(value: Array | Sequence[float], name: str) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D vector")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains a non-finite value")
    return result


def equal_blocks(dimension: int, count: int) -> tuple[slice, ...]:
    """Return contiguous non-empty blocks, allowing d % B != 0."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if count <= 0 or count > dimension:
        raise ValueError("block count must be in [1, dimension]")
    starts = np.linspace(0, dimension, count + 1, dtype=np.int64)
    blocks = tuple(slice(int(starts[i]), int(starts[i + 1])) for i in range(count))
    if any(block.start == block.stop for block in blocks):
        raise AssertionError("equal_blocks produced an empty block")
    return blocks


def validate_blocks(blocks: Iterable[slice], dimension: int) -> tuple[slice, ...]:
    normalized = tuple(blocks)
    cursor = 0
    for block in normalized:
        if not isinstance(block, slice) or block.step not in (None, 1):
            raise ValueError("blocks must be unit-stride slices")
        if block.start != cursor or block.stop is None or block.stop <= block.start:
            raise ValueError("blocks must be non-empty, contiguous, and ordered")
        cursor = block.stop
    if not normalized or cursor != dimension:
        raise ValueError("blocks must partition the full vector")
    return normalized


def edge_geometry(c: Array | Sequence[float], v: Array | Sequence[float]) -> EdgeGeometry:
    c64 = _vector(c, "c")
    v64 = _vector(v, "v")
    if c64.shape != v64.shape:
        raise ValueError("c and v must have the same shape")
    edge = v64 - c64
    length = float(np.linalg.norm(edge))
    if not np.isfinite(length):
        raise ValueError("edge length is non-finite")
    if length == 0.0:
        return EdgeGeometry(length=0.0, direction=None, zero_length=True)
    return EdgeGeometry(length=length, direction=edge / length, zero_length=False)


def pivot_offset(p: Array, c: Array, direction: Array) -> float:
    p64 = _vector(p, "p")
    c64 = _vector(c, "c")
    u64 = _vector(direction, "direction")
    if p64.shape != c64.shape or p64.shape != u64.shape:
        raise ValueError("p, c, and direction must have the same shape")
    return float(np.dot(p64 - c64, u64))


def orthogonality_defect(transform: Array) -> float:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transform must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("transform contains a non-finite value")
    gram_error = np.eye(matrix.shape[0], dtype=np.float64) - matrix.T @ matrix
    return float(np.linalg.norm(gram_error, ord=2))


def interval_length_term(support_upper: float, length_lower: float, length_upper: float) -> float:
    """min_{ell in [lo,hi]} ell^2 - 2 ell support_upper."""
    values = (support_upper, length_lower, length_upper)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("interval inputs must be finite")
    if length_lower < 0.0 or length_upper < length_lower:
        raise ValueError("invalid non-negative edge-length interval")
    minimizer = float(np.clip(support_upper, length_lower, length_upper))
    return minimizer * minimizer - 2.0 * minimizer * support_upper


def compute_block_supports(
    w: Array,
    y: Array,
    y_hat: Array,
    blocks: Iterable[slice],
    *,
    residual_norm_uppers: Sequence[float] | None = None,
    true_norm_uppers: Sequence[float] | None = None,
    certificate_tolerance: float = 1e-12,
) -> tuple[BlockSupport, ...]:
    w64 = _vector(w, "w")
    y64 = _vector(y, "y")
    yhat64 = _vector(y_hat, "y_hat")
    if w64.shape != y64.shape or w64.shape != yhat64.shape:
        raise ValueError("w, y, and y_hat must have the same shape")
    normalized_blocks = validate_blocks(blocks, w64.size)
    if residual_norm_uppers is not None and len(residual_norm_uppers) != len(normalized_blocks):
        raise ValueError("one residual certificate is required per block")
    if true_norm_uppers is not None and len(true_norm_uppers) != len(normalized_blocks):
        raise ValueError("one true-norm certificate is required per block")

    result: list[BlockSupport] = []
    for index, block in enumerate(normalized_blocks):
        wb = w64[block]
        yb = y64[block]
        yhatb = yhat64[block]
        query_norm = float(np.linalg.norm(wb))
        actual_true_norm = float(np.linalg.norm(yb))
        actual_residual_norm = float(np.linalg.norm(yb - yhatb))
        true_upper = (
            actual_true_norm
            if true_norm_uppers is None
            else float(true_norm_uppers[index])
        )
        residual_upper = (
            actual_residual_norm
            if residual_norm_uppers is None
            else float(residual_norm_uppers[index])
        )
        if not np.isfinite(true_upper) or not np.isfinite(residual_upper):
            raise ValueError("block certificate is non-finite")
        if true_upper + certificate_tolerance < actual_true_norm:
            raise ValueError("true-norm certificate is not an upper bound")
        if residual_upper + certificate_tolerance < actual_residual_norm:
            raise ValueError("residual certificate is not an upper bound")
        if true_upper < 0.0 or residual_upper < 0.0:
            raise ValueError("block certificates must be non-negative")
        norm_support = query_norm * true_upper
        reconstruction_support = float(np.dot(wb, yhatb)) + query_norm * residual_upper
        selected_support = min(norm_support, reconstruction_support)
        result.append(
            BlockSupport(
                block=block,
                query_norm=query_norm,
                true_norm_upper=true_upper,
                residual_norm_upper=residual_upper,
                norm_support=norm_support,
                reconstruction_support=reconstruction_support,
                selected_support=selected_support,
            )
        )
    return tuple(result)


def block_certified_lower_bound(
    q: Array,
    c: Array,
    v: Array,
    p: Array,
    transform: Array,
    y_hat: Array,
    blocks: Iterable[slice],
    *,
    offset_upper: float | None = None,
    residual_norm_uppers: Sequence[float] | None = None,
    true_norm_uppers: Sequence[float] | None = None,
    transform_defect_upper: float | None = None,
    length_interval: tuple[float, float] | None = None,
) -> BoundResult:
    q64 = _vector(q, "q")
    c64 = _vector(c, "c")
    v64 = _vector(v, "v")
    p64 = _vector(p, "p")
    if not (q64.shape == c64.shape == v64.shape == p64.shape):
        raise ValueError("q, c, v, and p must have the same shape")
    geometry = edge_geometry(c64, v64)
    current_distance = float(np.dot(q64 - c64, q64 - c64))
    exact_distance = float(np.dot(q64 - v64, q64 - v64))
    if geometry.zero_length:
        return BoundResult(
            lower_bound=current_distance,
            exact_distance=exact_distance,
            support_upper=0.0,
            transform_defect_padding=0.0,
            block_supports=(),
            valid=True,
            status="zero_length_exact",
        )

    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (q64.size, q64.size) or not np.all(np.isfinite(matrix)):
        raise ValueError("transform has an invalid shape or value")
    direction = geometry.direction
    assert direction is not None
    x = q64 - p64
    w = matrix @ x
    y = matrix @ direction
    yhat64 = _vector(y_hat, "y_hat")
    supports = compute_block_supports(
        w,
        y,
        yhat64,
        blocks,
        residual_norm_uppers=residual_norm_uppers,
        true_norm_uppers=true_norm_uppers,
    )
    exact_offset = pivot_offset(p64, c64, direction)
    stored_offset_upper = exact_offset if offset_upper is None else float(offset_upper)
    if not np.isfinite(stored_offset_upper) or stored_offset_upper < exact_offset:
        raise ValueError("offset_upper must upper-bound the exact pivot offset")
    actual_defect = orthogonality_defect(matrix)
    certified_defect = actual_defect if transform_defect_upper is None else float(transform_defect_upper)
    if not np.isfinite(certified_defect) or certified_defect < actual_defect:
        raise ValueError("transform_defect_upper is not a valid upper bound")
    defect_padding = certified_defect * float(np.linalg.norm(x))
    support_upper = stored_offset_upper + sum(item.selected_support for item in supports) + defect_padding

    if length_interval is None:
        length_term = geometry.length * geometry.length - 2.0 * geometry.length * support_upper
    else:
        lo, hi = length_interval
        if not (lo <= geometry.length <= hi):
            raise ValueError("length interval does not contain the exact edge length")
        length_term = interval_length_term(support_upper, lo, hi)
    lower_bound = max(0.0, current_distance + length_term)
    return BoundResult(
        lower_bound=lower_bound,
        exact_distance=exact_distance,
        support_upper=support_upper,
        transform_defect_padding=defect_padding,
        block_supports=supports,
        valid=True,
        status="valid",
    )


def progressive_lower_bounds(
    current_distance: float,
    length: float,
    offset_upper: float,
    supports: Sequence[BlockSupport],
    reveal_order: Sequence[int],
    *,
    extra_support_padding: float = 0.0,
) -> tuple[float, ...]:
    if sorted(reveal_order) != list(range(len(supports))):
        raise ValueError("reveal_order must be a permutation of block indices")
    revealed: set[int] = set()
    result: list[float] = []
    for index in (-1, *reveal_order):
        if index >= 0:
            revealed.add(index)
        support = offset_upper + extra_support_padding
        support += sum(
            item.selected_support if block_index in revealed else item.norm_support
            for block_index, item in enumerate(supports)
        )
        result.append(max(0.0, current_distance + length * length - 2.0 * length * support))
    return tuple(result)

