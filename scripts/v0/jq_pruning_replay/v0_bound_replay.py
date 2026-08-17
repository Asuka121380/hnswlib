#!/usr/bin/env python3
"""Vectorized port of the V0 conservative bound arithmetic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _next_up(value: np.ndarray | float) -> np.ndarray:
    return np.nextafter(np.asarray(value, dtype=np.float64), np.inf)


def _next_down(value: np.ndarray | float) -> np.ndarray:
    return np.nextafter(np.asarray(value, dtype=np.float64), -np.inf)


def _add_up(left, right) -> np.ndarray:
    return _next_up(np.asarray(left, dtype=np.float64) + np.asarray(right, dtype=np.float64))


def _add_down(left, right) -> np.ndarray:
    return _next_down(
        np.asarray(left, dtype=np.float64) + np.asarray(right, dtype=np.float64)
    )


def _multiply_up(left, right) -> np.ndarray:
    return _next_up(
        np.asarray(left, dtype=np.float64) * np.asarray(right, dtype=np.float64)
    )


def _multiply_down(left, right) -> np.ndarray:
    return _next_down(
        np.asarray(left, dtype=np.float64) * np.asarray(right, dtype=np.float64)
    )


def float_squared_l2_padding_upper(
    observed_current_squared_distance: np.ndarray,
    edge_length: np.ndarray,
    dimension: int,
) -> np.ndarray:
    operation_count = 8.0 * float(dimension) + 64.0
    scaled_epsilon = operation_count * float(np.finfo(np.float32).eps)
    if not np.isfinite(scaled_epsilon) or scaled_epsilon >= 1.0:
        raise ValueError("V0 float32 L2 rounding model is invalid")
    gamma = _next_up(scaled_epsilon / (1.0 - scaled_epsilon))
    current_upper = _next_up(observed_current_squared_distance / (1.0 - gamma))
    target_norm_upper = _add_up(_next_up(np.sqrt(current_upper)), _next_up(edge_length))
    target_squared_upper = _multiply_up(target_norm_upper, target_norm_upper)
    return _multiply_up(gamma, target_squared_upper)


@dataclass(frozen=True)
class BoundReplayResult:
    valid: np.ndarray
    length_squared_lower: np.ndarray
    cross_term_upper: np.ndarray
    approximate_squared_distance: np.ndarray
    direction_error_radius: np.ndarray
    operational_l2_padding: np.ndarray
    error_radius: np.ndarray
    lower_bound: np.ndarray

    def would_prune(self, threshold: np.ndarray) -> np.ndarray:
        values = np.asarray(threshold, dtype=np.float64)
        return self.valid & np.isfinite(values) & (values >= 0.0) & (
            self.lower_bound > values
        )


def evaluate_v0_bound(
    current_squared_distance: np.ndarray,
    edge_length: np.ndarray,
    residual_direction_inner_product_upper: np.ndarray,
    direction_error_upper: np.ndarray,
    *,
    dimension: int,
    stored_numeric_padding: np.ndarray | float = 0.0,
) -> BoundReplayResult:
    current = np.asarray(current_squared_distance, dtype=np.float64)
    length = np.asarray(edge_length, dtype=np.float64)
    dot_upper = np.asarray(residual_direction_inner_product_upper, dtype=np.float64)
    error = np.asarray(direction_error_upper, dtype=np.float64)
    stored = np.broadcast_to(
        np.asarray(stored_numeric_padding, dtype=np.float64), current.shape
    )
    if not (current.shape == length.shape == dot_upper.shape == error.shape == stored.shape):
        raise ValueError("bound inputs must have identical shapes")
    valid = (
        np.isfinite(current)
        & (current >= 0.0)
        & np.isfinite(length)
        & (length > 0.0)
        & np.isfinite(dot_upper)
        & np.isfinite(error)
        & (error >= 0.0)
        & np.isfinite(stored)
        & (stored >= 0.0)
    )

    length_squared_lower = _multiply_down(length, length)
    twice_length = 2.0 * length
    cross_term_upper = _multiply_up(twice_length, dot_upper)
    base_plus_length_lower = _add_down(current, length_squared_lower)
    approximate = _add_down(base_plus_length_lower, -cross_term_upper)
    root_upper = _next_up(np.sqrt(np.maximum(current, 0.0)))
    direction_radius = _multiply_up(_multiply_up(twice_length, root_upper), error)
    error_radius = _add_up(direction_radius, stored)
    operational = float_squared_l2_padding_upper(current, length, dimension)
    error_radius = _add_up(error_radius, operational)
    lower = _add_down(approximate, -error_radius)
    lower = np.where(lower > 0.0, lower, 0.0)

    finite_outputs = (
        np.isfinite(length_squared_lower)
        & np.isfinite(cross_term_upper)
        & np.isfinite(approximate)
        & np.isfinite(direction_radius)
        & (direction_radius >= 0.0)
        & np.isfinite(operational)
        & (operational >= 0.0)
        & np.isfinite(error_radius)
        & (error_radius >= 0.0)
        & np.isfinite(lower)
    )
    valid &= finite_outputs
    return BoundReplayResult(
        valid=valid,
        length_squared_lower=length_squared_lower,
        cross_term_upper=cross_term_upper,
        approximate_squared_distance=approximate,
        direction_error_radius=direction_radius,
        operational_l2_padding=operational,
        error_radius=error_radius,
        lower_bound=lower,
    )


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("invalid Wilson interval counts")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, float(center - radius)), min(1.0, float(center + radius))
