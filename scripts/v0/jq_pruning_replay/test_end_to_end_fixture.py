#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from jq_quantizer import JQConfig, JQQuantizer
from v0_bound_replay import evaluate_v0_bound


class EndToEndFixtureTest(unittest.TestCase):
    def test_selected_cells_feed_a_safe_v0_bound(self) -> None:
        rng = np.random.default_rng(29)
        dimension = 8
        base = rng.normal(size=(40, dimension)).astype(np.float32)
        query = rng.normal(size=(12, dimension)).astype(np.float32)
        current_id = np.arange(12, dtype=np.int64)
        candidate_id = np.arange(12, dtype=np.int64) + 17
        source = base[current_id]
        target = base[candidate_id]
        difference = target.astype(np.float64) - source.astype(np.float64)
        length = np.linalg.norm(difference, axis=1)
        direction = (difference / length[:, None]).astype(np.float32)

        quantizer = JQQuantizer(
            JQConfig(dimension=dimension, subspaces=2, rotation_seed=17)
        )
        rotated_direction = quantizer.rotate(direction)
        # Add held-in graph directions just as the frozen diagnostic does.
        training = rng.normal(size=(96, dimension)).astype(np.float32)
        training /= np.linalg.norm(training, axis=1)[:, None]
        quantizer.fit_rotated(quantizer.rotate(training))
        codes = quantizer.encode_rotated(rotated_direction)
        error = quantizer.direction_error_upper(rotated_direction, codes)

        residual = query.astype(np.float64) - source.astype(np.float64)
        rotated_residual = residual @ quantizer.rotation.astype(np.float64).T
        dot_upper = quantizer.selected_inner_product_upper(rotated_residual, codes)
        current_squared = np.sum(residual * residual, axis=1)
        result = evaluate_v0_bound(
            current_squared,
            length,
            dot_upper,
            error,
            dimension=dimension,
        )
        exact = np.sum((query.astype(np.float64) - target.astype(np.float64)) ** 2, axis=1)
        self.assertTrue(np.all(result.valid))
        self.assertTrue(np.all(result.lower_bound <= exact + 1e-10))


if __name__ == "__main__":
    unittest.main()
