#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from v0_bound_replay import evaluate_v0_bound, wilson_interval


class V0BoundReplayTest(unittest.TestCase):
    def test_exact_direction_bound_does_not_exceed_exact_distance(self) -> None:
        rng = np.random.default_rng(13)
        dimension = 16
        q = rng.normal(size=(100, dimension))
        c = rng.normal(size=(100, dimension))
        d = rng.normal(size=(100, dimension))
        edge = d - c
        length = np.linalg.norm(edge, axis=1)
        direction = edge / length[:, None]
        residual = q - c
        current = np.sum(residual * residual, axis=1)
        dot = np.sum(residual * direction, axis=1)
        exact = np.sum((q - d) ** 2, axis=1)
        result = evaluate_v0_bound(
            current,
            length,
            np.nextafter(dot, np.inf),
            np.zeros(100),
            dimension=dimension,
        )
        self.assertTrue(np.all(result.valid))
        self.assertTrue(np.all(result.lower_bound <= exact + 1e-10))

    def test_direction_error_makes_bound_conservative(self) -> None:
        current = np.asarray([4.0])
        length = np.asarray([2.0])
        dot = np.asarray([0.5])
        exact_direction = evaluate_v0_bound(
            current, length, dot, np.asarray([0.0]), dimension=8
        )
        noisy = evaluate_v0_bound(
            current, length, dot, np.asarray([0.4]), dimension=8
        )
        self.assertLess(noisy.lower_bound[0], exact_direction.lower_bound[0])

    def test_wilson_interval(self) -> None:
        low, high = wilson_interval(20, 211_836)
        self.assertLess(low, 20 / 211_836)
        self.assertGreater(high, 20 / 211_836)


if __name__ == "__main__":
    unittest.main()
