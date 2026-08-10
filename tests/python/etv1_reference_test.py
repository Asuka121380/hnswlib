#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "edge_transform_v1"))

from etv1_reference import (  # noqa: E402
    block_certified_lower_bound,
    edge_geometry,
    equal_blocks,
    interval_length_term,
    orthogonality_defect,
    progressive_lower_bounds,
)


class ETV1ReferenceTest(unittest.TestCase):
    def test_zero_length_edge_uses_exact_anchor_distance(self) -> None:
        q = np.array([2.0, -1.0])
        c = np.array([0.5, 0.25])
        result = block_certified_lower_bound(
            q, c, c, np.zeros(2), np.eye(2), np.zeros(2), equal_blocks(2, 1)
        )
        self.assertEqual(result.status, "zero_length_exact")
        self.assertEqual(result.lower_bound, result.exact_distance)
        self.assertIsNone(edge_geometry(c, c).direction)

    def test_interval_formula_handles_negative_support(self) -> None:
        safe = interval_length_term(-1.0, 1.0, 2.0)
        naive = 1.0 - 2.0 * 2.0 * -1.0
        self.assertEqual(safe, 3.0)
        self.assertEqual(naive, 5.0)

    def test_nonorthogonal_transform_is_closed_by_defect(self) -> None:
        q = np.array([1.0, -0.5, 0.25])
        c = np.array([0.1, 0.2, -0.3])
        v = np.array([0.4, -0.1, 0.5])
        p = np.array([-0.2, 0.0, 0.1])
        transform = np.diag([0.9, 1.1, 1.05])
        u = edge_geometry(c, v).direction
        assert u is not None
        y = transform @ u
        result = block_certified_lower_bound(
            q,
            c,
            v,
            p,
            transform,
            y,
            equal_blocks(3, 3),
            transform_defect_upper=orthogonality_defect(transform),
        )
        self.assertLessEqual(result.lower_bound, result.exact_distance + 1e-12)
        self.assertGreater(result.transform_defect_padding, 0.0)

    def test_random_bounds_and_progressive_monotonicity(self) -> None:
        rng = np.random.default_rng(20260810)
        for _ in range(2000):
            d = int(rng.integers(2, 33))
            b = int(rng.integers(1, min(8, d) + 1))
            q, c, p = rng.normal(size=(3, d))
            u = rng.normal(size=d)
            u /= np.linalg.norm(u)
            ell = float(10.0 ** rng.uniform(-3.0, 2.0))
            v = c + ell * u
            y_hat = u - rng.normal(size=d) * 0.1
            blocks = equal_blocks(d, b)
            result = block_certified_lower_bound(q, c, v, p, np.eye(d), y_hat, blocks)
            self.assertLessEqual(result.lower_bound, result.exact_distance + 1e-10)
            progressive = progressive_lower_bounds(
                float(np.dot(q - c, q - c)),
                ell,
                float(np.dot(p - c, u)),
                result.block_supports,
                tuple(range(b)),
            )
            self.assertTrue(all(left <= right + 1e-12 for left, right in zip(progressive, progressive[1:])))
            self.assertTrue(all(value <= result.exact_distance + 1e-10 for value in progressive))


if __name__ == "__main__":
    unittest.main()

