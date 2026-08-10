#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "edge_transform_v1"))

from etv1_quantizer import fit_opq, fit_pq, random_orthogonal  # noqa: E402


class ETV1QuantizerTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.vectors = rng.normal(size=(512, 8)).astype(np.float32)
        self.vectors /= np.linalg.norm(self.vectors, axis=1, keepdims=True)

    def test_identity_pq_shapes_and_reconstruction(self) -> None:
        model = fit_pq(self.vectors, m=2, ksub=8, seed=7, max_iter=10, batch_size=128)
        reconstructed = model.reconstruct_transformed(self.vectors[:17])
        self.assertEqual(reconstructed.shape, (17, 8))
        self.assertTrue(np.all(np.isfinite(reconstructed)))
        self.assertLess(float(np.mean((self.vectors[:17] - reconstructed) ** 2)), 0.2)

    def test_random_transform_is_orthogonal(self) -> None:
        transform = random_orthogonal(8, 9)
        np.testing.assert_allclose(transform.T @ transform, np.eye(8), atol=1e-12)

    def test_weighted_opq_is_finite_and_orthogonal(self) -> None:
        weights = np.linspace(0.1, 2.0, self.vectors.shape[0])
        model = fit_opq(
            self.vectors,
            m=2,
            ksub=8,
            seed=11,
            iterations=2,
            sample_weight=weights,
            max_iter=8,
            batch_size=128,
        )
        self.assertTrue(np.all(np.isfinite(model.codebooks)))
        np.testing.assert_allclose(model.transform.T @ model.transform, np.eye(8), atol=1e-10)


if __name__ == "__main__":
    unittest.main()

