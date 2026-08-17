#!/usr/bin/env python3

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from jq_quantizer import JQConfig, JQQuantizer, analytical_gaussian_init


class JQQuantizerTest(unittest.TestCase):
    def test_config_rejects_invalid_layout(self) -> None:
        with self.assertRaises(ValueError):
            JQConfig(dimension=10, subspaces=3).validate()
        with self.assertRaises(ValueError):
            JQConfig(dimension=8, subspaces=2, bits=4).validate()

    def test_rotation_is_deterministic_and_near_orthogonal(self) -> None:
        config = JQConfig(dimension=8, subspaces=2, rotation_seed=17)
        first = JQQuantizer(config)
        second = JQQuantizer(config)
        np.testing.assert_array_equal(first.rotation, second.rotation)
        self.assertLess(first.orthogonality_defect_fro, 1e-5)

    def test_analytical_initializer_is_deterministic(self) -> None:
        rng = np.random.default_rng(3)
        data = rng.normal(size=(128, 4)).astype(np.float32)
        first = analytical_gaussian_init(data, 256)
        second = analytical_gaussian_init(data, 256)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (256, 4))
        self.assertTrue(np.isfinite(first).all())

    def test_encode_is_nearest_centroid_and_decode_matches(self) -> None:
        rng = np.random.default_rng(5)
        directions = rng.normal(size=(64, 8)).astype(np.float32)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        quantizer = JQQuantizer(JQConfig(dimension=8, subspaces=2))
        rotated = quantizer.rotate(directions)
        quantizer.fit_rotated(rotated)
        codes = quantizer.encode_rotated(rotated, batch_size=7)
        decoded = quantizer.decode_rotated(codes)
        self.assertEqual(codes.shape, (64, 2))
        self.assertEqual(codes.dtype, np.uint8)
        self.assertEqual(decoded.shape, directions.shape)
        for row in (0, 17, 63):
            for subspace in range(2):
                block = rotated[row, subspace * 4 : (subspace + 1) * 4]
                table = quantizer.centroids[subspace].astype(np.float64)
                distances = np.sum((table - block) ** 2, axis=1)
                self.assertEqual(int(codes[row, subspace]), int(np.argmin(distances)))

    def test_error_upper_bounds_original_space_reconstruction_error(self) -> None:
        rng = np.random.default_rng(9)
        directions = rng.normal(size=(48, 8)).astype(np.float32)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        quantizer = JQQuantizer(JQConfig(dimension=8, subspaces=2))
        rotated = quantizer.rotate(directions)
        quantizer.fit_rotated(rotated)
        codes = quantizer.encode_rotated(rotated)
        upper = quantizer.direction_error_upper(rotated, codes)
        reconstructed_rotated = quantizer.decode_rotated(codes).astype(np.float64)
        reconstructed_original = reconstructed_rotated @ quantizer.rotation.astype(np.float64)
        exact = np.linalg.norm(directions.astype(np.float64) - reconstructed_original, axis=1)
        self.assertTrue(np.all(upper >= exact))

    def test_artifact_save_refuses_overwrite(self) -> None:
        rng = np.random.default_rng(11)
        directions = rng.normal(size=(32, 8)).astype(np.float32)
        quantizer = JQQuantizer(JQConfig(dimension=8, subspaces=2))
        quantizer.fit_rotated(quantizer.rotate(directions))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact"
            manifest = quantizer.save_artifacts(output)
            self.assertEqual(manifest["official_jhq_commit"], "1636e197a36871db4a79d66d22f9f9dd0fa51e31")
            with self.assertRaises(FileExistsError):
                quantizer.save_artifacts(output)


if __name__ == "__main__":
    unittest.main()
