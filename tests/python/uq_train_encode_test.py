from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.edge_estimation.train_encode import (  # noqa: E402
    edge_geometry, load_internal_to_base, pack_code_matrix,
)
from scripts.edge_estimation.trainers.product_model import ProductCodebookModel  # noqa: E402
from scripts.edge_estimation.trainers.additive_model import ProductResidualModel  # noqa: E402


class TrainEncodeTest(unittest.TestCase):
    def test_dimension_960_product_and_residual_encoding_shapes(self) -> None:
        dimension = 960
        values = np.linspace(-1.0, 1.0, dimension, dtype=np.float32)[None, :]
        pq_books = np.zeros((32, 2, 30), dtype=np.float32)
        pq_books[:, 1, :] = 0.25
        pq = ProductCodebookModel(
            "opq", "d960_fixture", "test", pq_books, 1,
            lambda transformed: np.ones((transformed.shape[0], 32), dtype=np.uint16),
            rotation=np.eye(dimension, dtype=np.float32))
        pq_codes, transformed, reconstructed = pq.encode_batch(values)
        self.assertEqual(pq_codes.shape, (1, 32))
        self.assertEqual(transformed.shape, (1, dimension))
        self.assertEqual(reconstructed.shape, (1, dimension))
        self.assertEqual(pack_code_matrix(pq_codes, 1).shape, (1, 4))
        prq_books = np.zeros((32, 2, 60), dtype=np.float32)
        prq_books[::2, 1, :] = 0.1
        prq_books[1::2, 1, :] = 0.2
        prq = ProductResidualModel(
            "prq", "d960_fixture", "test", prq_books, 1, 16, 2,
            lambda transformed: np.ones((transformed.shape[0], 32), dtype=np.uint16))
        prq_codes, _, prq_reconstructed = prq.encode_batch(values)
        self.assertEqual(prq_codes.shape, (1, 32))
        self.assertEqual(prq_reconstructed.shape, (1, dimension))
        self.assertTrue(np.allclose(prq_reconstructed, 0.3, atol=1e-6))

    def test_pack4_uses_canonical_low_nibble_first_layout(self) -> None:
        codes = np.asarray([[1, 2, 3, 4], [15, 0, 8, 7]], dtype=np.uint16)
        packed = pack_code_matrix(codes, 4)
        np.testing.assert_array_equal(
            packed, np.asarray([[0x21, 0x43], [0x0F, 0x78]], dtype=np.uint8))

    def test_non_identity_internal_mapping_controls_edge_geometry(self) -> None:
        with tempfile.TemporaryDirectory(prefix="uq-mapping-") as temporary:
            mapping_path = Path(temporary) / "mapping.npy"
            np.save(mapping_path, np.asarray([2, 0, 1], dtype=np.uint64))
            mapping, _, metadata = load_internal_to_base(
                {"internal_to_label": str(mapping_path)}, 3, 3)
            self.assertEqual(metadata["kind"], "npy_internal_to_external_label")
            base = np.asarray([[10.0], [20.0], [30.0]], dtype=np.float32)
            offsets = np.asarray([0, 1, 2, 2], dtype=np.uint64)
            targets = np.asarray([1, 2], dtype=np.uint32)
            _, rows, lengths, directions = edge_geometry(
                base, mapping, offsets, targets, np.asarray([0, 1], dtype=np.int64))
            np.testing.assert_array_equal(rows, np.asarray([2, 0]))
            np.testing.assert_allclose(lengths, [20.0, 10.0])
            np.testing.assert_allclose(directions[:, 0], [-1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
