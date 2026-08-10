#!/usr/bin/env python3

from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "edge_transform_v1"))
from extract_hnsw_internal_labels import HEADER, extract  # noqa: E402


class LabelExtractorTest(unittest.TestCase):
    def test_extracts_strided_labels_from_native_level_zero_layout(self) -> None:
        labels = np.asarray([4, 1, 3, 0, 2], dtype=np.uint64)
        stride, label_offset = 48, 32
        header = HEADER.pack(0, 5, 5, stride, label_offset, 8, 2, 0, 16, 32, 16, 1.0, 200)
        level0 = bytearray(labels.size * stride)
        for internal_id, label in enumerate(labels):
            struct.pack_into("<Q", level0, internal_id * stride + label_offset, int(label))
        with tempfile.TemporaryDirectory(prefix="etv1-index-") as temporary:
            index = Path(temporary) / "index.bin"
            index.write_bytes(header + level0)
            actual, parsed = extract(index)
        np.testing.assert_array_equal(actual, labels)
        self.assertEqual(parsed["label_offset"], label_offset)
        self.assertEqual(parsed["cur_element_count"], labels.size)


if __name__ == "__main__":
    unittest.main()
