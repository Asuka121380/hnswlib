#!/usr/bin/env python3
"""Focused tests for strict real-dataset vector and ground-truth validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "baseline_trace" / "validate_dataset.py"
SPEC = importlib.util.spec_from_file_location("validate_dataset", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {SCRIPT}")
validate_dataset = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validate_dataset)


def write_fvecs(path: Path, rows: list[list[float]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(struct.pack("<i", len(row)))
            handle.write(struct.pack(f"<{len(row)}f", *row))


def write_ivecs(path: Path, rows: list[list[int]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(struct.pack("<i", len(row)))
            handle.write(struct.pack(f"<{len(row)}i", *row))


class DatasetValidationTest(unittest.TestCase):
    def test_valid_files_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vectors = root / "base.fvecs"
            truth = root / "truth.ivecs"
            write_fvecs(vectors, [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
            write_ivecs(truth, [[0, 1], [2, 0]])
            validate_dataset.inspect(vectors, "fvecs", 2, 3)
            validate_dataset.inspect(truth, "ivecs", 2, 2)
            validate_dataset.validate_ground_truth_labels(truth, 2, 2, 3)

    def test_inconsistent_record_prefix_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.fvecs"
            write_fvecs(path, [[0.0, 1.0], [2.0, 3.0]])
            with path.open("r+b") as handle:
                handle.seek(12)
                handle.write(struct.pack("<i", 3))
            with self.assertRaisesRegex(RuntimeError, "Dimension mismatch"):
                validate_dataset.inspect(path, "fvecs", 2, 2)

    def test_out_of_range_ground_truth_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            truth = Path(directory) / "truth.ivecs"
            write_ivecs(truth, [[0, 3]])
            with self.assertRaisesRegex(RuntimeError, "out of range"):
                validate_dataset.validate_ground_truth_labels(truth, 2, 1, 3)


if __name__ == "__main__":
    unittest.main()
