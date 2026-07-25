#!/usr/bin/env python3
"""Dependency-free tests for the streaming Milestone 5 sidecar validator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "v0"

import sys

sys.path.insert(0, str(SCRIPT_DIR))

from create_v0_sidecar_fixture import create_fixture
from validate_v0_sidecar import ValidationError, validate
from v0pq_format import write_v0pq


class V0SidecarValidatorTest(unittest.TestCase):
    @staticmethod
    def _matching_codebook(path: Path) -> None:
        write_v0pq(
            path,
            dimension=4,
            M=2,
            nbits=1,
            centroids=(
                0.0,
                0.5,
                1.0,
                1.5,
                -1.0,
                -0.5,
                2.0,
                2.5,
            ),
            training_metadata={"quantizer": "synthetic"},
            source_manifest_sha256="11" * 32,
            source_directions_sha256="22" * 32,
        )

    def test_valid_fixture_and_expected_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.v0meta"
            codebook = Path(temporary) / "fixture.v0pq"
            create_fixture(path)
            self._matching_codebook(codebook)
            summary = validate(
                path,
                expected_v0pq=codebook,
                expected_node_count=3,
                expected_edge_count=3,
                expected_zero_length_count=1,
            )
            self.assertEqual("valid", summary["status"])
            self.assertEqual(2, summary["exact_only_count"])
            self.assertEqual(1, summary["zero_length_edge_count"])
            self.assertEqual(str(codebook), summary["expected_v0pq"])

    def test_corrupt_fixture_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.v0meta"
            create_fixture(path)
            damaged = bytearray(path.read_bytes())
            damaged[300] ^= 1
            path.write_bytes(damaged)
            with self.assertRaises(ValidationError):
                validate(path)


if __name__ == "__main__":
    unittest.main()
