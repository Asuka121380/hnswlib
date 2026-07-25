#!/usr/bin/env python3
"""Dependency-free local tests for the Milestone 4 PQ pipeline."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "v0"
sys.path.insert(0, str(SCRIPT_DIR))

from v0pq_format import V0PQFormatError, read_v0pq, sha256_file, write_v0pq


class V0PQLocalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _sample(self, row_count: int = 300) -> tuple[Path, Path]:
        directions = self.root / "directions.f32"
        rows = [
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ]
        directions.write_bytes(
            b"".join(struct.pack("<4f", *rows[index % 4]) for index in range(row_count))
        )
        digest = sha256_file(directions)
        manifest = self.root / "manifest.json"
        value = {
            "format": "hnswlib_v0_edge_direction_sample",
            "format_version": 1,
            "sampling_algorithm": "reservoir_algorithm_r_splitmix64_v1",
            "traversal_order": "source_id_then_layer0_slot",
            "sampling_population": "all_nonzero_layer0_directed_edges",
            "metric": "squared_l2",
            "sampled_object": "unit_edge_direction",
            "vector_source": "embedded_hnsw_level0_data",
            "dtype": "float32",
            "byte_order": "little",
            "layout": "row_major",
            "dimension": 4,
            "node_count": 4,
            "requested_sample_count": row_count,
            "produced_sample_count": row_count,
            "directed_edge_count": row_count,
            "valid_edge_count": row_count,
            "zero_length_edge_count": 0,
            "directions_bytes": directions.stat().st_size,
            "directions_sha256": digest,
            "directions_path": str(directions),
            "seed": 11,
        }
        manifest.write_text(json.dumps(value), encoding="utf-8")
        return directions, manifest

    def _codebook(self, directions: Path, manifest: Path) -> Path:
        path = self.root / "tiny.v0pq"
        metadata = {
            "quantizer_name": "faiss_product_quantizer",
            "faiss_version": "synthetic-test",
            "faiss_backend": "cpu",
            "dimension": 4,
            "M_pq": 2,
            "nbits": 1,
            "training_seed": 17,
            "split_seed": 29,
            "split_algorithm": "numpy_default_rng_permutation_v1",
            "iterations": 1,
            "nredo": 1,
            "min_points_per_centroid": 1,
            "max_points_per_centroid": 256,
            "training_count": 200,
            "validation_count": 100,
            "source_manifest_sha256": sha256_file(manifest),
            "source_directions_sha256": sha256_file(directions),
            "centroid_layout": "M_ksub_dsub_row_major_float32",
            "creation_timestamp_utc": "2026-07-25T00:00:00+00:00",
            "producer_git_commit": "synthetic-test",
        }
        write_v0pq(
            path,
            dimension=4,
            M=2,
            nbits=1,
            centroids=(1, 0, 0, 1, 1, 0, 0, 1),
            training_metadata=metadata,
            source_manifest_sha256=sha256_file(manifest),
            source_directions_sha256=sha256_file(directions),
        )
        return path

    def _run(self, script: str, *arguments: object) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT_DIR / script), *(str(value) for value in arguments)],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )

    def test_v0pq_round_trip_and_corruption_detection(self) -> None:
        directions, manifest = self._sample()
        codebook_path = self._codebook(directions, manifest)
        codebook = read_v0pq(codebook_path)
        self.assertEqual((4, 2, 1, 2, 2), (
            codebook.header.dimension,
            codebook.header.M,
            codebook.header.nbits,
            codebook.header.ksub,
            codebook.header.dsub,
        ))
        self.assertEqual(8, len(codebook.centroids))
        with self.assertRaises(V0PQFormatError):
            self._codebook(directions, manifest)
        damaged = self.root / "damaged.v0pq"
        data = bytearray(codebook_path.read_bytes())
        data[260] ^= 1
        damaged.write_bytes(data)
        with self.assertRaises(V0PQFormatError):
            read_v0pq(damaged)

    def test_validator_and_custom_reconstruction(self) -> None:
        directions, manifest = self._sample()
        codebook = self._codebook(directions, manifest)
        validated = self._run(
            "validate_v0pq.py",
            "--codebook", codebook,
            "--expected-manifest", manifest,
            "--expected-directions", directions,
        )
        self.assertEqual("valid", json.loads(validated.stdout)["status"])
        metrics = self.root / "reconstruction.json"
        self._run(
            "compare_codebook_reconstruction.py",
            "--codebook", codebook,
            "--directions", directions,
            "--manifest", manifest,
            "--sample-rows", 8,
            "--output-metrics", metrics,
        )
        result = json.loads(metrics.read_text(encoding="utf-8"))
        self.assertEqual(8, result["checked_rows"])
        self.assertLessEqual(result["max_custom_lut_identity_difference"], 1e-12)
        self.assertFalse(result["faiss_check_performed"])

    def test_training_input_check_does_not_import_faiss(self) -> None:
        directions, manifest = self._sample()
        metrics = self.root / "input-check.json"
        self._run(
            "train_pq_codebook.py",
            "--directions", directions,
            "--manifest", manifest,
            "--M-pq", 2,
            "--min-points-per-centroid", 1,
            "--output-metrics", metrics,
            "--check-input-only",
        )
        result = json.loads(metrics.read_text(encoding="utf-8"))
        self.assertEqual("input-valid", result["status"])
        self.assertEqual(270, result["training_count"])

    def test_pilot_summary_without_plot_dependencies(self) -> None:
        paths = []
        for width, error in ((1, 0.2), (2, 0.1)):
            path = self.root / f"m{width}.json"
            row = {
                "status": "trained",
                "dimension": 4,
                "M_pq": width,
                "nbits": 8,
                "source_manifest_sha256": "a" * 64,
                "source_directions_sha256": "b" * 64,
                "split_seed": 29,
                "codebook_bytes": 100 * width,
                "per_vector_code_bytes": width,
                "per_query_lut_bytes_float32": width * 1024,
                "mean_error": error,
                "median_error": error,
                "p90_error": error,
                "p95_error": error,
                "p99_error": error,
                "max_error": error,
                "reconstruction_mse": error * error,
                "training_seconds": 1.0,
                "validation_seconds": 0.1,
            }
            path.write_text(json.dumps(row), encoding="utf-8")
            paths.append(path)
        output = self.root / "summary"
        self._run(
            "analyze_pq_pilot.py",
            "--metrics", paths[0],
            "--metrics", paths[1],
            "--expected-M", "1,2",
            "--max-p99-error", 0.15,
            "--output-dir", output,
            "--skip-plots",
        )
        aggregate = json.loads((output / "pq_pilot_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(2, aggregate["recommended_M_pq"])
        self.assertTrue((output / "pq_pilot_summary.csv").is_file())
        self.assertTrue((output / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()
