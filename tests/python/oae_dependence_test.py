#!/usr/bin/env python3
"""Small deterministic tests for the OAE necessary-condition experiment."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import json
import hashlib
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "oae_dependence"
sys.path.insert(0, str(SCRIPT_DIR))

from oae_dependence_core import Fvecs, SearchEvents, evaluate_primary, fit_model, fixed_probe, reconstruct_pq

V0_DIR = REPO_ROOT / "scripts" / "v0"
sys.path.insert(0, str(V0_DIR))
from v0pq_format import write_v0pq


def write_fvecs(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values, dtype="<f4")
    raw = np.empty((len(values), values.shape[1] + 1), dtype="<i4")
    raw[:, 0] = values.shape[1]
    raw[:, 1:] = values.view("<i4")
    raw.tofile(path)


class OaeDependenceTest(unittest.TestCase):
    def test_probe_is_deterministic_and_orthonormal(self) -> None:
        first, second = fixed_probe(12, 5, 17), fixed_probe(12, 5, 17)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.allclose(first.T @ first, np.eye(5), atol=1e-6))

    def test_positive_edge_query_dependence_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_values = np.asarray([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
            rng = np.random.default_rng(99)
            query_values = []
            group = []
            for index in range(120):
                selected = index % 2
                group.append(selected)
                mean = np.asarray([3, 0, 0, 0] if selected == 0 else [0, 3, 0, 0])
                query_values.append(mean + rng.normal(scale=0.25, size=4))
            write_fvecs(root / "base.fvecs", base_values)
            write_fvecs(root / "learn.fvecs", np.asarray(query_values))
            base, learn = Fvecs(root / "base.fvecs", 4), Fvecs(root / "learn.fvecs", 4)
            train_ids, validation_ids = np.arange(80), np.arange(80, 120)
            train = SearchEvents(train_ids, np.zeros(80, dtype=np.int64), np.asarray(group[:80]) + 1)
            validation = SearchEvents(validation_ids, np.zeros(40, dtype=np.int64), np.asarray(group[80:]) + 1)
            model = fit_model(train, base, learn, probe_count=4, direction_clusters=2, length_bins=1,
                              shrinkage=1.0, probe_seed=7, group_seed=7, kmeans_iterations=3,
                              group_fit_edge_cap=0, batch_size=32)
            _, report = evaluate_primary(validation, base, learn, model, batch_size=32,
                                         bootstrap_repetitions=400, bootstrap_seed=7,
                                         relative_gate=0.02, minimum_noninferior_probes=2,
                                         maximum_query_contribution=0.10)
            self.assertEqual(report["status"], "PASS_EDGE_QUERY_DEPENDENCE")
            base.close()
            learn.close()

    def test_pq_reconstruction_uses_independent_subspaces(self) -> None:
        centroids = np.asarray([[[0, 0], [1, 0]], [[0, 0], [0, 1]]], dtype=np.float32)
        values = np.asarray([[0.9, 0.1, 0.1, 0.8]], dtype=np.float32)
        self.assertTrue(np.allclose(reconstruct_pq(values, centroids), [[1, 0, 0, 1]]))

    def test_synthetic_controls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "synthetic.json"
            result = subprocess.run([sys.executable, str(SCRIPT_DIR / "run_synthetic_tests.py"),
                                     "--output", str(output)], cwd=REPO_ROOT)
            self.assertEqual(result.returncode, 0)

    def test_end_to_end_artifact_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_values = np.asarray([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
            rng = np.random.default_rng(123)
            queries = np.asarray([
                (np.asarray([3, 0, 0, 0]) if index % 2 == 0 else np.asarray([0, 3, 0, 0]))
                + rng.normal(scale=0.2, size=4) for index in range(80)
            ])
            base_path, learn_path = root / "base.fvecs", root / "gist_learn.fvecs"
            write_fvecs(base_path, base_values)
            write_fvecs(learn_path, queries)
            index_path, mapping_path = root / "index.bin", root / "mapping.npy"
            index_path.write_bytes(b"frozen-index")
            np.save(mapping_path, np.arange(3), allow_pickle=False)
            codebook_path = root / "tiny.v0pq"
            zero_hash = hashlib.sha256(b"").hexdigest()
            write_v0pq(codebook_path, dimension=4, M=2, nbits=1,
                       centroids=(0, 0, 1, 0, 0, 0, 0, 1),
                       training_metadata={"test": True},
                       source_manifest_sha256=zero_hash, source_directions_sha256=zero_hash)
            config = {
                "schema": "test", "dataset": "gist1m", "dimension": 4, "n_base": 3,
                "paths": {"learn": str(learn_path), "base": str(base_path), "index": str(index_path),
                          "mapping": str(mapping_path), "v0_codebook": str(codebook_path)},
                "query_splits": {"training": {"offset": 0, "count": 40},
                                 "validation": {"offset": 40, "count": 40}},
                "excluded_query_ranges": [{"name": "unused", "offset": 100, "count": 1}],
                "trace": {"k": 1, "ef_search": 10, "M": 2, "ef_construction": 10,
                          "seed": 1, "threads": 1, "dco_sample_modulus": 1,
                          "dco_sample_remainder": 0, "collect_geometry": True,
                          "collect_distance_timing": False},
                "model": {"probe_count": 4, "direction_clusters": 2, "length_bins": 1,
                          "shrinkage_pseudocount": 1.0, "probe_seed": 7, "group_seed": 7,
                          "bootstrap_seed": 7, "bootstrap_repetitions": 100,
                          "kmeans_iterations": 2, "group_fit_edge_cap": 0,
                          "batch_size": 32, "secondary_batch_size": 32},
                "gates": {"primary_relative_improvement_minimum": 0.02,
                          "minimum_noninferior_probes": 2,
                          "maximum_single_query_contribution": 0.10},
                "resource_caps": {"learn_queries_total": 80},
            }
            config_path, run = root / "config.json", root / "run"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            def call(script: str, *args: object) -> None:
                result = subprocess.run([sys.executable, str(SCRIPT_DIR / script), *(str(x) for x in args)],
                                        cwd=REPO_ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            call("freeze_inputs.py", "--config", config_path, "--output-dir", run)
            for role, offset in (("training", 0), ("validation", 40)):
                trace = root / f"{role}.csv"
                with trace.open("w", encoding="utf-8") as handle:
                    handle.write("query_id,current_node_label,neighbor_label,geometry_valid,edge_length_cd\n")
                    for local in range(40):
                        handle.write(f"{local},0,{(offset + local) % 2 + 1},1,1\n")
                    handle.write("0,1,2,1,0\n")
                call("build_search_event_dataset.py", "--trace-csv", trace,
                     "--query-manifest", run / f"{role}_query_manifest.json",
                     "--output", run / f"{role}_events.npz")
                event_manifest = json.loads((run / f"{role}_events.npz.manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(event_manifest["zero_length_events_skipped"], 1)
            call("run_synthetic_tests.py", "--output", run / "synthetic_test_report.json")
            call("evaluate_dependence.py", "--input-dir", run,
                 "--training-events", run / "training_events.npz",
                 "--validation-events", run / "validation_events.npz", "--output-dir", run)
            call("validate_run.py", "--run-dir", run)
            report = json.loads((run / "validation_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
