#!/usr/bin/env python3
"""Stage 2 unit and end-to-end tests on a synthetic operational dataset."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "eq_rcp"
sys.path.insert(0, str(SCRIPT_DIR))

from build_operational_dataset import build_dataset  # noqa: E402
from run_stage2_ceiling import run_experiment  # noqa: E402
from stage2_quantization import (  # noqa: E402
    MetricTransform,
    ProductQuantizer,
    paired_bootstrap_ci,
)
from validate_stage2_ceiling import validate  # noqa: E402


def write_fvecs(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values, dtype="<f4")
    with path.open("wb") as handle:
        for row in values:
            handle.write(struct.pack("<i", values.shape[1]))
            handle.write(row.tobytes())


def make_stage1(root: Path) -> Path:
    rng = np.random.default_rng(123)
    base_values = rng.normal(size=(12, 4)).astype(np.float32)
    query_values = rng.normal(size=(90, 4)).astype(np.float32)
    base_path = root / "base.fvecs"
    query_path = root / "query.fvecs"
    mapping_path = root / "mapping.npy"
    mapping_manifest_path = root / "mapping.json"
    records_path = root / "records.parquet"
    write_fvecs(base_path, base_values)
    write_fvecs(query_path, query_values)
    mapping = np.arange(len(base_values), dtype=np.int64)
    np.save(mapping_path, mapping, allow_pickle=False)
    mapping_manifest_path.write_text(
        json.dumps({"count": len(mapping), "index_sha256": "0" * 64}), encoding="utf-8"
    )
    query_id = np.arange(len(query_values), dtype=np.int64)
    source = query_id % len(base_values)
    target = (source + 1 + (query_id % 3)) % len(base_values)
    q = query_values.astype(np.float64)
    c = base_values[source].astype(np.float64)
    v = base_values[target].astype(np.float64)
    current = np.sum((q - c) ** 2, axis=1)
    candidate = np.sum((q - v) ** 2, axis=1)
    threshold = np.quantile(candidate, 0.35) + 0.15 * np.sin(query_id)
    pq.write_table(
        pa.table(
            {
                "query_id": query_id,
                "current_node_id": source,
                "candidate_id": target,
                "graph_layer": np.zeros(len(query_id), dtype=np.int32),
                "ef_search": np.full(len(query_id), 20, dtype=np.int32),
                "current_squared_distance": current,
                "threshold": threshold,
                "shadow_exact_squared_distance": candidate,
            }
        ),
        records_path,
    )
    logical = {
        "records": "test/records",
        "base_vectors": "test/base",
        "query_vectors": "test/query",
        "internal_to_label": "test/mapping",
        "mapping_manifest": "test/mapping-manifest",
        "index": "test/index",
    }
    stage1_config = {
        "format": "eq_rcp_operational_dataset",
        "version": 1,
        "dataset_id": "stage2-test-stage1",
        "dataset_role": "development",
        "logical_artifacts": logical,
        "paths": {
            "records": str(records_path),
            "base_vectors": str(base_path),
            "query_vectors": str(query_path),
            "internal_to_label": str(mapping_path),
            "mapping_manifest": str(mapping_manifest_path),
            "index": None,
        },
        "expected_sha256": {"index": "0" * 64},
        "query_mapping": {"mode": "query_id_is_row", "offset": 0},
        "split": {
            "seed": 99,
            "fractions": [
                {"name": "development_quantizer_train", "fraction": 0.4},
                {"name": "development_decoder_train", "fraction": 0.3},
                {"name": "development_model_selection", "fraction": 0.3},
            ],
        },
        "build": {"batch_size": 16},
        "validation": {
            "expected_rows": len(query_id),
            "distance_abs_tolerance": 1e-10,
            "require_index_adjacency_validation": False,
        },
    }
    stage1_config_path = root / "stage1-config.json"
    stage1_config_path.write_text(json.dumps(stage1_config), encoding="utf-8")
    stage1_dir = root / "stage1"
    build_dataset(stage1_config_path, stage1_dir)
    return stage1_dir


def run_tests() -> None:
    rng = np.random.default_rng(5)
    vectors = rng.normal(size=(64, 8)).astype(np.float32)
    weights = np.ones(64, dtype=np.float64)
    pq_model = ProductQuantizer.fit(vectors, weights, 4, 4, 3, 5, 16)
    codes = pq_model.encode(vectors)
    assert codes.shape == (64, 4)
    assert pq_model.decode(codes).shape == vectors.shape
    assert pq_model.code_bytes == 4

    x = rng.normal(size=(80, 8)).astype(np.float32)
    for kind in ["diagonal", "block", "full"]:
        metric = MetricTransform.fit(x, np.ones(len(x)), kind, 2, 0.1, 1e-3)
        transformed = metric.transform(vectors)
        restored = metric.inverse(transformed)
        assert np.max(np.abs(restored - vectors)) < 2e-5
        query_transformed = metric.transform_query(x[:10])
        left = np.einsum("ij,ij->i", x[:10], vectors[:10])
        right = np.einsum("ij,ij->i", query_transformed, transformed[:10])
        assert np.max(np.abs(left - right)) < 2e-4

    ci = paired_bootstrap_ci(np.asarray([3, 4, 5]), np.asarray([1, 2, 3]), 7, 200, 0.95)
    assert ci["ci_lower"] > 0

    with tempfile.TemporaryDirectory(prefix="eq-rcp-stage2-") as temp:
        root = Path(temp)
        stage1_dir = make_stage1(root)
        stage1_manifest = json.loads(
            (stage1_dir / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        config = {
            "format": "eq_rcp_flat_oae_ceiling",
            "version": 1,
            "run_role": "development_pilot",
            "allow_formal_gate_decision": False,
            "stage1_dataset_dir": str(stage1_dir),
            "expected_stage1_dataset_semantic_sha256": stage1_manifest[
                "dataset_semantic_sha256"
            ],
            "selection": {
                "seed": 9,
                "quantizer_train": {
                    "split": "development_quantizer_train",
                    "max_queries": None,
                },
                "decoder_train": {
                    "split": "development_decoder_train",
                    "max_queries": None,
                },
                "evaluation": {
                    "split": "development_model_selection",
                    "max_queries": None,
                },
            },
            "methods": [
                "direct_delta_pq",
                "direct_delta_opq",
                "oae_diagonal",
                "oae_block",
                "oae_full",
                "direction_pq_reference",
            ],
            "seeds": [3],
            "quantizer": {
                "code_bytes": 2,
                "subquantizers": 2,
                "centroids": 2,
                "max_iter": 4,
                "batch_size": 16,
                "opq_iterations": 1,
                "metric_block_size": 2,
                "metric_shrinkage": 0.1,
                "metric_ridge": 0.001,
            },
            "objective_weight": {
                "mode": "raw",
                "minimum": 0.05,
                "tau": 0.1,
            },
            "evaluation": {
                "near_margin_quantile": 0.2,
                "diagnostic_false_prune_alpha": 0.1,
            },
            "flexible_code_oracle": {
                "learning_rate": 0.1,
                "max_iter": 20,
                "max_leaf_nodes": 7,
                "min_samples_leaf": 2,
                "l2_regularization": 0.1,
            },
            "bootstrap": {"seed": 4, "samples": 200, "confidence": 0.95},
            "gate": {
                "minimum_q99_reduction": -10.0,
                "minimum_paired_ci_lower": -10.0,
                "minimum_code_oracle_coverage_gain": -10.0,
                "minimum_ideal_coverage": 0.0,
            },
        }
        config_path = root / "stage2-config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        output = root / "stage2"
        manifest = run_experiment(config_path, output)
        assert manifest["gate_decision"] == "PENDING_FORMAL_STAGE2"
        report = validate(config_path, output)
        assert report["status"] == "PASS"
        assert report["method_seed_pairs"] == 6


if __name__ == "__main__":
    run_tests()
    print("EQ-RCP Stage 2 tests: PASS")
