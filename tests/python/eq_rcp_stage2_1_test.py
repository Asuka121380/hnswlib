#!/usr/bin/env python3
"""Stage 2.1 unit and end-to-end tests on a synthetic operational dataset."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "eq_rcp"
sys.path.insert(0, str(SCRIPT_DIR))

from build_operational_dataset import build_dataset  # noqa: E402
from run_stage2_1_selection import run_experiment  # noqa: E402
from stage2_quantization import (  # noqa: E402
    MetricTransform,
    ProductQuantizer,
    paired_bootstrap_ci,
)
from stage2_1_quantization import (  # noqa: E402
    ExactLengthQuantizer,
    UnitDirectionQuantizer,
    radial_tangential_diagnostics,
)
from reassess_stage2_1_closeout import generate as generate_closeout  # noqa: E402
from validate_stage2_1_selection import validate  # noqa: E402


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
                {"name": "development_quantizer_train", "fraction": 0.25},
                {"name": "development_decoder_train", "fraction": 0.20},
                {"name": "development_model_selection", "fraction": 0.20},
                {"name": "development_risk_calibration", "fraction": 0.20},
                {"name": "development_final_test", "fraction": 0.15},
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

    base = ProductQuantizer.fit(vectors, weights, 4, 4, 3, 5, 16)
    from stage2_quantization import FlatPQQuantizer
    exact = ExactLengthQuantizer("norm_corrected_pq", FlatPQQuantizer("base", base))
    reconstructed, exact_codes = exact.reconstruct(vectors)
    assert exact_codes.shape == (64, 4)
    assert np.max(np.abs(np.linalg.norm(reconstructed, axis=1) - np.linalg.norm(vectors, axis=1))) < 1e-5
    zero_reconstructed, _ = exact.reconstruct(np.zeros((2, 8), dtype=np.float32))
    assert np.array_equal(zero_reconstructed, np.zeros_like(zero_reconstructed))

    direction = UnitDirectionQuantizer.fit(
        "gain_shape_opq", vectors, weights,
        {"subquantizers": 4, "centroids": 4, "max_iter": 4, "batch_size": 16, "opq_iterations": 1},
        8, "opq",
    )
    direction_reconstructed, _ = direction.reconstruct(vectors)
    assert np.max(np.abs(np.linalg.norm(direction_reconstructed, axis=1) - np.linalg.norm(vectors, axis=1))) < 1e-5
    aggregate, strata = radial_tangential_diagnostics(
        vectors, direction_reconstructed, np.arange(64), vectors,
        np.linspace(-1, 1, 64), 4, 4,
    )
    assert aggregate["count"] == 64
    assert len(strata) == 8

    with tempfile.TemporaryDirectory(prefix="eq-rcp-stage2-") as temp:
        root = Path(temp)
        stage1_dir = make_stage1(root)
        stage1_manifest = json.loads(
            (stage1_dir / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        config = {
            "format": "eq_rcp_stage2_1_representation_selection",
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
                "model_selection": {
                    "split": "development_model_selection",
                    "max_queries": None,
                },
                "risk_calibration": {
                    "split": "development_risk_calibration",
                    "max_queries": None,
                },
                "final_test": {
                    "split": "development_final_test",
                    "max_queries": None,
                },
            },
            "methods": [
                "direct_delta_pq",
                "direct_delta_opq",
                "norm_corrected_pq",
                "norm_corrected_opq",
                "direction_pq_exact_length",
                "gain_shape_opq",
                "grouped_oae_diagonal",
                "grouped_oae_block",
                "opq_operational_refined",
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
                "length_groups": 2,
                "refinement_iterations": 1,
                "refinement_ridge": 0.01,
                "rollback_tolerance": 0.0,
            },
            "objective_weight": {
                "mode": "raw",
                "minimum": 0.05,
                "tau": 0.1,
            },
            "evaluation": {
                "near_margin_quantile": 0.2,
                "diagnostic_false_prune_alpha": 0.1,
                "risk_confidence": 0.95,
            },
            "analysis": {"length_bins": 3, "margin_bins": 3, "attribution_tracking_tolerance": 0.02},
            "flexible_code_oracle": {
                "learning_rate": 0.1,
                "max_iter": 20,
                "max_leaf_nodes": 7,
                "min_samples_leaf": 2,
                "l2_regularization": 0.1,
            },
            "bootstrap": {"seed": 4, "samples": 200, "confidence": 0.95},
            "gate": {
                "target_q99_reduction": 0.10,
                "minimum_paired_ci_lower": -10.0,
                "maximum_near_q99_ratio": 100.0,
                "maximum_q999_ratio": 100.0,
                "minimum_diagnostic_coverage": 0.0,
                "maximum_false_prune_rate_over_good": 1.0,
                "minimum_code_oracle_coverage": 0.0,
                "maximum_model_parameter_bytes": 10000000,
                "allow_opq_fallback": True,
            },
        }
        config_path = root / "stage2-config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        output = root / "stage2"
        manifest = run_experiment(config_path, output)
        assert manifest["gate_decision"] == "PENDING_FORMAL_STAGE2_1"
        assert manifest["selected_flat_representation"] in config["methods"]
        selection_report = json.loads(
            (output / "selection_report.json").read_text(encoding="utf-8")
        )
        assert selection_report["selected_flat_representation"] == manifest["selected_flat_representation"]
        assert all(
            report["calibrated_risk_deferred"] is True
            for report in selection_report["candidate_reports"].values()
        )
        report = validate(config_path, output)
        assert report["status"] == "PASS"
        assert report["selection_method_seed_pairs"] == len(config["methods"])
        assert report["final_method_seed_pairs"] in {1, 2}
        assert manifest["protocol"]["evaluator_protocol_version"] == 2
        assert manifest["protocol"]["final_test_consumed"] is True
        assert set(manifest["selection"]) == {
            "quantizer_train", "decoder_train", "model_selection",
            "risk_calibration", "final_test",
        }
        original_manifest_bytes = (output / "stage2_1_manifest.json").read_bytes()
        original_gate_bytes = (output / "gate_report.json").read_bytes()
        closeout = root / "stage2-closeout"
        reassessment = generate_closeout(output, closeout)
        assert (output / "stage2_1_manifest.json").read_bytes() == original_manifest_bytes
        assert (output / "gate_report.json").read_bytes() == original_gate_bytes
        assert reassessment["original_mixed_gate_was_not_modified"] is True
        split_report = validate(config_path, output, closeout)
        assert split_report["closeout_status"] == "PASS"
        assert split_report["risk_status"] == "PENDING_RISK_ESTIMATOR"
        representation = json.loads(
            (closeout / "representation_gate_report.json").read_text(encoding="utf-8")
        )
        estimator = json.loads(
            (closeout / "estimator_risk_diagnostic_report.json").read_text(encoding="utf-8")
        )
        assert representation["status"] != estimator["status"]
        assert representation["frozen_selected_artifacts"] == estimator["frozen_selected_artifacts"]
        assert representation["risk_status"] == "PENDING_RISK_ESTIMATOR"

        estimator_path = closeout / "estimator_risk_diagnostic_report.json"
        original_estimator = estimator_path.read_text(encoding="utf-8")
        mutated_estimator = json.loads(original_estimator)
        mutated_estimator["status"] = "GO_REAL_PRUNING_RISK"
        estimator_path.write_text(json.dumps(mutated_estimator), encoding="utf-8")
        try:
            validate(config_path, output, closeout)
            raise AssertionError("risk report must not authorize pruning or overwrite representation")
        except ValueError as error:
            assert "estimator risk report" in str(error)
        estimator_path.write_text(original_estimator, encoding="utf-8")
        repeat_output = root / "stage2-repeat"
        repeat_manifest = run_experiment(config_path, repeat_output)
        validate(config_path, repeat_output)
        assert (
            manifest["stage2_1_results_semantic_sha256"]
            == repeat_manifest["stage2_1_results_semantic_sha256"]
        )

        missing = json.loads(json.dumps(config))
        del missing["selection"]["final_test"]
        missing_path = root / "missing-final.json"
        missing_path.write_text(json.dumps(missing), encoding="utf-8")
        try:
            run_experiment(missing_path, root / "missing-final-output")
            raise AssertionError("missing final_test role must fail closed")
        except ValueError as error:
            assert "exactly five" in str(error)

        overlap = json.loads(json.dumps(config))
        overlap["selection"]["final_test"]["split"] = "development_model_selection"
        overlap_path = root / "overlap.json"
        overlap_path.write_text(json.dumps(overlap), encoding="utf-8")
        try:
            run_experiment(overlap_path, root / "overlap-output")
            raise AssertionError("split role swap must fail closed")
        except ValueError as error:
            assert "split/role mismatch" in str(error)

        manifest_path = output / "stage2_1_manifest.json"
        original_manifest = manifest_path.read_text(encoding="utf-8")
        mutated = json.loads(original_manifest)
        mutated["protocol"]["risk_cutoffs_frozen_before_final_test"] = False
        manifest_path.write_text(json.dumps(mutated), encoding="utf-8")
        try:
            validate(config_path, output)
            raise AssertionError("post-final protocol mutation must fail closed")
        except ValueError as error:
            assert "freeze-order" in str(error)
        manifest_path.write_text(original_manifest, encoding="utf-8")

        legacy = json.loads(original_manifest)
        del legacy["protocol"]
        manifest_path.write_text(json.dumps(legacy), encoding="utf-8")
        try:
            validate(config_path, output)
            raise AssertionError("legacy three-way manifest must be rejected")
        except ValueError as error:
            assert "legacy three-way" in str(error)
        manifest_path.write_text(original_manifest, encoding="utf-8")

        query_path = stage1_dir / "queries.parquet"
        original_query_bytes = query_path.read_bytes()
        query_table = pq.read_table(query_path)
        selection_rows = query_table.filter(
            pc.equal(query_table["split"], "development_model_selection")
        ).slice(0, 1)
        leaked_row = selection_rows.to_pylist()[0]
        leaked_row["split"] = "development_risk_calibration"
        leaked = pa.Table.from_pylist([leaked_row], schema=query_table.schema)
        pq.write_table(pa.concat_tables([query_table, leaked]), query_path)
        try:
            run_experiment(config_path, root / "query-leakage-output")
            raise AssertionError("query-level overlap must fail closed")
        except ValueError as error:
            assert "query selections overlap" in str(error)
        finally:
            query_path.write_bytes(original_query_bytes)


if __name__ == "__main__":
    run_tests()
    print("EQ-RCP Stage 2.1 tests: PASS")
