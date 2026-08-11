#!/usr/bin/env python3
"""Fail-closed tests for the Stage 3 input freezer."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "eq_rcp"
sys.path.insert(0, str(SCRIPT_DIR))

from freeze_stage3_inputs import build_manifest  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def expect_failure(fn, phrase: str) -> None:
    try:
        fn()
    except (ValueError, FileNotFoundError) as error:
        assert phrase in str(error), (phrase, str(error))
    else:
        raise AssertionError(f"expected failure containing: {phrase}")


def fixture(root: Path) -> tuple[Path, Path, Path]:
    result = root / "result"
    reassessment = root / "reassessment"
    contract_path = root / "contract.json"
    seeds = [17, 43, 97]
    artifacts = []
    for name, parameterization in [
        ("gain_shape_opq", "unit_direction_exact_gain"),
        ("direct_delta_opq", "opq"),
    ]:
        for seed in seeds:
            relative = Path("models") / f"{name}-seed{seed}.npz"
            artifact_path = result / relative
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(f"{name}:{seed}".encode())
            artifacts.append(
                {
                    "name": name,
                    "seed": seed,
                    "relative_path": str(relative),
                    "artifact_sha256": sha(artifact_path),
                    "model_semantic_sha256": hashlib.sha256(f"model:{name}:{seed}".encode()).hexdigest(),
                    "parameterization_contract": parameterization,
                    "code_bytes": 32,
                    "encoder_inputs": ["Delta", "static_group"],
                    "forbidden_encoder_inputs": ["query", "threshold", "label"],
                }
            )
    roles = {
        name: {
            "split": f"development_{name}",
            "query_count": 3,
            "event_count": 5,
            "query_ids_sha256": hashlib.sha256(name.encode()).hexdigest(),
        }
        for name in [
            "quantizer_train",
            "decoder_train",
            "model_selection",
            "risk_calibration",
            "final_test",
        ]
    }
    source = {
        "run_role": "development_full",
        "fully_sampled": True,
        "selected_flat_representation": "gain_shape_opq",
        "stage2_1_results_semantic_sha256": "1" * 64,
        "stage1_dataset_semantic_sha256": "2" * 64,
        "selection": roles,
        "protocol": {
            "required_roles": list(roles),
            "final_test_consumed": True,
        },
        "artifacts": artifacts,
    }
    write(result / "stage2_1_manifest.json", source)
    write(result / "gate_report.json", {"decision": "NO_GO_REPRESENTATION"})
    source_hashes = {
        "stage2_1_manifest_sha256": sha(result / "stage2_1_manifest.json"),
        "original_mixed_gate_report_sha256": sha(result / "gate_report.json"),
        "original_mixed_gate_decision": "NO_GO_REPRESENTATION",
        "stage2_1_results_semantic_sha256": "1" * 64,
    }
    representation = {
        "status": "GO_STAGE3_REPRESENTATION_DEV",
        "selected_flat_representation": "gain_shape_opq",
        "selected_parameterization_contract": "unit_direction_exact_gain",
        "safety_fallback": "direct_delta_opq",
        "source_run": source_hashes,
    }
    risk = {"status": "PENDING_RISK_ESTIMATOR"}
    write(reassessment / "representation_gate_report.json", representation)
    write(reassessment / "estimator_risk_diagnostic_report.json", risk)
    reassessed = {
        "representation_status": representation["status"],
        "risk_status": risk["status"],
        "representation_gate_report_sha256": sha(reassessment / "representation_gate_report.json"),
        "estimator_risk_diagnostic_report_sha256": sha(
            reassessment / "estimator_risk_diagnostic_report.json"
        ),
    }
    write(reassessment / "stage2_1_read_only_reassessment.json", reassessed)
    contract = {
        "format": "eq_rcp_stage3_experiment_contract",
        "version": 1,
        "scope": "offline_progressive_representation_development",
        "required_representation_status": "GO_STAGE3_REPRESENTATION_DEV",
        "required_risk_status": "PENDING_RISK_ESTIMATOR",
        "selected_flat_representation": "gain_shape_opq",
        "selected_parameterization_contract": "unit_direction_exact_gain",
        "safety_fallback": "direct_delta_opq",
        "flat_code_bytes": 32,
        "candidate_layouts": [{"name": "flat_32", "coarse_bytes": 32, "residual_bytes": 0}],
        "allocation_strategies": ["uniform"],
        "residual_contract": {"family": "gain_shape_tangent_direction_residual"},
        "evaluation_roles": ["bit_allocation_selection", "stopping_calibration", "final_test"],
        "gate": {"max_final_q99_relative_degradation": 0.05},
        "risk_semantics": {"coverage_and_false_prune_are_diagnostic_only": True},
        "forbidden_actions": ["real_hnsw_pruning"],
    }
    write(contract_path, contract)
    return result, reassessment, contract_path


def run_tests() -> None:
    with tempfile.TemporaryDirectory(prefix="eq-rcp-stage3-input-") as temp:
        root = Path(temp)
        result, reassessment, contract = fixture(root)
        pending = build_manifest(result, reassessment, contract)
        assert pending["status"] == "PENDING_FRESH_STAGE3_EVALUATION_QUERIES"
        assert not pending["experiment_ready"]
        assert pending["risk_status"] == "PENDING_RISK_ESTIMATOR"
        assert pending["real_pruning_authorized"] is False
        assert pending["frozen_artifacts"]["matched_seeds"] == [17, 43, 97]

        evaluation_path = root / "evaluation.json"
        evaluation = {
            "format": "eq_rcp_stage3_evaluation_queries",
            "version": 1,
            "dataset_semantic_sha256": "3" * 64,
            "fresh_against_stage2_1": True,
            "roles": {
                role: {
                    "query_count": 10,
                    "event_count": 20,
                    "query_ids_sha256": hashlib.sha256(role.encode()).hexdigest(),
                }
                for role in ["bit_allocation_selection", "stopping_calibration", "final_test"]
            },
        }
        write(evaluation_path, evaluation)
        ready = build_manifest(result, reassessment, contract, evaluation_path)
        assert ready["status"] == "READY_FOR_OFFLINE_STAGE3"
        assert ready["experiment_ready"]

        invalid = copy.deepcopy(evaluation)
        invalid["fresh_against_stage2_1"] = False
        write(evaluation_path, invalid)
        expect_failure(
            lambda: build_manifest(result, reassessment, contract, evaluation_path),
            "not certified fresh",
        )

        source_path = result / "stage2_1_manifest.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source["run_role"] = "development_pilot"
        write(source_path, source)
        expect_failure(
            lambda: build_manifest(result, reassessment, contract),
            "pilot or partially sampled",
        )

    print("eq_rcp_stage3_input_test: PASS")


if __name__ == "__main__":
    run_tests()
