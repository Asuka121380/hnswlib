#!/usr/bin/env python3
"""Freeze the Stage 3 input boundary without mutating Stage 2.1 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FORMAT = "eq_rcp_stage3_input_manifest"
VERSION = 1
REQUIRED_STAGE2_ROLES = [
    "quantizer_train",
    "decoder_train",
    "model_selection",
    "risk_calibration",
    "final_test",
]


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def _artifact_bindings(
    result_dir: Path, manifest: dict[str, Any], name: str
) -> list[dict[str, Any]]:
    source = [item for item in manifest.get("artifacts", []) if item.get("name") == name]
    if not source:
        raise ValueError(f"frozen Stage 2.1 artifact is missing: {name}")
    bindings: list[dict[str, Any]] = []
    for item in sorted(source, key=lambda value: int(value["seed"])):
        artifact_path = result_dir / Path(str(item["relative_path"]).replace("\\", "/"))
        if not artifact_path.is_file():
            raise FileNotFoundError(f"artifact file is missing: {artifact_path}")
        _require_hash(artifact_path, item["artifact_sha256"], f"{name} seed {item['seed']}")
        bindings.append(
            {
                "name": name,
                "seed": int(item["seed"]),
                "relative_path": item["relative_path"],
                "artifact_sha256": item["artifact_sha256"],
                "model_semantic_sha256": item["model_semantic_sha256"],
                "parameterization_contract": item["parameterization_contract"],
                "code_bytes": int(item["code_bytes"]),
                "encoder_inputs": item["encoder_inputs"],
                "forbidden_encoder_inputs": item["forbidden_encoder_inputs"],
            }
        )
    seeds = [item["seed"] for item in bindings]
    if len(seeds) < 3 or len(seeds) != len(set(seeds)):
        raise ValueError(f"{name} must bind at least three distinct frozen seeds")
    return bindings


def _evaluation_binding(path: Path | None, contract: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if path is None:
        return {
            "status": "REQUIRED_NOT_BOUND",
            "required_roles": contract["evaluation_roles"],
            "fresh_against_stage2_1": "UNVERIFIED",
        }, False
    evaluation = _read(path)
    if evaluation.get("format") != "eq_rcp_stage3_evaluation_queries" or evaluation.get("version") != 1:
        raise ValueError("invalid Stage 3 evaluation-query manifest format/version")
    if evaluation.get("fresh_against_stage2_1") is not True:
        raise ValueError("Stage 3 evaluation queries are not certified fresh against Stage 2.1")
    roles = evaluation.get("roles") or {}
    required = contract["evaluation_roles"]
    if sorted(roles) != sorted(required):
        raise ValueError("Stage 3 evaluation manifest does not bind the exact required roles")
    query_hashes: set[str] = set()
    for role in required:
        binding = roles[role]
        if int(binding.get("query_count", 0)) <= 0 or int(binding.get("event_count", 0)) <= 0:
            raise ValueError(f"Stage 3 evaluation role is empty: {role}")
        query_hash = binding.get("query_ids_sha256")
        if not isinstance(query_hash, str) or len(query_hash) != 64 or query_hash in query_hashes:
            raise ValueError(f"invalid or duplicate query-ID hash for role: {role}")
        query_hashes.add(query_hash)
    return {
        "status": "BOUND_FRESH",
        "manifest_path": str(path.resolve()),
        "manifest_sha256": _sha256(path),
        "dataset_semantic_sha256": evaluation["dataset_semantic_sha256"],
        "fresh_against_stage2_1": True,
        "roles": roles,
    }, True


def build_manifest(
    result_dir: Path,
    reassessment_dir: Path,
    contract_path: Path,
    evaluation_manifest_path: Path | None = None,
) -> dict[str, Any]:
    source = _read(result_dir / "stage2_1_manifest.json")
    gate = _read(result_dir / "gate_report.json")
    representation = _read(reassessment_dir / "representation_gate_report.json")
    risk = _read(reassessment_dir / "estimator_risk_diagnostic_report.json")
    reassessment = _read(reassessment_dir / "stage2_1_read_only_reassessment.json")
    contract = _read(contract_path)

    if contract.get("format") != "eq_rcp_stage3_experiment_contract" or contract.get("version") != 1:
        raise ValueError("invalid Stage 3 experiment contract format/version")
    if representation.get("status") != contract["required_representation_status"]:
        raise ValueError("Stage 2.1 representation status does not authorize offline Stage 3")
    if risk.get("status") != contract["required_risk_status"]:
        raise ValueError("unexpected Stage 2.1 risk status")
    selected = representation.get("selected_flat_representation")
    fallback = representation.get("safety_fallback")
    if selected != contract["selected_flat_representation"] or fallback != contract["safety_fallback"]:
        raise ValueError("Stage 3 contract attempts to change the frozen representation or fallback")
    if representation.get("selected_parameterization_contract") != contract["selected_parameterization_contract"]:
        raise ValueError("Stage 3 parameterization contract disagrees with Stage 2.1")
    if reassessment.get("representation_status") != representation["status"] or reassessment.get(
        "risk_status"
    ) != risk["status"]:
        raise ValueError("Stage 2.1 reassessment reports disagree")
    if source.get("run_role") != "development_full" or not source.get("fully_sampled"):
        raise ValueError("Stage 3 rejects pilot or partially sampled Stage 2.1 inputs")
    if source.get("selected_flat_representation") != selected:
        raise ValueError("source Stage 2.1 manifest disagrees on selected representation")
    protocol = source.get("protocol") or {}
    if protocol.get("required_roles") != REQUIRED_STAGE2_ROLES:
        raise ValueError("source Stage 2.1 manifest does not bind the five-way protocol")
    if not protocol.get("final_test_consumed"):
        raise ValueError("source Stage 2.1 final-test consumption state is missing")
    if gate.get("decision") != "NO_GO_REPRESENTATION":
        raise ValueError("historical mixed Gate decision changed unexpectedly")

    source_binding = representation["source_run"]
    _require_hash(
        result_dir / "stage2_1_manifest.json",
        source_binding["stage2_1_manifest_sha256"],
        "Stage 2.1 manifest",
    )
    _require_hash(
        result_dir / "gate_report.json",
        source_binding["original_mixed_gate_report_sha256"],
        "historical mixed Gate",
    )
    _require_hash(
        reassessment_dir / "representation_gate_report.json",
        reassessment["representation_gate_report_sha256"],
        "representation reassessment",
    )
    _require_hash(
        reassessment_dir / "estimator_risk_diagnostic_report.json",
        reassessment["estimator_risk_diagnostic_report_sha256"],
        "risk reassessment",
    )

    selected_artifacts = _artifact_bindings(result_dir, source, selected)
    fallback_artifacts = _artifact_bindings(result_dir, source, fallback)
    selected_seeds = [item["seed"] for item in selected_artifacts]
    if selected_seeds != [item["seed"] for item in fallback_artifacts]:
        raise ValueError("selected and fallback artifacts do not bind matched seeds")
    if any(item["code_bytes"] != contract["flat_code_bytes"] for item in selected_artifacts):
        raise ValueError("selected artifacts violate the frozen flat-code budget")

    evaluation, ready = _evaluation_binding(evaluation_manifest_path, contract)
    status = "READY_FOR_OFFLINE_STAGE3" if ready else "PENDING_FRESH_STAGE3_EVALUATION_QUERIES"
    return {
        "format": FORMAT,
        "version": VERSION,
        "status": status,
        "experiment_ready": ready,
        "authorized_scope": contract["scope"] if ready else "input_freeze_only",
        "selected_flat_representation": selected,
        "selected_parameterization_contract": representation["selected_parameterization_contract"],
        "safety_fallback": fallback,
        "risk_status": risk["status"],
        "real_pruning_authorized": False,
        "historical_mixed_gate_preserved": True,
        "historical_mixed_gate_decision": gate["decision"],
        "source_stage2_1": {
            "result_directory_name": result_dir.name,
            "stage2_1_results_semantic_sha256": source["stage2_1_results_semantic_sha256"],
            "stage1_dataset_semantic_sha256": source["stage1_dataset_semantic_sha256"],
            "source_hashes": source_binding,
            "consumed_roles": source["selection"],
        },
        "frozen_artifacts": {
            "selected": selected_artifacts,
            "fallback": fallback_artifacts,
            "matched_seeds": selected_seeds,
        },
        "progressive_contract": {
            "flat_code_bytes": contract["flat_code_bytes"],
            "candidate_layouts": contract["candidate_layouts"],
            "allocation_strategies": contract["allocation_strategies"],
            "residual_contract": contract["residual_contract"],
            "gate": contract["gate"],
        },
        "evaluation_queries": evaluation,
        "risk_semantics": contract["risk_semantics"],
        "forbidden_actions": contract["forbidden_actions"],
        "contract_sha256": _sha256(contract_path),
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-1-result-dir", type=Path, required=True)
    parser.add_argument("--reassessment-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_manifest(
        args.stage2_1_result_dir,
        args.reassessment_dir,
        args.contract,
        args.evaluation_manifest,
    )
    write_manifest(args.output, manifest)
    print(json.dumps({"status": manifest["status"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
