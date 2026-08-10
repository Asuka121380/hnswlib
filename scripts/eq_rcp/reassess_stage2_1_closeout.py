#!/usr/bin/env python3
"""Create immutable, split-gate Stage 2.1 closeout reports.

The source run is treated as read-only. Reports are written to a separate,
initially empty directory and bind back to the original manifest, mixed Gate,
selected artifacts, and Stage 2.1 semantic hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from operational_dataset import read_json, write_json


FORMAT = "eq_rcp_stage2_1_closeout"
VERSION = 1
REPRESENTATION_REQUIRED_CRITERIA = [
    "positive_q99",
    "target_q99",
    "paired_ci",
    "threshold_near_noninferior",
    "q999_control",
    "code_oracle_information",
    "seed_stability",
    "runtime_metadata_budget",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_artifacts(manifest: dict[str, Any], selected: str) -> list[dict[str, Any]]:
    artifacts = [item for item in manifest["artifacts"] if item.get("name") == selected]
    if not artifacts:
        raise ValueError("selected representation has no frozen model artifacts")
    result = [
        {
            "name": item["name"],
            "seed": int(item["seed"]),
            "relative_path": item["relative_path"],
            "artifact_sha256": item["artifact_sha256"],
            "model_semantic_sha256": item["model_semantic_sha256"],
            "parameterization_contract": item["parameterization_contract"],
            "encoder_inputs": item["encoder_inputs"],
        }
        for item in artifacts
    ]
    return sorted(result, key=lambda item: item["seed"])


def _source_binding(result_dir: Path, manifest: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage2_1_manifest_sha256": _sha256(result_dir / "stage2_1_manifest.json"),
        "stage2_1_results_semantic_sha256": manifest["stage2_1_results_semantic_sha256"],
        "original_mixed_gate_report_sha256": _sha256(result_dir / "gate_report.json"),
        "original_mixed_gate_decision": gate["decision"],
        "selection_report_sha256": _sha256(result_dir / "selection_report.json"),
        "calibration_manifest_sha256": _sha256(result_dir / "calibration_manifest.json"),
    }


def build_split_reports(result_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(result_dir / "stage2_1_manifest.json")
    gate = read_json(result_dir / "gate_report.json")
    selection = read_json(result_dir / "selection_report.json")
    calibration = read_json(result_dir / "calibration_manifest.json")
    selected = selection["selected_flat_representation"]
    if selected is None:
        raise ValueError("cannot close out a run without a selected representation")
    if selected != manifest.get("selected_flat_representation") or selected != gate.get(
        "selected_flat_representation"
    ):
        raise ValueError("source run disagrees on the frozen selected representation")

    final_criteria = gate.get("final_test_criteria") or {}
    criteria = final_criteria.get("criteria") or {}
    missing = [name for name in REPRESENTATION_REQUIRED_CRITERIA if name not in criteria]
    if missing:
        raise ValueError(f"source Gate lacks representation criteria: {missing}")
    representation_pass = all(bool(criteria[name]) for name in REPRESENTATION_REQUIRED_CRITERIA)
    if representation_pass and bool(gate.get("formal_eligible")):
        representation_status = "GO_CPP_REPRESENTATION"
    elif representation_pass and bool(gate.get("development_full_eligible")):
        representation_status = "GO_STAGE3_REPRESENTATION_DEV"
    elif representation_pass:
        representation_status = "PENDING_FORMAL_STAGE2_1"
    else:
        representation_status = "NO_GO_REPRESENTATION"

    artifact_binding = _selected_artifacts(manifest, selected)
    source = _source_binding(result_dir, manifest, gate)
    representation = {
        "format": FORMAT,
        "version": VERSION,
        "report_type": "representation_gate_report",
        "status": representation_status,
        "selected_flat_representation": selected,
        "selected_parameterization_contract": selection["selected_parameterization_contract"],
        "safety_fallback": selection["safety_fallback"],
        "frozen_selected_artifacts": artifact_binding,
        "required_criteria": REPRESENTATION_REQUIRED_CRITERIA,
        "criteria": {name: bool(criteria[name]) for name in REPRESENTATION_REQUIRED_CRITERIA},
        "informational_estimator_criteria": {
            name: bool(criteria[name])
            for name in ["diagnostic_coverage", "diagnostic_risk"]
            if name in criteria
        },
        "representation_pass": representation_pass,
        "evaluation_role": "final_test",
        "risk_status": "PENDING_RISK_ESTIMATOR",
        "source_run": source,
        "historical_gate_preserved": True,
    }

    entries = [item for item in calibration["entries"] if item["method"] == selected]
    expected_seeds = [item["seed"] for item in artifact_binding]
    if sorted(int(item["seed"]) for item in entries) != expected_seeds:
        raise ValueError("risk diagnostics do not cover the frozen selected artifact seeds")
    estimator = {
        "format": FORMAT,
        "version": VERSION,
        "report_type": "estimator_risk_diagnostic_report",
        "status": "PENDING_RISK_ESTIMATOR",
        "selected_flat_representation": selected,
        "frozen_selected_artifacts": artifact_binding,
        "risk_unit": calibration["risk_unit"],
        "event_level_calibration_pass": all(
            bool(item["scalar_risk"]["pass"] and item["code_oracle_risk"]["pass"])
            for item in entries
        ),
        "calibration_query_count": min(int(item["calibration_query_count"]) for item in entries),
        "diagnostic_entries": entries,
        "formal_guarantee": False,
        "blocking_reasons": [
            "events_within_a_query_are_not_certified_independent",
            "query_level_full_control_flow_risk_has_not_been_calibrated",
            "the_existing_final_test_is_consumed_and_cannot_certify_a_modified_estimator",
        ],
        "authorized_actions": ["offline_stage3_representation_development"],
        "forbidden_actions": ["real_hnsw_pruning", "GO_REAL_PRUNING_RISK"],
        "source_run": source,
        "historical_gate_preserved": True,
    }
    return representation, estimator


def build_reassessment(
    result_dir: Path,
    representation: dict[str, Any],
    estimator: dict[str, Any],
    representation_path: Path,
    estimator_path: Path,
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "version": VERSION,
        "report_type": "stage2_1_read_only_reassessment",
        "representation_status": representation["status"],
        "risk_status": estimator["status"],
        "selected_flat_representation": representation["selected_flat_representation"],
        "representation_gate_report_sha256": _sha256(representation_path),
        "estimator_risk_diagnostic_report_sha256": _sha256(estimator_path),
        "source_run": representation["source_run"],
        "source_result_directory_name": result_dir.name,
        "original_mixed_gate_was_not_modified": True,
        "decision_rule": (
            "The representation status authorizes only the stated representation-development "
            "scope. The independent risk status controls pruning authorization and cannot "
            "overwrite the representation decision."
        ),
    }


def generate(result_dir: Path, output_dir: Path) -> dict[str, Any]:
    tracked = [
        result_dir / "stage2_1_manifest.json",
        result_dir / "gate_report.json",
        result_dir / "selection_report.json",
        result_dir / "calibration_manifest.json",
    ]
    before = {path.name: _sha256(path) for path in tracked}
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"closeout output directory must be empty: {output_dir}")
    representation, estimator = build_split_reports(result_dir)
    representation_path = output_dir / "representation_gate_report.json"
    estimator_path = output_dir / "estimator_risk_diagnostic_report.json"
    write_json(representation_path, representation)
    write_json(estimator_path, estimator)
    reassessment = build_reassessment(
        result_dir, representation, estimator, representation_path, estimator_path
    )
    write_json(output_dir / "stage2_1_read_only_reassessment.json", reassessment)
    after = {path.name: _sha256(path) for path in tracked}
    if before != after:
        raise RuntimeError("source Stage 2.1 run changed during read-only reassessment")
    return reassessment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.result_dir, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
