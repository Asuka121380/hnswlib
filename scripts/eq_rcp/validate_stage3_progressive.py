#!/usr/bin/env python3
"""Validate Stage 3 freeze order, artifacts, same-bit contract, and outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from operational_dataset import read_json, write_json
from stage3_quantization import STAGE3_FORMAT, STAGE3_VERSION


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(input_manifest_path: Path, result_dir: Path) -> dict[str, Any]:
    failures: list[str] = []
    source = read_json(input_manifest_path)
    manifest = read_json(result_dir / "stage3_manifest.json")
    gate = read_json(result_dir / "gate_report.json")
    selection = read_json(result_dir / "selection_report.json")
    calibration = read_json(result_dir / "calibration_manifest.json")
    final = read_json(result_dir / "final_metrics.json")
    if manifest.get("format") != STAGE3_FORMAT or manifest.get("version") != STAGE3_VERSION:
        failures.append("invalid_stage3_manifest_format")
    if source.get("status") != "READY_FOR_OFFLINE_STAGE3" or not source.get("experiment_ready"):
        failures.append("source_input_not_ready")
    if manifest.get("input_manifest_sha256") != _sha(input_manifest_path):
        failures.append("input_manifest_hash_mismatch")
    protocol = manifest.get("protocol") or {}
    required_protocol = {
        "fresh_evaluation_queries": True,
        "flat_representation_reselection": False,
        "selection_frozen_before_calibration": True,
        "cutoffs_frozen_before_final_test": True,
        "risk_is_diagnostic_only": True,
    }
    for name, expected in required_protocol.items():
        if protocol.get(name) != expected:
            failures.append(f"protocol_{name}")
    expected_outputs = manifest.get("outputs") or {}
    for name, expected in expected_outputs.items():
        path = result_dir / name
        if not path.is_file() or _sha(path) != expected:
            failures.append(f"output_hash_{name}")
    artifact_candidates: set[str] = set()
    for artifact in manifest.get("artifacts", []):
        path = result_dir / Path(artifact["relative_path"])
        if not path.is_file() or _sha(path) != artifact["artifact_sha256"]:
            failures.append(f"artifact_hash_{artifact.get('candidate')}_{artifact.get('seed')}")
        if int(artifact["total_code_bytes"]) != 32:
            failures.append(f"artifact_not_32_bytes_{artifact.get('candidate')}_{artifact.get('seed')}")
        if artifact.get("parameterization_contract") != "unit_direction_exact_gain":
            failures.append(f"artifact_parameterization_{artifact.get('candidate')}")
        artifact_candidates.add(str(artifact.get("candidate")))
    if any("oae" in name for name in artifact_candidates):
        failures.append("flat_oae_family_was_reopened")
    selected = selection.get("selected_candidate")
    fallback = selection.get("fallback_candidate")
    allowed_final = {selected, fallback}
    actual_final = {row.get("candidate") for row in final.get("rows", [])}
    if actual_final != allowed_final:
        failures.append("final_test_evaluated_unfrozen_candidates")
    if calibration.get("cutoffs_frozen_before_final_test") is not True:
        failures.append("cutoff_freeze_order")
    if gate.get("risk_status") != "PENDING_RISK_ESTIMATOR" or gate.get(
        "real_pruning_authorized"
    ) is not False:
        failures.append("risk_scope_violation")
    if gate.get("decision") not in {"GO_PROGRESSIVE_STAGE3_DEV", "RETAIN_FROZEN_FLAT_32B"}:
        failures.append("invalid_stage3_decision")
    status = "PASS" if not failures else "FAIL"
    return {
        "format": "eq_rcp_stage3_validation",
        "version": 1,
        "status": status,
        "failures": failures,
        "selected_candidate": gate.get("selected_candidate"),
        "risk_status": gate.get("risk_status"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = validate(args.input_manifest, args.result_dir)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
