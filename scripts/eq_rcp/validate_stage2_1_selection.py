#!/usr/bin/env python3
"""Fail-closed validator for an EQ-RCP Stage 2.1 result directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import numpy as np

from operational_dataset import read_json, sha256_bytes, write_json
from stage2_quantization import Stage1OperationalData
from stage2_1_quantization import STAGE21_FORMAT, STAGE21_VERSION
from run_stage2_1_selection import REQUIRED_ROLES
from reassess_stage2_1_closeout import build_reassessment, build_split_reports


def _validate_closeout(result_dir: Path, reassessment_dir: Path) -> dict[str, Any]:
    representation_path = reassessment_dir / "representation_gate_report.json"
    estimator_path = reassessment_dir / "estimator_risk_diagnostic_report.json"
    reassessment_path = reassessment_dir / "stage2_1_read_only_reassessment.json"
    representation = read_json(representation_path)
    estimator = read_json(estimator_path)
    reassessment = read_json(reassessment_path)
    expected_representation, expected_estimator = build_split_reports(result_dir)
    if representation != expected_representation:
        raise ValueError("representation Gate report is not the read-only derivation of the source run")
    if estimator != expected_estimator:
        raise ValueError("estimator risk report is not the read-only derivation of the source run")
    expected_reassessment = build_reassessment(
        result_dir,
        representation,
        estimator,
        representation_path,
        estimator_path,
    )
    if reassessment != expected_reassessment:
        raise ValueError("read-only reassessment report/hash mismatch")
    if representation["selected_flat_representation"] != estimator["selected_flat_representation"]:
        raise ValueError("representation and estimator reports bind different frozen representations")
    if representation["frozen_selected_artifacts"] != estimator["frozen_selected_artifacts"]:
        raise ValueError("representation and estimator reports bind different frozen artifacts")
    if estimator["status"] != "PENDING_RISK_ESTIMATOR":
        raise ValueError("Stage 2.1 event diagnostics cannot authorize real pruning")
    if representation["risk_status"] != estimator["status"]:
        raise ValueError("estimator status was allowed to overwrite or diverge from representation status")
    if reassessment["representation_status"] != representation["status"]:
        raise ValueError("reassessment representation status mismatch")
    if reassessment["risk_status"] != estimator["status"]:
        raise ValueError("reassessment risk status mismatch")
    return {
        "representation_status": representation["status"],
        "risk_status": estimator["status"],
        "closeout_status": "PASS",
    }


def validate(
    config_path: Path,
    result_dir: Path,
    reassessment_dir: Path | None = None,
) -> dict[str, Any]:
    config = read_json(config_path)
    manifest = read_json(result_dir / "stage2_1_manifest.json")
    gate = read_json(result_dir / "gate_report.json")
    if manifest.get("format") != STAGE21_FORMAT or manifest.get("version") != STAGE21_VERSION:
        raise ValueError("unsupported Stage 2.1 manifest")
    if config.get("format") != STAGE21_FORMAT or config.get("version") != STAGE21_VERSION:
        raise ValueError("config/manifest format mismatch")
    protocol = manifest.get("protocol")
    if not protocol or protocol.get("evaluator_protocol_version") != 2:
        raise ValueError("legacy three-way Stage 2.1 manifest is not Gate-eligible")
    if protocol.get("required_roles") != REQUIRED_ROLES:
        raise ValueError("five-way protocol role order mismatch")
    for flag in [
        "representation_frozen_before_risk_calibration",
        "risk_cutoffs_frozen_before_final_test",
        "final_test_consumed",
    ]:
        if protocol.get(flag) is not True:
            raise ValueError(f"invalid freeze-order protocol flag: {flag}")
    selection = config.get("selection", {})
    if set(selection) - {"seed"} != set(REQUIRED_ROLES):
        raise ValueError("config must define exactly five query roles")
    if set(manifest.get("selection", {})) != set(REQUIRED_ROLES):
        raise ValueError("manifest must record exactly five query roles")
    for name, expected in manifest["outputs"].items():
        actual = hashlib.sha256((result_dir / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"output hash mismatch: {name}")
    selection_rows = pq.read_table(result_dir / "selection_comparison.parquet").to_pylist()
    expected_pairs = len(config["methods"]) * len(config["seeds"])
    if len(selection_rows) != expected_pairs:
        raise ValueError("selection method/seed result count mismatch")
    if any(int(row["code_bytes"]) != int(config["quantizer"]["code_bytes"]) for row in selection_rows):
        raise ValueError("same-bit contract violated")
    if {row["method"] for row in selection_rows} != set(config["methods"]):
        raise ValueError("method set mismatch")
    if any(row.get("evaluation_role") != "model_selection" for row in selection_rows):
        raise ValueError("selection metrics contain a non-selection role")
    rows = pq.read_table(result_dir / "comparison.parquet").to_pylist()
    selection_report = read_json(result_dir / "selection_report.json")
    calibration = read_json(result_dir / "calibration_manifest.json")
    selected = selection_report["selected_flat_representation"]
    audit_methods = {name for name in [selected, "direct_delta_opq"] if name is not None}
    if len(rows) != len(audit_methods) * len(config["seeds"]):
        raise ValueError("final audit method/seed result count mismatch")
    if {row["method"] for row in rows} != audit_methods:
        raise ValueError("final test evaluated an unfrozen method or omitted baseline")
    if any(row.get("evaluation_role") != "final_test" for row in rows):
        raise ValueError("final comparison contains a non-final role")
    entries = calibration.get("entries", [])
    if len(entries) != len(audit_methods) * len(config["seeds"]):
        raise ValueError("calibration entry count mismatch")
    if {(item["method"], int(item["seed"])) for item in entries} != {
        (method, int(seed)) for method in audit_methods for seed in config["seeds"]
    }:
        raise ValueError("calibration methods/seeds do not match frozen final audit")
    if calibration.get("selected_flat_representation") != selected:
        raise ValueError("selection/calibration representation mismatch")
    if calibration.get("risk_unit") != "event_conditioned_on_good_candidate":
        raise ValueError("unsupported Stage 2.1 calibration risk unit")
    for entry in entries:
        for key in ["scalar_risk", "code_oracle_risk"]:
            risk = entry.get(key)
            if not risk or not isinstance(risk.get("pass"), bool):
                raise ValueError(f"missing frozen risk UCB: {key}")
            if bool(risk["pass"]) != (
                float(risk["upper_confidence_bound"]) <= float(risk["target"])
            ):
                raise ValueError(f"inconsistent risk UCB decision: {key}")
    if protocol["selection_report_sha256"] != hashlib.sha256(
        (result_dir / "selection_report.json").read_bytes()
    ).hexdigest():
        raise ValueError("selection freeze hash mismatch")
    if protocol["calibration_manifest_sha256"] != hashlib.sha256(
        (result_dir / "calibration_manifest.json").read_bytes()
    ).hexdigest():
        raise ValueError("calibration freeze hash mismatch")
    if manifest["gate_decision"] != gate["decision"]:
        raise ValueError("gate decision mismatch")
    if gate["selected_flat_representation"] not in set(config["methods"]) | {None}:
        raise ValueError("selected representation is not a preregistered method")
    if selected != gate["selected_flat_representation"] or selected != manifest["selected_flat_representation"]:
        raise ValueError("selection/final Gate/manifest representation mismatch")
    if gate.get("gate_role") != "final_test":
        raise ValueError("Gate is not based on final_test")

    data = Stage1OperationalData(Path(config["stage1_dataset_dir"]))
    try:
        selected_queries: dict[str, np.ndarray] = {}
        for role in REQUIRED_ROLES:
            entry = selection[role]
            if not str(entry.get("split", "")).endswith(role):
                raise ValueError(f"split/role mismatch: {role}")
            maximum = entry.get("max_queries")
            query_ids = data.split_query_ids(
                entry["split"], None if maximum is None else int(maximum),
                int(selection["seed"]),
            )
            selected_queries[role] = query_ids
            record = manifest["selection"][role]
            if record["split"] != entry["split"]:
                raise ValueError(f"split role mismatch: {role}")
            if int(record["query_count"]) != len(query_ids):
                raise ValueError(f"query count mismatch: {role}")
            event_count = len(data.event_indices(query_ids))
            if int(record["event_count"]) != event_count:
                raise ValueError(f"event count mismatch: {role}")
            expected_hash = sha256_bytes(np.asarray(query_ids, dtype="<i8").tobytes())
            if record["query_ids_sha256"] != expected_hash:
                raise ValueError(f"query-ID hash mismatch: {role}")
        for index, left in enumerate(REQUIRED_ROLES):
            for right in REQUIRED_ROLES[index + 1:]:
                if np.intersect1d(selected_queries[left], selected_queries[right]).size:
                    raise ValueError(f"five-way query leakage: {left}/{right}")
    finally:
        data.close()
    for artifact in manifest["artifacts"]:
        path = result_dir / artifact["relative_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["artifact_sha256"]:
            raise ValueError(f"model artifact hash mismatch: {path.name}")
        if artifact["encoder_inputs"] != ["Delta", "static_group"]:
            raise ValueError("encoder-input isolation contract mismatch")
    report = {
        "status": "PASS", "selection_method_seed_pairs": len(selection_rows),
        "final_method_seed_pairs": len(rows),
        "gate_decision": gate["decision"],
        "selected_flat_representation": gate["selected_flat_representation"],
    }
    if reassessment_dir is not None:
        report.update(_validate_closeout(result_dir, reassessment_dir))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--reassessment-dir", type=Path)
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args(argv)
    report = validate(args.config, args.result_dir, args.reassessment_dir)
    if args.report_output is not None:
        write_json(args.report_output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
