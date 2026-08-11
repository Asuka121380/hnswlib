#!/usr/bin/env python3
"""Fail-closed validator for a completed OAE dependence run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oae_dependence_core import semantic_sha256, sha256_file, write_json


REQUIRED = {
    "input_manifest.json", "experiment_contract.json", "training_query_manifest.json",
    "validation_query_manifest.json", "probe_matrix.npy", "edge_group_model.npz",
    "group_statistics.npz", "primary_per_query_metrics.parquet", "primary_report.json",
    "v0_residual_per_query_metrics.parquet", "v0_residual_report.json",
    "synthetic_test_report.json", "run_manifest.json",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    errors = [f"missing artifact: {name}" for name in sorted(REQUIRED) if not (run_dir / name).is_file()]
    if not errors:
        input_manifest = json.loads((run_dir / "input_manifest.json").read_text(encoding="utf-8"))
        train = json.loads((run_dir / "training_query_manifest.json").read_text(encoding="utf-8"))
        validation = json.loads((run_dir / "validation_query_manifest.json").read_text(encoding="utf-8"))
        synthetic = json.loads((run_dir / "synthetic_test_report.json").read_text(encoding="utf-8"))
        run = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        if not set(train["original_query_ids"]).isdisjoint(validation["original_query_ids"]):
            errors.append("training and validation query IDs overlap")
        if not input_manifest["declarations"].get("official_gist_query_not_used"):
            errors.append("official query exclusion declaration is absent")
        if semantic_sha256(input_manifest.get("semantic_payload")) != input_manifest.get("semantic_sha256"):
            errors.append("input semantic hash mismatch")
        if synthetic.get("status") != "PASS":
            errors.append("synthetic controls did not pass")
        if not run["declarations"].get("validation_parameters_frozen"):
            errors.append("validation parameters were not declared frozen")
        for name, binding in run["artifacts"].items():
            path = run_dir / name
            if not path.is_file() or sha256_file(path) != binding["sha256"]:
                errors.append(f"artifact digest mismatch: {name}")
        stored = run.pop("semantic_sha256")
        if semantic_sha256(run) != stored:
            errors.append("run semantic hash mismatch")
    report = {"status": "PASS" if not errors else "FAIL", "errors": errors,
              "run_dir": str(run_dir)}
    output = args.output or (run_dir / "validation_report.json")
    write_json(output, report)
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
