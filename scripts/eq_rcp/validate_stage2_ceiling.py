#!/usr/bin/env python3
"""Validate Stage 2 artifacts, hashes, fixed-bit contract, and Gate logic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from operational_dataset import read_json, sha256_file, write_json
from run_stage2_ceiling import (
    _gate_report,
    _semantic_artifacts,
    _semantic_comparison_rows,
    _semantic_config,
)
from stage2_quantization import (
    STAGE2_FORMAT,
    STAGE2_VERSION,
    Stage1OperationalData,
    model_semantic_sha256,
    results_semantic_sha256,
)


def validate(config_path: Path, result_dir: Path) -> dict[str, object]:
    config = read_json(config_path)
    manifest = read_json(result_dir / "stage2_manifest.json")
    if manifest.get("format") != STAGE2_FORMAT or manifest.get("version") != STAGE2_VERSION:
        raise ValueError("unsupported Stage 2 manifest format/version")
    if config.get("format") != STAGE2_FORMAT or config.get("version") != STAGE2_VERSION:
        raise ValueError("unsupported Stage 2 config format/version")
    for output in manifest["outputs"].values():
        path = result_dir / output["relative_path"]
        if sha256_file(path) != output["sha256"]:
            raise ValueError(f"Stage 2 output SHA-256 mismatch: {path.name}")
    for artifact in manifest["artifacts"]:
        path = result_dir / artifact["relative_path"]
        if sha256_file(path) != artifact["artifact_sha256"]:
            raise ValueError(f"Stage 2 model SHA-256 mismatch: {path.name}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if model_semantic_sha256(arrays, artifact) != artifact["model_semantic_sha256"]:
            raise ValueError(f"Stage 2 model semantic SHA-256 mismatch: {path.name}")
        if int(artifact["code_bytes"]) != int(config["quantizer"]["code_bytes"]):
            raise ValueError(f"model violates fixed code budget: {path.name}")

    comparison = pq.read_table(result_dir / "comparison.parquet").to_pylist()
    per_query = pq.read_table(result_dir / "per_query_metrics.parquet").to_pylist()
    expected_pairs = {
        (str(method), int(seed))
        for method in config["methods"]
        for seed in config["seeds"]
    }
    actual_pairs = {(str(row["method"]), int(row["seed"])) for row in comparison}
    if actual_pairs != expected_pairs:
        raise ValueError("comparison rows do not cover exact method/seed matrix")

    data = Stage1OperationalData(Path(config["stage1_dataset_dir"]))
    try:
        if data.manifest["dataset_semantic_sha256"] != manifest["stage1_dataset_semantic_sha256"]:
            raise ValueError("Stage 1 semantic hash differs from Stage 2 manifest")
        selection = config["selection"]
        query_selection = {}
        fully_sampled = True
        for role in ["quantizer_train", "decoder_train", "evaluation"]:
            maximum = selection[role].get("max_queries")
            if maximum is not None:
                fully_sampled = False
            query_selection[role] = data.split_query_ids(
                selection[role]["split"],
                None if maximum is None else int(maximum),
                int(selection["seed"]),
            )
            recorded = manifest["selection"][role]
            if int(recorded["query_count"]) != len(query_selection[role]):
                raise ValueError("query-selection count mismatch")
            event_count = len(data.event_indices(query_selection[role]))
            if int(recorded["event_count"]) != event_count:
                raise ValueError("event-selection count mismatch")
        roles = list(query_selection)
        for left in range(len(roles)):
            for right in range(left + 1, len(roles)):
                if np.intersect1d(query_selection[roles[left]], query_selection[roles[right]]).size:
                    raise ValueError("Stage 2 validator found query leakage")
        gate = _gate_report(config, data, comparison, per_query, fully_sampled)
        recorded_gate = read_json(result_dir / "gate_report.json")
        if gate != recorded_gate:
            raise ValueError("Gate report is not reproducible from comparison data")
        semantic_payload = {
            "format": STAGE2_FORMAT,
            "version": STAGE2_VERSION,
            "semantic_config": _semantic_config(config),
            "stage1_dataset_semantic_sha256": manifest[
                "stage1_dataset_semantic_sha256"
            ],
            "query_selection": {
                role: query_selection[role].tolist()
                for role in sorted(query_selection)
            },
            "comparison": _semantic_comparison_rows(comparison),
            "per_query": per_query,
            "artifacts": _semantic_artifacts(manifest["artifacts"]),
            "gate": gate,
        }
        semantic_hash = results_semantic_sha256(semantic_payload)
        if semantic_hash != manifest["stage2_results_semantic_sha256"]:
            raise ValueError("Stage 2 results semantic SHA-256 mismatch")
        report: dict[str, object] = {
            "status": "PASS",
            "stage2_results_semantic_sha256": semantic_hash,
            "method_seed_pairs": len(actual_pairs),
            "query_split_isolation": "PASS",
            "fixed_code_bytes": int(config["quantizer"]["code_bytes"]),
            "gate_decision": gate["decision"],
            "diagnostic_gate_pass": gate["diagnostic_gate_pass"],
        }
        write_json(result_dir / "validation_report.json", report)
        return report
    finally:
        data.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = validate(args.config, args.result_dir)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
