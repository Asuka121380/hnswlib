#!/usr/bin/env python3
"""Run recall/DCO measurements for every distinct active QPS case."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qps_config import ConfigError, configurations, load_config


FIELDS = (
    "configuration_id", "ef_search", "beta", "mode", "query_start",
    "query_count", "baseline_recall", "v0_recall", "recall_loss",
    "recall_loss_queries", "catastrophic_queries", "dco_reduction",
    "exact_distance_saved", "first_pruned", "retry_exact",
    "decision_disagreement", "near_threshold_disagreement",
    "relative_difference_max",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def active_quality_cases(resolved: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefetch does not change results, so collapse equivalent QPS cases."""
    result: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for case in configurations(resolved):
        if case["method"] == "baseline":
            continue
        key = (int(case["ef_search"]), str(case["beta"]), str(case["method"]))
        if key in seen:
            continue
        seen.add(key)
        item = dict(case)
        item["id"] = (
            f"{case['method']}-beta{str(case['beta']).replace('.', 'p')}"
            f"-ef{case['ef_search']}")
        result.append(item)
    return result


def summary_row(
    case: dict[str, Any], data: dict[str, Any], resolved: dict[str, Any],
) -> dict[str, Any]:
    return {
        "configuration_id": case["id"],
        "ef_search": case["ef_search"],
        "beta": case["beta"],
        "mode": case["method"],
        "query_start": resolved["query_start"],
        "query_count": data["query_count"],
        "baseline_recall": data["mean_baseline_recall_at_k"],
        "v0_recall": data["mean_v0_recall_at_k"],
        "recall_loss": data["mean_recall_loss"],
        "recall_loss_queries": data["recall_loss_queries"],
        "catastrophic_queries": data["catastrophic_recall_loss_queries"],
        "dco_reduction": data["baseline_relative_exact_dco_reduction"],
        "exact_distance_saved": data["exact_distance_saved"],
        "first_pruned": data["approx_first_pruned"],
        "retry_exact": data["approx_retry_exact_distance"],
        "decision_disagreement": data["fast_reference_decision_disagreement"],
        "near_threshold_disagreement":
            data["fast_reference_near_threshold_disagreement"],
        "relative_difference_max":
            data["fast_reference_relative_difference_max"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--index-path", required=True, type=Path)
    parser.add_argument("--sidecar-path", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--experiment-commit", required=True)
    parser.add_argument("--git-branch", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        requested, resolved = load_config(args.config)
    except ConfigError as error:
        raise SystemExit(f"invalid quality configuration: {error}") from error
    required = (args.runner, args.dataset_config, args.index_path, args.sidecar_path)
    for path in required:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    cases = active_quality_cases(resolved)
    if not cases:
        raise SystemExit("quality matrix requires at least one active case")

    contract = {
        "schema_version": 1,
        "experiment_commit": args.experiment_commit,
        "resolved_config": resolved,
        "runner_sha256": sha256(args.runner),
        "dataset_config": str(args.dataset_config.resolve()),
        "index_path": str(args.index_path.resolve()),
        "sidecar_path": str(args.sidecar_path.resolve()),
        "quality_cases": cases,
    }
    manifest_path = args.run_root / "quality_manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise SystemExit("quality run root exists; pass --resume")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("contract") != contract:
            raise SystemExit("existing quality manifest contract does not match")
    else:
        args.run_root.mkdir(parents=True, exist_ok=True)
        atomic_json(args.run_root / "requested_config.json", requested)
        atomic_json(args.run_root / "resolved_config.json", resolved)
        manifest = {
            "schema_version": 1, "status": "running",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "contract": contract, "completed": [],
        }
        atomic_json(manifest_path, manifest)

    completed = {item["configuration_id"]: item
                 for item in manifest.get("completed", [])}
    runs = args.run_root / "quality"
    runs.mkdir(exist_ok=True)
    for case in cases:
        output_dir = runs / str(case["id"])
        summary = output_dir / "summary.json"
        previous = completed.get(case["id"])
        if previous:
            if not summary.is_file() or sha256(summary) != previous["summary_sha256"]:
                raise SystemExit(f"quality result mismatch: {summary}")
            print(f"REUSE {case['id']}")
            continue
        command = [
            str(args.runner),
            "--dataset-config", str(args.dataset_config),
            "--index-path", str(args.index_path),
            "--sidecar-path", str(args.sidecar_path),
            "--output-dir", str(output_dir),
            "--mode", str(case["method"]),
            "--run-id", f"quality-{case['id']}",
            "--query-start", str(resolved["query_start"]),
            "--query-count", str(resolved["query_count"]),
            "--k", str(resolved["k"]),
            "--ef-search", str(case["ef_search"]),
            "--approx-beta", str(case["beta"]),
            "--producer-git-commit", args.experiment_commit,
            "--git-branch", args.git_branch,
            "--working-tree-dirty", "false",
        ]
        print(f"START {case['id']}", flush=True)
        subprocess.run(command, check=True)
        if not summary.is_file():
            raise SystemExit(f"runner did not produce {summary}")
        data = json.loads(summary.read_text(encoding="utf-8"))
        if data.get("status") != "valid":
            raise SystemExit(f"invalid quality result: {summary}")
        completed[case["id"]] = {
            "configuration_id": case["id"], "case": case,
            "summary": str(summary.relative_to(args.run_root)),
            "summary_sha256": sha256(summary), "command": command,
        }
        manifest["completed"] = list(completed.values())
        atomic_json(manifest_path, manifest)
        print(f"DONE {case['id']}", flush=True)

    rows = []
    for case in cases:
        data = json.loads((runs / str(case["id"]) / "summary.json").read_text(
            encoding="utf-8"))
        rows.append(summary_row(case, data, resolved))
    rows.sort(key=lambda row: (
        int(row["ef_search"]), float(row["beta"]), str(row["mode"])))
    output = args.run_root / "quality_summary.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["summary"] = str(output.relative_to(args.run_root))
    atomic_json(manifest_path, manifest)
    print(f"quality_matrix_complete rows={len(rows)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
