#!/usr/bin/env python3
"""Freeze and validate the Stage 0 contract for V0 margin + retry work."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REQUIRED_RUN_FILES = (
    "metadata.json",
    "summary.json",
    "complete.json",
    "query_metrics.csv",
    "shadow_records.csv.gz",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def first_version_line(command: List[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as error:
        return "unavailable: {}".format(error)
    output = completed.stdout.strip().splitlines()
    return output[0] if output else "unavailable"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return value


def parse_recorded_checksums(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not path.exists():
        return result
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        result[Path(parts[1]).name] = parts[0].lower()
    return result


def csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def contract_checks(
    metadata: Dict[str, Any],
    summary: Dict[str, Any],
    complete: Dict[str, Any],
    query_metric_rows: int,
) -> List[Dict[str, Any]]:
    expected_pairs: Iterable[Tuple[str, Any, Any]] = (
        ("dataset", metadata.get("dataset"), "gist1m"),
        ("dimension", metadata.get("dimension"), 960),
        ("query_count", metadata.get("query_count"), 1000),
        ("k", metadata.get("k"), 10),
        ("ef_search", metadata.get("ef_search"), 200),
        ("shadow_sample_modulus", metadata.get("shadow_sample_modulus"), 16),
        ("shadow_sample_remainder", metadata.get("shadow_sample_remainder"), 0),
        ("mode", metadata.get("mode"), "shadow"),
        ("summary_status", summary.get("status"), "valid"),
        ("summary_query_count", summary.get("query_count"), 1000),
        ("mismatch_queries", summary.get("mismatch_queries"), 0),
        ("lower_bound_violation", summary.get("lower_bound_violation"), 0),
        ("complete_status", complete.get("status"), "complete"),
        ("query_metrics_rows", query_metric_rows, 1000),
    )
    checks = []
    for name, actual, expected in expected_pairs:
        checks.append(
            {
                "name": name,
                "actual": actual,
                "expected": expected,
                "passed": actual == expected,
            }
        )
    baseline_recall = float(summary.get("mean_baseline_recall_at_k", -1.0))
    v0_recall = float(summary.get("mean_v0_recall_at_k", -2.0))
    checks.append(
        {
            "name": "baseline_v0_recall_equal",
            "actual": abs(baseline_recall - v0_recall),
            "expected": 0.0,
            "passed": baseline_recall == v0_recall,
        }
    )
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--shadow-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-branch", default="v0-margin-retry")
    parser.add_argument(
        "--baseline-commit",
        default="374b51dfbb6888a5e7bc87b8313113e5060b5c93",
    )
    parser.add_argument("--baseline-branch", default="v0-edge-quantisation")
    parser.add_argument(
        "--initial-clean-observed",
        action="store_true",
        help="Record that the caller inspected a clean worktree before Stage 0 edits.",
    )
    parser.add_argument("--ctest-total", type=int, default=0)
    parser.add_argument("--ctest-failed", type=int, default=-1)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    source = args.shadow_run_dir.resolve()
    output = args.output_dir.resolve()

    if not (repo / ".git").exists():
        raise SystemExit("--repo is not a Git worktree: {}".format(repo))
    missing = [name for name in REQUIRED_RUN_FILES if not (source / name).is_file()]
    if missing:
        raise SystemExit("shadow run is missing: {}".format(", ".join(missing)))
    if output.exists() and any(output.iterdir()) and not args.force:
        raise SystemExit("output directory is not empty; pass --force: {}".format(output))
    output.mkdir(parents=True, exist_ok=True)

    metadata = load_json(source / "metadata.json")
    summary = load_json(source / "summary.json")
    complete = load_json(source / "complete.json")
    query_rows = csv_row_count(source / "query_metrics.csv")
    checks = contract_checks(metadata, summary, complete, query_rows)

    actual_checksums = {name: sha256_file(source / name) for name in REQUIRED_RUN_FILES}
    recorded_checksums = parse_recorded_checksums(source / "checksums.sha256")
    checksum_checks = []
    for name, recorded in sorted(recorded_checksums.items()):
        if name not in actual_checksums:
            continue
        checksum_checks.append(
            {
                "file": name,
                "recorded": recorded,
                "actual": actual_checksums[name],
                "passed": recorded == actual_checksums[name],
            }
        )

    current_branch = run_git(repo, "branch", "--show-current")
    head_commit = run_git(repo, "rev-parse", "HEAD")
    baseline_ref_commit = run_git(repo, "rev-parse", args.baseline_branch)
    status_porcelain = run_git(repo, "status", "--porcelain")
    branch_checks = [
        {
            "name": "expected_branch",
            "actual": current_branch,
            "expected": args.expected_branch,
            "passed": current_branch == args.expected_branch,
        },
        {
            "name": "baseline_branch_commit",
            "actual": baseline_ref_commit,
            "expected": args.baseline_commit,
            "passed": baseline_ref_commit == args.baseline_commit,
        },
        {
            "name": "implementation_branch_start",
            "actual": head_commit,
            "expected": args.baseline_commit,
            "passed": head_commit == args.baseline_commit,
        },
        {
            "name": "initial_clean_observed",
            "actual": bool(args.initial_clean_observed),
            "expected": True,
            "passed": bool(args.initial_clean_observed),
        },
        {
            "name": "v0_ctest_suite",
            "actual": "{}/{} failed".format(args.ctest_failed, args.ctest_total),
            "expected": "0/positive failed",
            "passed": args.ctest_total > 0 and args.ctest_failed == 0,
        },
    ]

    source_provenance_warnings = []
    if metadata.get("producer_git_commit") != args.baseline_commit:
        source_provenance_warnings.append(
            "Frozen shadow input was produced by commit {}, not the current baseline {}.".format(
                metadata.get("producer_git_commit"), args.baseline_commit
            )
        )
    if str(metadata.get("working_tree_dirty", "")).lower() == "true":
        source_provenance_warnings.append(
            "Frozen shadow input metadata records a dirty producer worktree."
        )

    hard_checks = checks + checksum_checks + branch_checks
    gate_passed = all(bool(item["passed"]) for item in hard_checks)
    input_contract = {
        "format": "v0_margin_retry_input_contract",
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_directory": str(source),
        "source_files_sha256": actual_checksums,
        "recorded_checksum_verification": checksum_checks,
        "source_metadata": metadata,
        "source_summary": summary,
        "source_complete": complete,
        "contract_checks": checks,
        "source_provenance_warnings": source_provenance_warnings,
        "gate0_input_contract_passed": all(item["passed"] for item in checks + checksum_checks),
    }
    baseline_manifest = {
        "format": "v0_margin_retry_baseline_manifest",
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo),
        "baseline_branch": args.baseline_branch,
        "baseline_commit": args.baseline_commit,
        "implementation_branch": current_branch,
        "head_commit": head_commit,
        "working_tree_dirty_after_implementation": bool(status_porcelain),
        "working_tree_status": status_porcelain.splitlines(),
        "branch_checks": branch_checks,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "git": first_version_line(["git", "--version"]),
            "cmake": first_version_line(["cmake", "--version"]),
            "cxx": first_version_line(["g++", "--version"]),
            "cpu_count": os.cpu_count(),
        },
        "test_validation": {
            "build_directory": str(args.build_dir.resolve()) if args.build_dir else None,
            "ctest_total": args.ctest_total,
            "ctest_failed": args.ctest_failed,
            "passed": args.ctest_total > 0 and args.ctest_failed == 0,
        },
        "gate0_passed": gate_passed,
        "source_provenance_warnings": source_provenance_warnings,
    }
    build_configurations = {
        "format": "v0_margin_retry_build_configurations",
        "format_version": 1,
        "stage0_validation_build": {
            "CMAKE_BUILD_TYPE": "Release",
            "HNSWLIB_ENABLE_EDGE_QUANT_V0": True,
            "HNSWLIB_ENABLE_V0_SHADOW_VALIDATION": True,
            "HNSWLIB_ENABLE_V0_REAL_PRUNING": True,
            "HNSWLIB_BUILD_FAISS_PQ_TOOLS": False,
        },
        "note": "Actual configure/build/test commands and results are recorded in STAGE0_BASELINE_REPORT.md.",
    }

    write_json(output / "input_contract.json", input_contract)
    write_json(output / "baseline_manifest.json", baseline_manifest)
    write_json(output / "build_configurations.json", build_configurations)
    shutil.copy2(source / "query_metrics.csv", output / "baseline_query_metrics.csv")
    with (output / "checksums.sha256").open("w", encoding="utf-8") as handle:
        for name, digest in sorted(actual_checksums.items()):
            handle.write("{}  {}\n".format(digest, source / name))

    report_lines = [
        "# Stage 0 Baseline Report",
        "",
        "- Gate 0 status: **{}**".format("PASS" if gate_passed else "FAIL"),
        "- Source branch: `{}`".format(args.baseline_branch),
        "- Frozen baseline commit: `{}`".format(args.baseline_commit),
        "- Implementation branch: `{}`".format(current_branch),
        "- Frozen shadow input: `{}`".format(source),
        "- Baseline Recall@10: `{}`".format(summary.get("mean_baseline_recall_at_k")),
        "- V0 Recall@10: `{}`".format(summary.get("mean_v0_recall_at_k")),
        "- Query count: `{}`".format(summary.get("query_count")),
        "- Shadow records written: `{}`".format(summary.get("shadow_records_written")),
        "- CTest: `{}` total, `{}` failed".format(args.ctest_total, args.ctest_failed),
        "",
        "## Hard checks",
        "",
    ]
    for item in hard_checks:
        report_lines.append(
            "- [{}] `{}`: actual=`{}`, expected=`{}`".format(
                "x" if item["passed"] else " ",
                item.get("name", item.get("file")),
                item.get("actual"),
                item.get("expected", item.get("recorded")),
            )
        )
    report_lines.extend(["", "## Provenance limitations", ""])
    if source_provenance_warnings:
        report_lines.extend("- {}".format(item) for item in source_provenance_warnings)
    else:
        report_lines.append("- None.")
    report_lines.extend(
        [
            "",
            "The provenance warnings do not alter the bytes frozen for Stage 1, but a later fresh rerun is required before claiming full end-to-end reproduction from the current baseline commit.",
            "",
        ]
    )
    (output / "STAGE0_BASELINE_REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print("stage0_gate={}".format("PASS" if gate_passed else "FAIL"))
    return 0 if gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
