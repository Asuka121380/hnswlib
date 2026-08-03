#!/usr/bin/env python3
"""Write a reproducibility manifest for a safe-bound diagnostic run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--index-path", required=True, type=Path)
    parser.add_argument("--sidecar-path", required=True, type=Path)
    parser.add_argument("--build-root", required=True, type=Path)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--git-branch", required=True)
    parser.add_argument("--working-tree-dirty", required=True)
    parser.add_argument("--run-stage-b", choices=("0", "1"), required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def version(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return result.stdout.splitlines()[0] if result.stdout else "unknown"
    except OSError as error:
        return f"unavailable: {error}"


def input_entry(path: Path) -> dict:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def main() -> int:
    args = parse_args()
    manifest = {
        "format": "hnswlib_v0_safe_bound_run_manifest",
        "format_version": 1,
        "shadow_schema_version": 2,
        "enabled_methods": ["current"],
        "validation_tolerance": 0.0,
        "repo_root": str(args.repo_root.resolve()),
        "build_root": str(args.build_root.resolve()),
        "producer_commit": args.producer_commit,
        "git_branch": args.git_branch,
        "working_tree_dirty": args.working_tree_dirty.lower() == "true",
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", "none"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST", "unknown"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", "1"),
        },
        "inputs": {
            "dataset_config": input_entry(args.dataset_config),
            "index": input_entry(args.index_path),
            "sidecar": input_entry(args.sidecar_path),
        },
        "experiment": {
            "dataset": "gist1m",
            "query_start": 0,
            "stage_a_query_count": 100,
            "stage_b_query_count": 1000 if args.run_stage_b == "1" else 0,
            "k": 10,
            "ef_search": 200,
            "shadow_sample_modulus": 16,
            "shadow_sample_remainder": 0,
            "real_pruning_enabled": False,
        },
        "build": {
            "type": "Release",
            "HNSWLIB_ENABLE_EDGE_QUANT_V0": True,
            "HNSWLIB_ENABLE_V0_SHADOW_VALIDATION": True,
            "HNSWLIB_ENABLE_V0_REAL_PRUNING": False,
            "cmake": version(["cmake", "--version"]),
        },
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
