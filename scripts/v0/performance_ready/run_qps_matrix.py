#!/usr/bin/env python3
"""Run a validated V0 QPS contract as randomized complete blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qps_config import (
    ConfigError,
    RESOURCE_PROFILES,
    configurations,
    load_config,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_schedule(
    configs: list[dict[str, Any]], blocks: int, seed: int,
) -> list[list[dict[str, Any]]]:
    schedule: list[list[dict[str, Any]]] = []
    for block in range(blocks):
        order = [dict(config) for config in configs]
        random.Random(seed + block).shuffle(order)
        schedule.append(order)
    return schedule


def file_identity(path: Path, *, include_sha256: bool) -> dict[str, Any]:
    stat = path.stat()
    identity: dict[str, Any] = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_sha256:
        identity["sha256"] = sha256(path)
    return identity


def write_or_verify(path: Path, value: object) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise SystemExit(f"existing identity file does not match: {path}")
        return
    atomic_write_json(path, value)


def resource_observation(profile: str) -> dict[str, Any]:
    return {
        "resource_profile": profile,
        "exclusive": profile == "formal-exclusive",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_cpus_on_node": os.environ.get("SLURM_CPUS_ON_NODE"),
        "slurm_job_cpus_per_node": os.environ.get("SLURM_JOB_CPUS_PER_NODE"),
        "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
        "slurm_cpu_bind": os.environ.get("SLURM_CPU_BIND"),
        "runner_threads": 1,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resource-profile", required=True,
                        choices=RESOURCE_PROFILES)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--index-path", required=True, type=Path)
    parser.add_argument("--sidecar-path", required=True, type=Path)
    parser.add_argument("--query-path", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--cmake-cache", required=True, type=Path)
    parser.add_argument("--experiment-commit", required=True)
    parser.add_argument("--numa-policy", required=True)
    parser.add_argument("--power-policy", required=True)
    parser.add_argument("--cpu-frequency-policy", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        requested, resolved = load_config(args.config, args.resource_profile)
    except ConfigError as error:
        raise SystemExit(f"invalid QPS configuration: {error}") from error
    configs = configurations(resolved)
    schedule = build_schedule(configs, resolved["blocks"], resolved["seed"])
    if args.dry_run:
        print(json.dumps({
            "resolved_config": resolved,
            "schedule": [[config["id"] for config in block]
                         for block in schedule],
        }, indent=2, sort_keys=True))
        return 0

    required = (
        args.runner, args.index_path, args.sidecar_path,
        args.query_path, args.cmake_cache,
    )
    for path in required:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")

    contract: dict[str, Any] = {
        "schema_version": 1,
        "kernel": "raw_fast_v1",
        "experiment_commit": args.experiment_commit,
        "resolved_config": resolved,
        "runner": file_identity(args.runner, include_sha256=True),
        "cmake_cache": file_identity(args.cmake_cache, include_sha256=True),
        "index": file_identity(args.index_path, include_sha256=False),
        "sidecar": file_identity(args.sidecar_path, include_sha256=False),
        "query": file_identity(args.query_path, include_sha256=False),
        "numa_policy": args.numa_policy,
        "power_policy": args.power_policy,
        "cpu_frequency_policy": args.cpu_frequency_policy,
        "single_thread_query": True,
    }

    manifest_path = args.run_root / "manifest.json"
    results_dir = args.run_root / "qps"
    was_complete = False
    if manifest_path.exists():
        if not args.resume:
            raise SystemExit("run root already has a manifest; pass --resume")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("contract") != contract:
            raise SystemExit("existing manifest contract does not match")
        if manifest.get("schedule") != schedule:
            raise SystemExit("existing manifest schedule does not match")
        was_complete = manifest.get("status") == "complete"
    else:
        if args.run_root.exists():
            allowed = {
                "logs", "submission.env", "requested_config.json",
                "resolved_config.json", "build_contract.json",
            }
            unexpected = {
                item.name for item in args.run_root.iterdir()
                if item.name not in allowed
            }
            if unexpected:
                raise SystemExit(
                    "run root contains unexpected files: " +
                    ", ".join(sorted(unexpected)))
        args.run_root.mkdir(parents=True, exist_ok=True)
        write_or_verify(args.run_root / "requested_config.json", requested)
        write_or_verify(args.run_root / "resolved_config.json", resolved)
        build_contract = {
            "schema_version": 1,
            "experiment_commit": args.experiment_commit,
            "runner": contract["runner"],
            "cmake_cache": contract["cmake_cache"],
            "kernel": contract["kernel"],
        }
        write_or_verify(args.run_root / "build_contract.json", build_contract)
        results_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "status": "running",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "host": platform.platform(),
            "python": platform.python_version(),
            "contract": contract,
            "schedule": schedule,
            "resource_observations": [resource_observation(
                args.resource_profile)],
            "completed": [],
        }
        atomic_write_json(manifest_path, manifest)

    if args.resume:
        write_or_verify(args.run_root / "requested_config.json", requested)
        write_or_verify(args.run_root / "resolved_config.json", resolved)
        observations = manifest.setdefault("resource_observations", [])
        current_job = os.environ.get("SLURM_JOB_ID")
        if not observations or observations[-1].get("slurm_job_id") != current_job:
            observations.append(resource_observation(args.resource_profile))
        manifest["status"] = "running"
        manifest.pop("last_error", None)
        atomic_write_json(manifest_path, manifest)

    completed_by_id = {
        str(item["run_id"]): item for item in manifest.get("completed", [])
    }
    for block, order in enumerate(schedule):
        for position, config in enumerate(order):
            run_id = f"b{block:02d}-p{position:03d}-{config['id']}"
            output = results_dir / f"{run_id}.json"
            previous = completed_by_id.get(run_id)
            if previous is not None:
                if not output.is_file():
                    raise SystemExit(f"completed result is missing: {output}")
                if sha256(output) != previous.get("sha256"):
                    raise SystemExit(f"completed result checksum mismatch: {output}")
                print(f"REUSE {run_id}")
                continue
            if output.exists():
                raise SystemExit(f"untracked result already exists: {output}")

            command = [
                str(args.runner),
                "--method", str(config["method"]),
                "--index-path", str(args.index_path),
                "--query-path", str(args.query_path),
                "--dimension", str(resolved["dimension"]),
                "--query-start", str(resolved["query_start"]),
                "--query-count", str(resolved["query_count"]),
                "--k", str(resolved["k"]),
                "--ef-search", str(resolved["ef_search"]),
                "--warmup-queries", str(resolved["warmup_queries"]),
                "--repeats", str(resolved["within_process_repeats"]),
                "--prefetch", (
                    "legacy" if config["method"] == "baseline"
                    else str(config["prefetch"])
                ),
                "--cpu", "-1",
                "--output", str(output),
            ]
            if config["method"] != "baseline":
                command.extend((
                    "--sidecar-path", str(args.sidecar_path),
                    "--beta", str(config["beta"]),
                ))
            print(f"START {run_id}", flush=True)
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as error:
                manifest["status"] = "failed"
                manifest["last_error"] = {
                    "run_id": run_id,
                    "returncode": error.returncode,
                    "failed_utc": datetime.now(timezone.utc).isoformat(),
                }
                atomic_write_json(manifest_path, manifest)
                raise
            item = {
                "run_id": run_id,
                "block": block,
                "position": position,
                "config": config,
                "command": command,
                "result": str(Path("qps") / output.name),
                "sha256": sha256(output),
            }
            completed_by_id[run_id] = item
            manifest["completed"] = list(completed_by_id.values())
            atomic_write_json(manifest_path, manifest)
            print(f"DONE {run_id}", flush=True)

    if len(completed_by_id) != resolved["expected_result_count"]:
        raise SystemExit(
            "completed result count does not match resolved configuration")
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest.pop("last_error", None)
    atomic_write_json(manifest_path, manifest)
    (args.run_root / "COMPLETE").write_text(
        resolved["contract_sha256"] + "\n", encoding="utf-8")
    if was_complete:
        print(f"qps_matrix_reuse_complete run_root={args.run_root}")
        return 0
    print(
        f"qps_matrix_complete runs={len(completed_by_id)} "
        f"run_root={args.run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
