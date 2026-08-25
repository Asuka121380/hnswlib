#!/usr/bin/env python3
"""Run the full V0 beta QPS matrix as randomized complete blocks."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_csv(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("CSV argument must not be empty")
    return result


def configuration_id(config: dict[str, object]) -> str:
    if config["method"] == "baseline":
        return "baseline"
    beta = str(config["beta"]).replace(".", "p")
    return f"{config['method']}-beta{beta}-{config['prefetch']}"


def build_configurations(
    betas: list[str], modes: list[str], prefetches: list[str]
) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = [{
        "id": "baseline", "method": "baseline",
        "beta": None, "prefetch": "none",
    }]
    for beta in betas:
        for mode in modes:
            for prefetch in prefetches:
                config: dict[str, object] = {
                    "method": mode,
                    "beta": beta,
                    "prefetch": prefetch,
                }
                config["id"] = configuration_id(config)
                configs.append(config)
    return configs


def build_schedule(
    configs: list[dict[str, object]], blocks: int, seed: int
) -> list[list[dict[str, object]]]:
    schedule: list[list[dict[str, object]]] = []
    for block in range(blocks):
        order = [dict(config) for config in configs]
        random.Random(seed + block).shuffle(order)
        schedule.append(order)
    return schedule


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--index-path", required=True, type=Path)
    parser.add_argument("--sidecar-path", required=True, type=Path)
    parser.add_argument("--query-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cmake-cache", required=True, type=Path)
    parser.add_argument("--betas", required=True)
    parser.add_argument("--modes", required=True)
    parser.add_argument("--prefetches", required=True)
    parser.add_argument("--dimension", type=int, default=960)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-count", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=200)
    parser.add_argument("--warmup-queries", type=int, default=100)
    parser.add_argument("--within-process-repeats", type=int, default=5)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--experiment-commit", required=True)
    parser.add_argument("--numa-policy", required=True)
    parser.add_argument("--power-policy", required=True)
    parser.add_argument("--cpu-frequency-policy", required=True)
    args = parser.parse_args()
    if args.blocks < 5 or args.within_process_repeats < 5:
        parser.error("blocks and within-process-repeats must both be >= 5")
    return args


def main() -> int:
    args = parse_args()
    for path in (
        args.runner, args.index_path, args.sidecar_path,
        args.query_path, args.cmake_cache,
    ):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")

    betas = parse_csv(args.betas)
    modes = parse_csv(args.modes)
    prefetches = parse_csv(args.prefetches)
    if set(modes) != {"approx-no-retry", "approx-retry"}:
        raise SystemExit("modes must be approx-no-retry,approx-retry")
    if set(prefetches) != {"legacy", "gate"}:
        raise SystemExit("prefetches must be legacy,gate")

    configs = build_configurations(betas, modes, prefetches)
    schedule = build_schedule(configs, args.blocks, args.seed)
    contract: dict[str, object] = {
        "schema_version": 1,
        "kernel": "raw_fast_v1",
        "experiment_commit": args.experiment_commit,
        "runner": str(args.runner.resolve()),
        "runner_sha256": sha256(args.runner),
        "cmake_cache": str(args.cmake_cache.resolve()),
        "cmake_cache_sha256": sha256(args.cmake_cache),
        "index_path": str(args.index_path.resolve()),
        "sidecar_path": str(args.sidecar_path.resolve()),
        "query_path": str(args.query_path.resolve()),
        "dimension": args.dimension,
        "query_start": args.query_start,
        "query_count": args.query_count,
        "k": args.k,
        "ef_search": args.ef_search,
        "warmup_queries": args.warmup_queries,
        "within_process_repeats": args.within_process_repeats,
        "blocks": args.blocks,
        "seed": args.seed,
        "betas": betas,
        "modes": modes,
        "prefetches": prefetches,
        "numa_policy": args.numa_policy,
        "power_policy": args.power_policy,
        "cpu_frequency_policy": args.cpu_frequency_policy,
    }

    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("contract") != contract:
            raise SystemExit("existing QPS manifest contract does not match")
        schedule = manifest["schedule"]
        if manifest.get("status") == "complete":
            print(f"full_beta_qps_reuse_complete output={args.output_dir}")
            return 0
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise SystemExit("QPS output directory exists without a manifest")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "status": "running",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "host": platform.platform(),
            "python": platform.python_version(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
            "slurm_cpu_bind": os.environ.get("SLURM_CPU_BIND"),
            "contract": contract,
            "schedule": schedule,
            "completed": [],
        }
        write_manifest(manifest_path, manifest)

    completed_by_id = {
        str(item["run_id"]): item
        for item in manifest.get("completed", [])
    }
    for block, order in enumerate(schedule):
        for position, config in enumerate(order):
            run_id = f"b{block:02d}-p{position:03d}-{config['id']}"
            output = args.output_dir / f"{run_id}.json"
            previous = completed_by_id.get(run_id)
            if previous and output.is_file() and sha256(output) == previous["sha256"]:
                print(f"REUSE {run_id}")
                continue

            command = [
                str(args.runner),
                "--method", str(config["method"]),
                "--index-path", str(args.index_path),
                "--query-path", str(args.query_path),
                "--dimension", str(args.dimension),
                "--query-start", str(args.query_start),
                "--query-count", str(args.query_count),
                "--k", str(args.k),
                "--ef-search", str(args.ef_search),
                "--warmup-queries", str(args.warmup_queries),
                "--repeats", str(args.within_process_repeats),
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
            subprocess.run(command, check=True)
            item = {
                "run_id": run_id,
                "block": block,
                "position": position,
                "config": config,
                "result": output.name,
                "sha256": sha256(output),
            }
            completed_by_id[run_id] = item
            manifest["completed"] = list(completed_by_id.values())
            write_manifest(manifest_path, manifest)
            print(f"DONE {run_id}", flush=True)

    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_manifest(manifest_path, manifest)
    print(
        f"full_beta_qps_complete runs={len(completed_by_id)} "
        f"output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
