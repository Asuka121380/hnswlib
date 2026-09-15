#!/usr/bin/env python3
"""Run isolated P0.3 component microbenchmark blocks and summarize them."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRICS = (
    "strict_lut_build_ns", "fast_lut_build_ns",
    "strict_lookup_ns", "fast_lookup_ns",
    "strict_estimator_ns", "fast_estimator_ns",
    "exact_l2_ns", "checked_record_ns", "direct_record_ns",
    "state_reset_ns", "state_mark_ns",
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


def file_identity(path: Path, include_sha256: bool) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_sha256:
        result["sha256"] = sha256(path)
    return result


def require_single_cpu_affinity() -> list[int]:
    if not hasattr(os, "sched_getaffinity"):
        raise SystemExit("single-CPU affinity is required but cannot be observed")
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) != 1:
        raise SystemExit(
            "single-CPU affinity is required; allowed CPUs are " +
            ",".join(str(cpu) for cpu in allowed))
    return allowed


def cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(
                encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def resource_observation(
    resource_profile: str, allowed_cpus: list[int], requested_node: str,
) -> dict[str, object]:
    return {
        "resource_profile": resource_profile,
        "claim_scope": ("formal" if resource_profile == "formal-exclusive"
                        else "exploratory"),
        "exclusive": resource_profile == "formal-exclusive",
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu_model": cpu_model(),
        "python": platform.python_version(),
        "requested_node": requested_node,
        "allowed_cpus": allowed_cpus,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_job_qos": os.environ.get("SLURM_JOB_QOS"),
        "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
        "slurm_cpu_bind": os.environ.get("SLURM_CPU_BIND"),
        "slurm_job_account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "numa_policy": "slurm-mem-bind-local",
        "cache_mode": "hot-kernel-working-set-after-explicit-warmup",
        "observed_utc": datetime.now(timezone.utc).isoformat(),
    }


def confidence_interval_95(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean
    # Two-sided Student-t critical values for the small block counts used here.
    critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,
                5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
                9: 2.262}.get(len(values) - 1, 1.96)
    half = critical * statistics.stdev(values) / math.sqrt(len(values))
    return mean - half, mean + half


def summarize(raw: list[dict[str, Any]]) -> dict[str, object]:
    first = raw[0]
    result: dict[str, object] = {
        "schema_version": 1,
        "kernel": first["kernel"],
        "dimension": first["dimension"],
        "pq_m": first["pq_m"],
        "pq_ksub": first["pq_ksub"],
        "iterations_per_block": first["iterations"],
        "warmup_iterations_per_block": first["warmup_iterations"],
        "working_set": first["working_set"],
        "blocks": len(raw),
        "aggregation": "median; 95% CI is Student-t interval around block mean",
        "distributions": {},
    }
    distributions = result["distributions"]
    assert isinstance(distributions, dict)
    for metric in METRICS:
        values = [float(item[metric]) for item in raw]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise SystemExit(f"invalid component measurement: {metric}")
        low, high = confidence_interval_95(values)
        median = statistics.median(values)
        result[metric] = median
        distributions[metric] = {
            "values": values,
            "median": median,
            "mean": statistics.fmean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "ci95_low": low,
            "ci95_high": high,
        }
    result["state_bytes_per_query"] = first["state_bytes_per_query"]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--cmake-cache", required=True, type=Path)
    parser.add_argument("--sidecar-path", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--resource-profile", required=True,
                        choices=("formal-exclusive", "exploratory-shared"))
    parser.add_argument("--requested-node", required=True)
    parser.add_argument("--experiment-commit", required=True)
    parser.add_argument("--git-branch", required=True)
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--warmup-iterations", type=int, default=10000)
    parser.add_argument("--working-set", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=7)
    args = parser.parse_args()
    if args.blocks < 5 or args.iterations <= 0 or args.warmup_iterations <= 0:
        parser.error("blocks must be >= 5 and iteration counts must be positive")
    if args.working_set <= 0:
        parser.error("working-set must be positive")
    return args


def main() -> int:
    args = parse_args()
    for path in (args.runner, args.cmake_cache, args.sidecar_path):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    allowed_cpus = require_single_cpu_affinity()
    observed_host = platform.node().split(".", 1)[0]
    if observed_host != args.requested_node:
        raise SystemExit(
            f"node mismatch: requested {args.requested_node}, got {observed_host}")
    if args.run_root.exists():
        allowed_preexisting = {"logs", "submission.env"}
        unexpected = {
            path.name for path in args.run_root.iterdir()
            if path.name not in allowed_preexisting
        }
        if unexpected:
            raise SystemExit(
                "run root contains experiment artifacts: " +
                ",".join(sorted(unexpected)))

    contract = {
        "schema_version": 1,
        "experiment": "p0.3-component-microbenchmark",
        "experiment_commit": args.experiment_commit,
        "git_branch": args.git_branch,
        "runner": file_identity(args.runner, include_sha256=True),
        "cmake_cache": file_identity(args.cmake_cache, include_sha256=True),
        "sidecar": file_identity(args.sidecar_path, include_sha256=False),
        "resource_profile": args.resource_profile,
        "requested_node": args.requested_node,
        "iterations": args.iterations,
        "warmup_iterations": args.warmup_iterations,
        "working_set": args.working_set,
        "blocks": args.blocks,
        "process_isolation": "one independent process per block",
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "resource_observations": [resource_observation(
            args.resource_profile, allowed_cpus, args.requested_node)],
        "completed": [],
    }
    raw_dir = args.run_root / "raw"
    raw_dir.mkdir(parents=True)
    manifest_path = args.run_root / "manifest.json"
    atomic_json(manifest_path, manifest)

    raw: list[dict[str, Any]] = []
    completed: list[dict[str, object]] = []
    for block in range(args.blocks):
        output = raw_dir / f"block-{block:02d}.json"
        command = [
            str(args.runner),
            "--sidecar-path", str(args.sidecar_path),
            "--output", str(output),
            "--iterations", str(args.iterations),
            "--warmup-iterations", str(args.warmup_iterations),
            "--working-set", str(args.working_set),
        ]
        print(f"START block={block}", flush=True)
        subprocess.run(command, check=True)
        data = json.loads(output.read_text(encoding="utf-8"))
        raw.append(data)
        completed.append({
            "block": block,
            "output": str(output.relative_to(args.run_root)),
            "sha256": sha256(output),
            "command": command,
        })
        manifest["completed"] = completed
        atomic_json(manifest_path, manifest)
        print(f"DONE block={block}", flush=True)

    summary_path = args.run_root / "component_summary.json"
    atomic_json(summary_path, summarize(raw))
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["summary"] = str(summary_path.relative_to(args.run_root))
    manifest["summary_sha256"] = sha256(summary_path)
    atomic_json(manifest_path, manifest)
    (args.run_root / "COMPLETE").write_text(
        f"{args.experiment_commit}\n", encoding="utf-8")
    print(f"component_microbenchmark_complete output={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
