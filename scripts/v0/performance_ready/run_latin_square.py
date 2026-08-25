#!/usr/bin/env python3
"""Run V0 methods as isolated processes in a frozen balanced block order."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path


METHODS = ("baseline", "approx-no-retry", "approx-retry")
BALANCED_ORDERS = (
    METHODS,
    ("approx-no-retry", "approx-retry", "baseline"),
    ("approx-retry", "baseline", "approx-no-retry"),
    tuple(reversed(METHODS)),
    ("baseline", "approx-retry", "approx-no-retry"),
    ("approx-no-retry", "baseline", "approx-retry"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--index-path", required=True, type=Path)
    parser.add_argument("--sidecar-path", type=Path)
    parser.add_argument("--query-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dimension", required=True, type=int)
    parser.add_argument("--query-count", required=True, type=int)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", required=True, type=int)
    parser.add_argument("--beta-no-retry", type=float)
    parser.add_argument("--beta-retry", type=float)
    parser.add_argument("--prefetch", choices=("legacy", "gate"),
                        required=True)
    parser.add_argument("--warmup-queries", type=int, default=100)
    parser.add_argument("--within-process-repeats", type=int, default=5)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cpu", type=int, default=-1)
    parser.add_argument("--comparison-mode", required=True,
                        choices=("fixed-ef", "matched-recall"))
    parser.add_argument("--cache-mode", required=True,
                        choices=("warm", "cold-start"))
    parser.add_argument("--numa-node", required=True)
    parser.add_argument("--power-policy", required=True)
    parser.add_argument("--cpu-frequency-policy", required=True)
    parser.add_argument("--cmake-cache", type=Path)
    parser.add_argument("--aa-baseline", action="store_true",
                        help="run three indistinguishable baseline instances")
    args = parser.parse_args()
    if args.blocks < 5 or args.within_process_repeats < 5:
        parser.error("blocks and within-process-repeats must both be >= 5")
    if not args.aa_baseline and (
            args.sidecar_path is None or args.beta_no_retry is None or
            args.beta_retry is None):
        parser.error("active schedule requires sidecar and both beta values")
    return args


def main() -> int:
    args = parse_args()
    required_paths = [args.runner, args.index_path, args.query_path]
    if args.sidecar_path is not None:
        required_paths.append(args.sidecar_path)
    for path in required_paths:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    orders = list(BALANCED_ORDERS)
    random.Random(args.seed).shuffle(orders)
    schedule = [orders[i % len(orders)] for i in range(args.blocks)]
    if args.aa_baseline:
        schedule = [("baseline", "baseline", "baseline")
                    for _ in range(args.blocks)]

    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kernel": "raw_fast_v1",
        "order_contract": "seeded-balanced-six-order-blocks",
        "seed": args.seed,
        "schedule": schedule,
        "runner_sha256": sha256(args.runner),
        "cmake_cache_sha256": (
            sha256(args.cmake_cache) if args.cmake_cache else None),
        "host": platform.platform(),
        "python": platform.python_version(),
        "single_thread_query": True,
        "cpu": args.cpu,
        "prefetch": args.prefetch,
        "comparison_mode": args.comparison_mode,
        "cache_mode": args.cache_mode,
        "numa_node": args.numa_node,
        "power_policy": args.power_policy,
        "cpu_frequency_policy": args.cpu_frequency_policy,
        "ef_search": args.ef_search,
        "query_start": args.query_start,
        "query_count": args.query_count,
        "blocks": [],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    completed: list[dict[str, object]] = []
    for block_index, order in enumerate(schedule):
        for position, method in enumerate(order):
            instance = f"b{block_index:02d}-p{position}-{method}"
            if args.aa_baseline:
                instance += f"-{position}"
            output = args.output_dir / f"{instance}.json"
            command = [
                str(args.runner), "--method", method,
                "--index-path", str(args.index_path),
                "--query-path", str(args.query_path),
                "--dimension", str(args.dimension),
                "--query-start", str(args.query_start),
                "--query-count", str(args.query_count),
                "--k", str(args.k),
                "--ef-search", str(args.ef_search),
                "--warmup-queries", str(args.warmup_queries),
                "--repeats", str(args.within_process_repeats),
                "--prefetch", args.prefetch,
                "--cpu", str(args.cpu),
                "--output", str(output),
            ]
            if method != "baseline":
                beta = (args.beta_retry if method == "approx-retry"
                        else args.beta_no_retry)
                command.extend(("--sidecar-path", str(args.sidecar_path),
                                "--beta", str(beta)))
            subprocess.run(command, check=True)
            completed.append({
                "block": block_index,
                "position": position,
                "method": method,
                "result": output.name,
                "result_sha256": sha256(output),
            })
            manifest["blocks"] = completed
            manifest_path.write_text(
                json.dumps(manifest, indent=2), encoding="utf-8")

    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
