#!/usr/bin/env python3
"""Project full-run trace time and storage from a completed pilot shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", required=True, type=Path)
    parser.add_argument("--target-queries", required=True, type=int)
    parser.add_argument("--budget-gb", type=float, default=60.0)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query_path = args.pilot_dir / "query_stats.csv"
    dco_path = args.pilot_dir / "dco_trace.csv"
    edge_path = args.pilot_dir / "edge_directions.csv"
    query = pd.read_csv(query_path)
    if query.empty:
        raise RuntimeError("Pilot contains no query rows")
    pilot_bytes = sum(path.stat().st_size for path in (query_path, dco_path, edge_path))
    scale = args.target_queries / len(query)
    projected_bytes = int(pilot_bytes * scale)
    projected_trace_ns = float(query["trace_query_latency_ns"].sum()) * scale
    report = {
        "pilot_queries": len(query),
        "target_queries": args.target_queries,
        "mean_n_dist": float(query["n_dist"].mean()),
        "pilot_bytes": pilot_bytes,
        "projected_bytes": projected_bytes,
        "projected_gib": projected_bytes / 2**30,
        "projected_trace_hours_excluding_queue_and_startup": projected_trace_ns / 1e9 / 3600,
        "budget_gb": args.budget_gb,
        "within_budget": projected_bytes <= args.budget_gb * 10**9,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["within_budget"]:
        raise SystemExit("Projected trace exceeds storage budget; increase DCO sampling modulus")
    print("trace_capacity_ok")


if __name__ == "__main__":
    main()
