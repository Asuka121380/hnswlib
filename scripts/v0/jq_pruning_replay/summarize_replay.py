#!/usr/bin/env python3
"""Aggregate fixed-seed JQ replay summaries without cherry-picking."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
from typing import Any


EXPECTED_SEEDS = (1234, 17, 42)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    partial = Path(str(path) + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    with partial.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


def summarize(run_dirs: list[Path]) -> dict[str, Any]:
    summaries = []
    for directory in run_dirs:
        complete = directory / "complete.json"
        summary_path = directory / "summary.json"
        if not complete.is_file() or not summary_path.is_file():
            raise ValueError(f"incomplete replay directory: {directory}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "valid" or summary.get("lower_bound_violation_count") != 0:
            raise ValueError(f"invalid replay summary: {directory}")
        summaries.append(summary)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for summary in summaries:
        grouped.setdefault(int(summary["subspaces"]), []).append(summary)
    configurations = []
    for subspaces, values in sorted(grouped.items()):
        seeds = tuple(sorted(int(value["rotation_seed"]) for value in values))
        if seeds != tuple(sorted(EXPECTED_SEEDS)):
            raise ValueError(
                f"M={subspaces} has seeds {seeds}, expected {tuple(sorted(EXPECTED_SEEDS))}"
            )
        rates = [float(value["theoretical_prune_rate"]) for value in values]
        configurations.append(
            {
                "subspaces": subspaces,
                "code_size_bytes": subspaces,
                "iso_budget_primary": subspaces == 32,
                "seeds": list(EXPECTED_SEEDS),
                "seed_results": values,
                "rate_median": statistics.median(rates),
                "rate_min": min(rates),
                "rate_max": max(rates),
            }
        )
    return {
        "format": "hnswlib_jq_pruning_replay_aggregate",
        "format_version": 1,
        "primary_metric": "event_weighted_theoretical_prune_rate",
        "configurations": configurations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize([path.resolve() for path in args.run_dir])
    _atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
