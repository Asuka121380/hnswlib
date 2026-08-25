#!/usr/bin/env python3
"""Evaluate the preregistered 25%-headroom component cost gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def number(value: dict[str, object], key: str) -> float:
    result = float(value[key])
    if result < 0:
        raise ValueError(f"{key} must be non-negative")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--microbenchmark", required=True, type=Path)
    parser.add_argument("--metrics-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    micro = load_object(args.microbenchmark)
    metrics = load_object(args.metrics_summary)

    queries = number(metrics, "query_count")
    eligible = number(metrics, "approx_eligible_first_visits")
    first_pruned = number(metrics, "approx_first_pruned")
    edge_scans = number(metrics, "edge_scans")
    net_saved = number(metrics, "exact_distance_saved")
    components = {
        "lut_build": queries * number(micro, "fast_lut_build_ns"),
        "raw_estimator": eligible * number(micro, "fast_estimator_ns"),
        "retry_state_initialization":
            queries * number(micro, "state_reset_ns"),
        "retry_state_mark": first_pruned * number(micro, "state_mark_ns"),
        "direct_metadata_access":
            edge_scans * number(micro, "direct_record_ns"),
    }
    left_ns = sum(components.values())
    saved_exact_ns = net_saved * number(micro, "exact_l2_ns")
    allowed_ns = 0.75 * saved_exact_ns
    passed = left_ns <= allowed_ns
    result = {
        "schema_version": 1,
        "kernel": "raw_fast_v1",
        "status": "PASS" if passed else "FAIL",
        "safety_headroom_fraction": 0.25,
        "component_cost_ns": components,
        "estimated_overhead_ns": left_ns,
        "estimated_saved_exact_ns": saved_exact_ns,
        "allowed_overhead_ns": allowed_ns,
        "overhead_to_allowed_ratio": (
            left_ns / allowed_ns if allowed_ns else None),
        "inputs": {
            "query_count": queries,
            "eligible_edges": eligible,
            "first_pruned": first_pruned,
            "edge_scans": edge_scans,
            "net_exact_distances_saved": net_saved,
        },
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"break-even gate {result['status']}: {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
