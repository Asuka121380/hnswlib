#!/usr/bin/env python3
"""Synthetic regression test for schema-v2 radius analysis."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


FIELDS = [
    "schema_version", "run_id", "query_id", "current_node_id", "candidate_id",
    "graph_layer", "bound_status", "ef_search", "current_squared_distance",
    "threshold", "edge_length", "direction_error", "anchor_projection",
    "anchor_projection_lower", "query_direction_inner_product_upper",
    "residual_direction_inner_product_upper", "length_squared_lower",
    "cross_term_upper", "base_plus_length_lower", "approximate_squared_distance",
    "current_distance_root_upper", "direction_error_radius",
    "stored_numeric_padding", "operational_l2_padding",
    "rounding_closure_padding", "error_radius", "lower_bound",
    "current_lb", "shadow_exact_squared_distance", "would_prune",
    "current_would_prune", "oracle_would_prune", "lower_bound_valid",
    "lower_bound_violation", "false_prune", "cap_lb", "cap_would_prune",
    "blockwise_lb", "blockwise_would_prune", "repr_lb_star",
    "repr_lb_star_would_prune",
]


def write_run(root: Path, broken_closure: bool = False) -> None:
    run = root / "run"
    run.mkdir(parents=True)
    (run / "metadata.json").write_text(
        json.dumps({"shadow_schema_version": 2, "enabled_methods": ["current"]}),
        encoding="utf-8",
    )
    (run / "summary.json").write_text(
        json.dumps({
            "shadow_schema_version": 2,
            "query_count": 2,
            "mismatch_queries": 0,
            "bound_evaluated": 4,
            "bound_pruned": 2,
            "raw_prunable": 2,
            "oracle_prunable": 4,
            "exact_distance_saved": 0,
            "lower_bound_violation": 0,
            "false_prune": 0,
        }),
        encoding="utf-8",
    )
    rows = []
    for query_id in range(2):
        for candidate in range(2):
            direction = 0.20
            stored = 0.05
            operational = 0.05
            closure = 0.01
            radius = direction + stored + operational + closure
            if broken_closure and query_id == 0 and candidate == 0:
                radius += 0.5
            approximate = 2.0 + candidate
            lower = max(0.0, approximate - radius)
            threshold = 2.0
            exact = 2.5 + candidate
            row = dict.fromkeys(FIELDS, "")
            row.update({
                "schema_version": 2,
                "run_id": "synthetic-v2",
                "query_id": query_id,
                "current_node_id": 10 + query_id,
                "candidate_id": 20 + candidate,
                "graph_layer": 0,
                "bound_status": 0,
                "ef_search": 200,
                "current_squared_distance": 1.0,
                "threshold": threshold,
                "edge_length": 1.0,
                "direction_error": 0.1,
                "anchor_projection": 0.25,
                "anchor_projection_lower": 0.249999999999,
                "query_direction_inner_product_upper": 0.5,
                "residual_direction_inner_product_upper": 0.25,
                "length_squared_lower": 0.999999999999,
                "cross_term_upper": 0.5,
                "base_plus_length_lower": 2.0,
                "approximate_squared_distance": approximate,
                "current_distance_root_upper": 1.000000000001,
                "direction_error_radius": direction,
                "stored_numeric_padding": stored,
                "operational_l2_padding": operational,
                "rounding_closure_padding": closure,
                "error_radius": radius,
                "lower_bound": lower,
                "current_lb": lower,
                "shadow_exact_squared_distance": exact,
                "would_prune": int(lower > threshold),
                "current_would_prune": int(lower > threshold),
                "oracle_would_prune": int(exact > threshold),
                "lower_bound_valid": 1,
                "lower_bound_violation": 0,
                "false_prune": 0,
            })
            rows.append(row)
    with (run / "shadow_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def invoke(root: Path) -> subprocess.CompletedProcess[str]:
    script = Path(__file__).with_name("analyze_radius_components.py")
    return subprocess.run(
        [
            os.environ.get("ANALYSIS_PYTHON", sys.executable),
            str(script),
            "--run-dir", str(root / "run"),
            "--output-dir", str(root / "analysis"),
            "--expected-query-count", "2",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "valid"
        write_run(root)
        result = invoke(root)
        if result.returncode != 0:
            raise RuntimeError(result.stdout)
        quality = json.loads((root / "analysis" / "data_quality.json").read_text(encoding="utf-8"))
        if quality["status"] != "pass" or quality["analysis_valid_rows"] != 4:
            raise RuntimeError(f"unexpected quality report: {quality}")
        for name in (
            "candidate_metrics.parquet",
            "radius_component_summary.csv",
            "radius_component_by_query.csv",
            "analysis_summary.md",
            "figures/01_radius_component_median_share.png",
            "figures/02_raw_margin_vs_radius.png",
        ):
            if not (root / "analysis" / name).is_file():
                raise RuntimeError(f"missing analysis artifact: {name}")

        broken = Path(temporary) / "broken"
        write_run(broken, broken_closure=True)
        broken_result = invoke(broken)
        if broken_result.returncode == 0:
            raise RuntimeError("broken component closure unexpectedly passed")

    print("schema-v2 radius component analysis test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
