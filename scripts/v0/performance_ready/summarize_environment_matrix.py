#!/usr/bin/env python3
"""Validate and summarize the platform x build-variant environment matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ACTIVE_ID = "approx-no-retry-beta1p45-legacy-ef500"
BASELINE_ID = "baseline-ef435"
VARIANTS = {"portable", "native"}


def parse_cell(value: str) -> tuple[str, str, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            "cell must be PLATFORM=portable|native=RUN_ROOT")
    platform, variant, root = parts
    if variant not in VARIANTS:
        raise argparse.ArgumentTypeError(f"invalid build variant: {variant}")
    return platform, variant, Path(root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", action="append", required=True,
                        type=parse_cell)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def read_qps(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    path = root / "qps" / "qps_summary.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        by_id = {row["configuration_id"]: row for row in csv.DictReader(handle)}
    missing = {ACTIVE_ID, BASELINE_ID} - by_id.keys()
    if missing:
        raise ValueError(f"{path} lacks configurations: {sorted(missing)}")
    return by_id[ACTIVE_ID], by_id[BASELINE_ID]


def load_cell(platform: str, variant: str, root: Path) -> dict[str, Any]:
    if not (root / "COMPLETE").is_file():
        raise ValueError(f"cell is incomplete: {root}")
    cell_values = {}
    for line in (root / "cell.env").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            cell_values[key] = value
    if cell_values.get("platform") != platform:
        raise ValueError(f"platform label mismatch in {root / 'cell.env'}")
    if cell_values.get("build_variant") != variant:
        raise ValueError(f"build variant mismatch in {root / 'cell.env'}")
    active, baseline = read_qps(root)
    component_path = root / "component" / "component_summary.json"
    component = json.loads(component_path.read_text(encoding="utf-8"))
    component_manifest = json.loads(
        (root / "component" / "manifest.json").read_text(encoding="utf-8"))
    qps_commit = active["experiment_commit"]
    if baseline["experiment_commit"] != qps_commit:
        raise ValueError(f"commit changed inside cell: {root}")
    component_contract = component_manifest["contract"]
    if component_contract["experiment_commit"] != qps_commit:
        raise ValueError(f"QPS/component commit mismatch in cell: {root}")
    if component_contract["resource_profile"] != active["resource_profile"]:
        raise ValueError(f"QPS/component resource profile mismatch in cell: {root}")
    if cell_values.get("git_commit") != qps_commit:
        raise ValueError(f"cell/QPS commit mismatch in cell: {root}")
    speedup = float(active["qps_speedup_mean"])
    exact = float(component["exact_l2_ns"])
    estimator = float(component["fast_estimator_ns"])
    lut = float(component["fast_lut_build_ns"])
    return {
        "platform": platform,
        "build_variant": variant,
        "run_root": str(root.resolve()),
        "experiment_commit": qps_commit,
        "resource_profile": active["resource_profile"],
        "baseline_qps": float(baseline["qps_mean"]),
        "active_qps": float(active["qps_mean"]),
        "speedup_ratio": speedup,
        "speedup_percent": (speedup - 1.0) * 100.0,
        "speedup_ci_low": float(active["qps_speedup_ci_low"]),
        "speedup_ci_high": float(active["qps_speedup_ci_high"]),
        "fast_lut_build_ns": lut,
        "fast_estimator_ns": estimator,
        "exact_l2_ns": exact,
        "estimator_to_exact_ratio": estimator / exact,
        "checksum_consistent": active["checksum_consistent"],
    }


def main() -> int:
    args = parse_args()
    if len(args.cell) != 4:
        raise SystemExit("exactly four --cell arguments are required")
    keys = {(platform, variant) for platform, variant, _ in args.cell}
    platform_order = list(dict.fromkeys(platform for platform, _, _ in args.cell))
    platforms = set(platform_order)
    expected = {(platform, variant) for platform in platform_order
                for variant in VARIANTS}
    if len(platforms) != 2 or keys != expected:
        raise SystemExit(
            "cells must form two platforms x {portable,native}")
    rows = [load_cell(*cell) for cell in args.cell]
    commits = {row["experiment_commit"] for row in rows}
    if len(commits) != 1:
        raise SystemExit(f"cells use different commits: {sorted(commits)}")
    profiles = {row["resource_profile"] for row in rows}
    if len(profiles) != 1:
        raise SystemExit(f"cells use different resource profiles: {sorted(profiles)}")
    if any(str(row["checksum_consistent"]).lower() != "true" for row in rows):
        raise SystemExit("one or more active configurations have inconsistent checksums")

    platform_position = {
        platform: position for position, platform in enumerate(platform_order)}
    rows.sort(key=lambda row: (
        platform_position[row["platform"]], row["build_variant"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_key = {(row["platform"], row["build_variant"]): row for row in rows}
    build_effects = {
        platform: (
            by_key[(platform, "native")]["speedup_percent"] -
            by_key[(platform, "portable")]["speedup_percent"]
        ) for platform in platform_order
    }
    platform_effects = {
        variant: (
            by_key[(platform_order[1], variant)]["speedup_percent"] -
            by_key[(platform_order[0], variant)]["speedup_percent"]
        ) for variant in sorted(VARIANTS)
    }
    report = {
        "schema_version": 1,
        "status": "complete",
        "experiment_commit": next(iter(commits)),
        "resource_profile": next(iter(profiles)),
        "platform_order_for_effect": platform_order,
        "build_effect_native_minus_portable_percentage_points": build_effects,
        "platform_effect_second_minus_first_percentage_points": platform_effects,
        "cells": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"environment_matrix_complete cells=4 output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
