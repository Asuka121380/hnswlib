#!/usr/bin/env python3
"""Summarize Stage 3 held-out projection distortion per coarse byte."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from operational_dataset import read_json, write_json


def summarize(selection_report: dict[str, Any]) -> dict[str, Any]:
    summary = selection_report["summary"]
    baseline = summary["flat_32__frozen_flat"]["mean_full_projection_abs_q99"]
    rows = []
    for candidate, metrics in sorted(summary.items()):
        if candidate == "flat_32__frozen_flat":
            coarse_bytes, residual_bytes = 32, 0
        else:
            layout = candidate.split("__", 1)[0].removeprefix("progressive_")
            coarse_bytes, residual_bytes = [int(value) for value in layout.split("_")]
        rows.append(
            {
                "candidate": candidate,
                "coarse_bytes": coarse_bytes,
                "residual_bytes": residual_bytes,
                "coarse_q99": metrics["mean_coarse_projection_abs_q99"],
                "full_q99": metrics["mean_full_projection_abs_q99"],
                "full_q99_ratio_to_frozen_flat": metrics["mean_full_projection_abs_q99"] / baseline,
                "coarse_q99_reduction_per_residual_byte": (
                    (metrics["mean_coarse_projection_abs_q99"] - metrics["mean_full_projection_abs_q99"])
                    / residual_bytes
                    if residual_bytes else 0.0
                ),
            }
        )
    return {
        "format": "eq_rcp_stage3_operational_rate_distortion",
        "version": 1,
        "baseline": "flat_32__frozen_flat",
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize(read_json(args.selection_report))
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
