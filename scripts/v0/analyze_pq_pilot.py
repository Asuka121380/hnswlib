#!/usr/bin/env python3
"""Aggregate reproducible M={8,16,32} V0 PQ pilot metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


FIELDS = [
    "M_pq",
    "nbits",
    "codebook_bytes",
    "per_vector_code_bytes",
    "per_query_lut_bytes_float32",
    "mean_error",
    "median_error",
    "p90_error",
    "p95_error",
    "p99_error",
    "max_error",
    "reconstruction_mse",
    "training_seconds",
    "validation_seconds",
]


def _atomic_text(path: Path, text: str) -> None:
    partial = Path(str(path) + ".partial")
    if path.exists() or partial.exists():
        raise RuntimeError(f"refusing to overwrite output: {path}")
    with partial.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


def _load(paths: list[Path], expected: list[int]) -> list[dict[str, Any]]:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(row.get("status") != "trained" for row in rows):
        raise RuntimeError("all input metrics must have status=trained")
    by_m = {int(row["M_pq"]): row for row in rows}
    if sorted(by_m) != sorted(expected) or len(by_m) != len(rows):
        raise RuntimeError(f"metrics must contain exactly M={sorted(expected)}")
    for field in ("dimension", "nbits", "source_manifest_sha256", "source_directions_sha256", "split_seed"):
        if len({str(row[field]) for row in rows}) != 1:
            raise RuntimeError(f"pilot metrics disagree on {field}")
    return [by_m[value] for value in sorted(expected)]


def _plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plot generation requires matplotlib; use --skip-plots locally") from exc
    widths = [row["M_pq"] for row in rows]
    p99 = [row["p99_error"] for row in rows]
    code_bytes = [row["per_vector_code_bytes"] for row in rows]
    plt.figure()
    plt.plot(code_bytes, p99, marker="o")
    plt.xlabel("PQ code bytes per directed edge")
    plt.ylabel("P99 direction reconstruction error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "error_vs_code_size.png", dpi=160)
    plt.close()

    plt.figure()
    for quantile in ("median_error", "p90_error", "p95_error", "p99_error"):
        plt.plot(widths, [row[quantile] for row in rows], marker="o", label=quantile)
    plt.xlabel("M_pq")
    plt.ylabel("Direction reconstruction error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "reconstruction_error_quantiles.png", dpi=160)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, action="append", required=True)
    parser.add_argument("--expected-M", default="8,16,32")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-p99-error", type=float)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    try:
        expected = [int(value) for value in args.expected_M.split(",")]
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise RuntimeError(f"output directory is not empty: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        rows = _load(args.metrics, expected)
        eligible = (
            [row for row in rows if row["p99_error"] <= args.max_p99_error]
            if args.max_p99_error is not None
            else []
        )
        recommended = min(eligible, key=lambda row: row["M_pq"])["M_pq"] if eligible else None

        csv_path = args.output_dir / "pq_pilot_summary.csv"
        partial_csv = Path(str(csv_path) + ".partial")
        with partial_csv.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        partial_csv.replace(csv_path)

        aggregate = {
            "format": "hnswlib_v0_pq_pilot_summary",
            "format_version": 1,
            "dimension": rows[0]["dimension"],
            "nbits": rows[0]["nbits"],
            "source_manifest_sha256": rows[0]["source_manifest_sha256"],
            "source_directions_sha256": rows[0]["source_directions_sha256"],
            "selection_rule": (
                f"smallest M with p99_error <= {args.max_p99_error}"
                if args.max_p99_error is not None
                else "manual; no numerical acceptance threshold supplied"
            ),
            "recommended_M_pq": recommended,
            "runs": rows,
        }
        _atomic_text(
            args.output_dir / "pq_pilot_metrics.json",
            json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        )
        report = [
            "# V0 PQ pilot report",
            "",
            f"- Dimension: {rows[0]['dimension']}",
            f"- PQ bits per subquantizer: {rows[0]['nbits']}",
            f"- Candidate widths: {', '.join(str(value) for value in expected)}",
            f"- Selection rule: {aggregate['selection_rule']}",
            f"- Recommended M: {recommended if recommended is not None else 'not selected'}",
            "",
            "| M | code bytes/edge | LUT bytes/query | mean error | P99 error | max error |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        report.extend(
            f"| {row['M_pq']} | {row['per_vector_code_bytes']} | "
            f"{row['per_query_lut_bytes_float32']} | {row['mean_error']:.8g} | "
            f"{row['p99_error']:.8g} | {row['max_error']:.8g} |"
            for row in rows
        )
        report.extend(
            [
                "",
                "The final M must be selected from the measured quality/storage trade-off. "
                "This report does not enable query-time pruning.",
                "",
            ]
        )
        _atomic_text(args.output_dir / "report.md", "\n".join(report))
        if not args.skip_plots:
            _plots(rows, args.output_dir)
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
