#!/usr/bin/env python3
"""Select baseline ef values by recall and freeze a matched-recall QPS contract."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def canonical_beta(value: object) -> str:
    return format(float(value), ".15g")


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    required = {"ef_search", "beta", "mode", "baseline_recall", "v0_recall"}
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise SystemExit(
                f"quality row {index} is missing: {', '.join(sorted(missing))}")
    return rows


def parse_candidate(value: str) -> tuple[str, str, str, int]:
    parts = value.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "candidate must be BETA:MODE:PREFETCH:EF_SEARCH")
    beta, mode, prefetch, ef = parts
    if mode not in ("approx-no-retry", "approx-retry"):
        raise argparse.ArgumentTypeError(f"invalid candidate mode: {mode}")
    if prefetch not in ("legacy", "gate"):
        raise argparse.ArgumentTypeError(f"invalid candidate prefetch: {prefetch}")
    try:
        ef_search = int(ef)
        beta = canonical_beta(beta)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return beta, mode, prefetch, ef_search


def baseline_curve(rows: list[dict[str, str]]) -> dict[int, float]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["ef_search"]), []).append(
            float(row["baseline_recall"]))
    curve: dict[int, float] = {}
    for ef_search, values in grouped.items():
        if max(values) - min(values) > 1e-12:
            raise SystemExit(
                f"baseline recall is inconsistent at ef={ef_search}: "
                f"{min(values)}..{max(values)}")
        curve[ef_search] = values[0]
    return curve


def choose_baseline(
    curve: dict[int, float], target: float, tolerance: float,
) -> tuple[int, float, float]:
    candidates = [
        (recall - target, ef_search, recall)
        for ef_search, recall in curve.items()
        if recall >= target and recall - target <= tolerance
    ]
    if not candidates:
        nearest = min(
            ((abs(recall - target), ef_search, recall)
             for ef_search, recall in curve.items()),
            default=None,
        )
        detail = "no baseline recall observations" if nearest is None else (
            f"nearest is ef={nearest[1]} recall={nearest[2]:.6f} "
            f"gap={nearest[2] - target:+.6f}")
        raise SystemExit(
            f"no conservative baseline match within {tolerance:.6f} for "
            f"target {target:.6f}; {detail}")
    gap, ef_search, recall = min(candidates)
    return ef_search, recall, gap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-summary", required=True, type=Path,
                        action="append")
    parser.add_argument("--template", required=True, type=Path,
                        help="base QPS JSON; matrix/cases fields are replaced")
    parser.add_argument("--candidate", required=True, action="append",
                        type=parse_candidate)
    parser.add_argument("--tolerance", type=float, default=0.001)
    parser.add_argument("--output-config", required=True, type=Path)
    parser.add_argument("--selection-output", required=True, type=Path)
    parser.add_argument("--experiment-name", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        raise SystemExit("tolerance must be finite and >= 0")
    rows = load_rows(args.quality_summary)
    curve = baseline_curve(rows)
    selections: list[dict[str, Any]] = []
    for beta, mode, prefetch, ef_search in args.candidate:
        matches = [
            row for row in rows
            if canonical_beta(row["beta"]) == beta and
            row["mode"] == mode and int(row["ef_search"]) == ef_search
        ]
        if len(matches) != 1:
            raise SystemExit(
                f"expected one quality row for {beta}:{mode}:ef{ef_search}, "
                f"found {len(matches)}")
        target = float(matches[0]["v0_recall"])
        baseline_ef, baseline_recall, gap = choose_baseline(
            curve, target, args.tolerance)
        selections.append({
            "beta": beta, "mode": mode, "prefetch": prefetch,
            "edgepq_ef_search": ef_search, "edgepq_recall": target,
            "baseline_ef_search": baseline_ef,
            "baseline_recall": baseline_recall, "recall_gap": gap,
        })

    template = json.loads(args.template.read_text(encoding="utf-8"))
    for key in ("betas", "modes", "prefetches", "cases"):
        template.pop(key, None)
    template["experiment_name"] = args.experiment_name
    template["include_baseline"] = False
    cases: list[dict[str, Any]] = []
    baseline_ids: dict[int, str] = {}
    for selection in selections:
        ef_search = selection["baseline_ef_search"]
        baseline_id = baseline_ids.setdefault(
            ef_search, f"baseline-ef{ef_search}")
        if not any(case.get("id") == baseline_id for case in cases):
            cases.append({
                "id": baseline_id, "method": "baseline",
                "ef_search": ef_search,
            })
        beta_label = selection["beta"].replace(".", "p")
        active_id = (
            f"{selection['mode']}-beta{beta_label}-{selection['prefetch']}"
            f"-ef{selection['edgepq_ef_search']}")
        cases.append({
            "id": active_id,
            "method": selection["mode"],
            "beta": float(selection["beta"]),
            "prefetch": selection["prefetch"],
            "ef_search": selection["edgepq_ef_search"],
            "baseline_id": baseline_id,
        })
        selection["configuration_id"] = active_id
        selection["baseline_id"] = baseline_id
    template["cases"] = cases

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(
        json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selection_report = {
        "schema_version": 1, "status": "complete",
        "matching_policy": "baseline_recall_at_least_edgepq",
        "tolerance": args.tolerance,
        "quality_summaries": [str(path.resolve())
                              for path in args.quality_summary],
        "qps_config": str(args.output_config.resolve()),
        "selections": selections,
    }
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)
    args.selection_output.write_text(
        json.dumps(selection_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    for selection in selections:
        print(
            f"MATCH {selection['configuration_id']} recall="
            f"{selection['edgepq_recall']:.6f} -> "
            f"{selection['baseline_id']} recall="
            f"{selection['baseline_recall']:.6f} "
            f"gap={selection['recall_gap']:+.6f}")
    print(f"matched_recall_config_complete output={args.output_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
