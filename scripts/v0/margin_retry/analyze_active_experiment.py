#!/usr/bin/env python3
"""Aggregate fixed-ef active raw-margin/no-retry/retry runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


EXPERIMENT_MODES = {"prune", "approx-no-retry", "approx-retry"}
QUERY_COLUMNS = {
    "query_id",
    "baseline_recall_at_k",
    "v0_recall_at_k",
    "results_equal",
    "ground_truth_hits_lost",
    "result_overlap_at_k",
    "recall_loss",
    "baseline_exact_distance_computed",
    "exact_distance_computed",
    "approx_eligible_first_visits",
    "approx_first_pruned",
    "approx_retry_exact_distance",
}


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_query_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = QUERY_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def beta_provenance(beta: float, mode: str) -> str:
    if mode == "prune":
        return "not_applicable_strict_safe_v0"
    if math.isclose(beta, 1.40, rel_tol=0.0, abs_tol=1e-12):
        return "stage1_fail_active_included"
    return "stage1_pass"


def load_run(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    metadata = load_json(path / "metadata.json")
    summary = load_json(path / "summary.json")
    complete = load_json(path / "complete.json")
    raw_rows = load_query_rows(path / "query_metrics.csv")
    mode = str(metadata.get("mode", ""))
    if mode not in EXPERIMENT_MODES:
        raise ValueError(f"{path}: unsupported experiment mode {mode!r}")
    if summary.get("status") != "valid" or complete.get("status") != "complete":
        raise ValueError(f"{path}: run is not valid and complete")
    beta = float(metadata["approx_beta"])
    ef_search = int(metadata["ef_search"])
    query_count = int(summary["query_count"])
    if len(raw_rows) != query_count:
        raise ValueError(f"{path}: query row count mismatch")

    baseline_dco = sum(int(row["baseline_exact_distance_computed"]) for row in raw_rows)
    active_dco = sum(int(row["exact_distance_computed"]) for row in raw_rows)
    if baseline_dco != int(summary["baseline_exact_distance_computed"]):
        raise ValueError(f"{path}: baseline DCO does not reaggregate")
    if active_dco != int(summary["exact_distance_computed"]):
        raise ValueError(f"{path}: active DCO does not reaggregate")

    aggregate: Dict[str, Any] = {
        "run_dir": str(path.resolve()),
        "mode": mode,
        "beta": beta,
        "beta_provenance": beta_provenance(beta, mode),
        "ef_search": ef_search,
        "query_count": query_count,
        "baseline_recall_at_k": float(summary["mean_baseline_recall_at_k"]),
        "active_recall_at_k": float(summary["mean_v0_recall_at_k"]),
        "recall_loss": float(summary["mean_recall_loss"]),
        "mismatch_query_rate": ratio(int(summary["mismatch_queries"]), query_count),
        "recall_loss_query_rate": ratio(int(summary["recall_loss_queries"]), query_count),
        "catastrophic_query_rate": ratio(
            int(summary["catastrophic_recall_loss_queries"]), query_count
        ),
        "baseline_exact_dco": baseline_dco,
        "active_exact_dco": active_dco,
        "baseline_relative_dco_reduction": ratio(baseline_dco - active_dco, baseline_dco),
        "first_prune_rate": ratio(
            int(summary["approx_first_pruned"]),
            int(summary["approx_eligible_first_visits"]),
        ),
        "retry_repayment_rate": ratio(
            int(summary["approx_retry_exact_distance"]),
            int(summary["approx_first_pruned"]),
        ),
        "state_machine_net_prune_rate": ratio(
            int(summary["exact_distance_saved"]),
            int(summary["approx_eligible_first_visits"]),
        ),
        "retry_inserted_candidate_rate": ratio(
            int(summary["approx_retry_inserted_candidate"]),
            int(summary["approx_retry_exact_distance"]),
        ),
        "retry_inserted_result_rate": ratio(
            int(summary["approx_retry_inserted_result"]),
            int(summary["approx_retry_exact_distance"]),
        ),
        "recall_recovery": "",
        "recall_recovery_not_needed": False,
        "numeric_gate_pass": False,
        "gate_classification": "NOT_RETRY_RUN",
    }
    enriched: List[Dict[str, Any]] = []
    for row in raw_rows:
        enriched.append(
            {
                "run_dir": aggregate["run_dir"],
                "mode": mode,
                "beta": beta,
                "beta_provenance": aggregate["beta_provenance"],
                "ef_search": ef_search,
                **row,
            }
        )
    return aggregate, enriched


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-query-count", type=int, default=1000)
    parser.add_argument("--expected-betas", default="1.40,1.45,1.55")
    parser.add_argument("--expected-ef-search", default="200")
    parser.add_argument("--require-strict", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.force:
        raise SystemExit(f"output directory is not empty; pass --force: {output}")
    output.mkdir(parents=True, exist_ok=True)

    aggregates: List[Dict[str, Any]] = []
    query_rows: List[Dict[str, Any]] = []
    keys: Dict[Tuple[int, float, str], Dict[str, Any]] = {}
    for run_dir in args.run_dir:
        aggregate, per_query = load_run(run_dir.resolve())
        key = (int(aggregate["ef_search"]), float(aggregate["beta"]), str(aggregate["mode"]))
        if key in keys:
            raise SystemExit(f"duplicate active run key: {key}")
        keys[key] = aggregate
        aggregates.append(aggregate)
        query_rows.extend(per_query)

    expected_betas = sorted(float(value) for value in args.expected_betas.split(","))
    expected_ef = sorted(int(value) for value in args.expected_ef_search.split(","))

    formal_retry_rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    for ef_search in expected_ef:
        if args.require_strict and (ef_search, 0.0, "prune") not in keys:
            failures.append(f"missing strict V0 run for ef={ef_search}")
        for beta in expected_betas:
            for mode in ("approx-no-retry", "approx-retry"):
                if (ef_search, beta, mode) not in keys:
                    failures.append(
                        f"missing active run for ef={ef_search} beta={beta} mode={mode}"
                    )
    for row in aggregates:
        if row["mode"] != "approx-retry":
            continue
        pair = keys.get((int(row["ef_search"]), float(row["beta"]), "approx-no-retry"))
        if pair is None:
            failures.append(
                f"missing retry-off pair for ef={row['ef_search']} beta={row['beta']}"
            )
            row["gate_classification"] = "MISSING_RETRY_OFF_PAIR"
            continue
        if (
            int(pair["query_count"]) != int(row["query_count"])
            or abs(
                float(pair["baseline_recall_at_k"])
                - float(row["baseline_recall_at_k"])
            )
            > 1e-12
            or int(pair["baseline_exact_dco"]) != int(row["baseline_exact_dco"])
        ):
            failures.append(
                f"retry pair baseline mismatch for ef={row['ef_search']} beta={row['beta']}"
            )
        denominator = float(row["baseline_recall_at_k"]) - float(pair["active_recall_at_k"])
        recovery: Optional[float]
        if denominator > 0.0:
            recovery = (
                float(row["active_recall_at_k"]) - float(pair["active_recall_at_k"])
            ) / denominator
            row["recall_recovery"] = recovery
        else:
            recovery = None
            row["recall_recovery"] = ""
            row["recall_recovery_not_needed"] = (
                float(pair["recall_loss"]) <= 0.005 + 1e-12
            )
        numeric_pass = (
            (recovery is not None and recovery >= 0.50
             or bool(row["recall_recovery_not_needed"]))
            and float(row["recall_loss"]) <= 0.005 + 1e-12
            and float(row["baseline_relative_dco_reduction"]) >= 0.15
        )
        row["numeric_gate_pass"] = numeric_pass
        if numeric_pass and float(row["catastrophic_query_rate"]) == 0.0:
            row["gate_classification"] = "GO"
        elif numeric_pass:
            row["gate_classification"] = "MANUAL_CATASTROPHIC_REVIEW"
        else:
            row["gate_classification"] = "NO_GO"
        if int(row["query_count"]) >= args.minimum_query_count:
            formal_retry_rows.append(row)

    if failures:
        overall = "INVALID_MATRIX"
    elif not formal_retry_rows:
        overall = "DEVELOPMENT_ONLY"
    elif any(row["gate_classification"] == "GO" for row in formal_retry_rows):
        overall = "GO"
    elif any(
        row["gate_classification"] == "MANUAL_CATASTROPHIC_REVIEW"
        for row in formal_retry_rows
    ):
        overall = "MANUAL_CATASTROPHIC_REVIEW"
    else:
        overall = "NO_GO"

    aggregates.sort(key=lambda row: (int(row["ef_search"]), float(row["beta"]), str(row["mode"])))
    write_csv(output / "active_experiment_summary.csv", aggregates)
    write_csv(output / "active_per_query.csv", query_rows)
    decision = {
        "format": "v0_margin_retry_active_experiment",
        "format_version": 1,
        "gate5_classification": overall,
        "minimum_query_count": args.minimum_query_count,
        "betas": expected_betas,
        "ef_search_values": expected_ef,
        "failures": failures,
        "formal_retry_rows": formal_retry_rows,
        "catastrophic_definition": "per-query Recall@k loss >= 0.2",
        "performance_claim_allowed": False,
    }
    (output / "active_experiment_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = [
        "# Unified Stage 3–5 Active Recall/DCO Report",
        "",
        f"- Gate 5 classification: **{overall}**",
        f"- Minimum formal query count: `{args.minimum_query_count}`",
        "- Recall is measured against exact ground truth at fixed efSearch.",
        "- This correctness/metrics experiment makes no QPS or latency claim.",
        "- beta=1.40 retains `stage1_fail_active_included` provenance but is fully evaluated.",
        "",
        "| ef | beta | mode | Recall | loss | DCO reduction | first prune | retry repayment | recall recovery | catastrophic | class |",
        "|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for row in aggregates:
        recovery_text = "not needed" if row["recall_recovery_not_needed"] else (
            "n/a" if row["recall_recovery"] == "" else f"{float(row['recall_recovery']):.3%}"
        )
        report.append(
            "| {ef} | {beta:.2f} | {mode} | {recall:.3%} | {loss:.3%} | {dco:.3%} | {prune:.3%} | {retry:.3%} | {recovery} | {cat:.3%} | {classification} |".format(
                ef=int(row["ef_search"]),
                beta=float(row["beta"]),
                mode=row["mode"],
                recall=float(row["active_recall_at_k"]),
                loss=float(row["recall_loss"]),
                dco=float(row["baseline_relative_dco_reduction"]),
                prune=float(row["first_prune_rate"]),
                retry=float(row["retry_repayment_rate"]),
                recovery=recovery_text,
                cat=float(row["catastrophic_query_rate"]),
                classification=row["gate_classification"],
            )
        )
    if failures:
        report.extend(["", "## Matrix failures", ""])
        report.extend(f"- {failure}" for failure in failures)
    (output / "STAGE3_5_ACTIVE_RECALL_DCO_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(f"stage3_5_gate={overall} runs={len(aggregates)}")
    return 0 if overall not in {"INVALID_MATRIX", "NO_GO"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
