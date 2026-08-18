#!/usr/bin/env python3
"""Aggregate Stage 2 hypothetical retry-shadow outputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


QUERY_COLUMNS = {
    "schema_version",
    "query_id",
    "beta",
    "eligible_first_visits",
    "first_pruned",
    "first_false_pruned",
    "pruned_revisited",
    "false_pruned_revisited",
    "unrevisited_first_pruned",
    "unrevisited_false_pruned",
    "duplicate_encounters_after_prune",
    "expanded_nodes",
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return value


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / float(denominator) if denominator else 0.0


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0
    proportion = successes / float(trials)
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def int_field(row: Mapping[str, str], name: str) -> int:
    return int(row[name])


def read_query_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = QUERY_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("missing query columns: {}".format(sorted(missing)))
        for raw in reader:
            row: Dict[str, Any] = {
                "schema_version": int_field(raw, "schema_version"),
                "query_id": int_field(raw, "query_id"),
                "beta": float(raw["beta"]),
            }
            for name in QUERY_COLUMNS.difference({"schema_version", "query_id", "beta"}):
                row[name] = int_field(raw, name)
            rows.append(row)
    if not rows:
        raise ValueError("retry per-query input is empty")
    return rows


def validate_query_rows(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    failures: List[str] = []
    seen = set()
    for row in rows:
        key = (int(row["query_id"]), float(row["beta"]))
        if key in seen:
            failures.append("duplicate query/beta row {}".format(key))
        seen.add(key)
        eligible = int(row["eligible_first_visits"])
        first = int(row["first_pruned"])
        false = int(row["first_false_pruned"])
        revisited = int(row["pruned_revisited"])
        false_revisited = int(row["false_pruned_revisited"])
        unrevisited = int(row["unrevisited_first_pruned"])
        false_unrevisited = int(row["unrevisited_false_pruned"])
        if not (0 <= false <= first <= eligible):
            failures.append("invalid first-prune counts for {}".format(key))
        if not (0 <= revisited <= first):
            failures.append("invalid revisit counts for {}".format(key))
        if not (0 <= false_revisited <= false):
            failures.append("invalid false-revisit counts for {}".format(key))
        if unrevisited != first - revisited:
            failures.append("unrevisited prune invariant failed for {}".format(key))
        if false_unrevisited != false - false_revisited:
            failures.append("unrevisited false invariant failed for {}".format(key))
        if int(row["duplicate_encounters_after_prune"]) < revisited:
            failures.append("duplicate encounter invariant failed for {}".format(key))
    return failures


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_beta: Dict[float, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_beta[float(row["beta"])].append(row)
    result: List[Dict[str, Any]] = []
    sum_fields = [
        "eligible_first_visits",
        "first_pruned",
        "first_false_pruned",
        "pruned_revisited",
        "false_pruned_revisited",
        "unrevisited_first_pruned",
        "unrevisited_false_pruned",
        "duplicate_encounters_after_prune",
        "expanded_nodes",
    ]
    for beta in sorted(by_beta):
        group = by_beta[beta]
        totals = {name: sum(int(row[name]) for row in group) for name in sum_fields}
        exposed = sum(1 for row in group if int(row["unrevisited_false_pruned"]) > 0)
        query_count = len(group)
        retry_ci = wilson_interval(
            totals["false_pruned_revisited"], totals["first_false_pruned"]
        )
        exposure_ci = wilson_interval(exposed, query_count)
        result.append(
            {
                "beta": beta,
                "distance_margin_a": math.sqrt(beta) - 1.0,
                "query_count": query_count,
                **totals,
                "first_prune_rate": safe_ratio(totals["first_pruned"], totals["eligible_first_visits"]),
                "false_per_pruned": safe_ratio(totals["first_false_pruned"], totals["first_pruned"]),
                "all_pruned_revisit_rate": safe_ratio(totals["pruned_revisited"], totals["first_pruned"]),
                "false_pruned_revisit_rate": safe_ratio(totals["false_pruned_revisited"], totals["first_false_pruned"]),
                "false_retry_ci_low": retry_ci[0],
                "false_retry_ci_high": retry_ci[1],
                "unrevisited_false_event_rate": safe_ratio(totals["unrevisited_false_pruned"], totals["eligible_first_visits"]),
                "unrevisited_false_query_exposure_count": exposed,
                "unrevisited_false_query_exposure_rate": safe_ratio(exposed, query_count),
                "query_exposure_ci_low": exposure_ci[0],
                "query_exposure_ci_high": exposure_ci[1],
                "potential_recovered_positive": totals["false_pruned_revisited"],
            }
        )
    return result


def stage1_gate_passes(row: Mapping[str, Any]) -> bool:
    prune_rate = float(row["first_prune_rate"])
    false_rate = float(row["false_per_pruned"])
    return (prune_rate >= 0.20 and false_rate <= 0.01) or (
        prune_rate >= 0.35 and false_rate <= 0.02
    )


def classify(row: Mapping[str, Any]) -> str:
    if not bool(row.get("stage1_eligible", True)):
        return "CONTROL_STAGE1_FAIL"
    prune_rate = float(row["first_prune_rate"])
    retry_rate = float(row["false_pruned_revisit_rate"])
    event_rate = float(row["unrevisited_false_event_rate"])
    exposure = float(row["unrevisited_false_query_exposure_rate"])
    if 0.20 <= prune_rate <= 0.50 and retry_rate >= 0.70:
        return "GO"
    if (
        0.40 <= retry_rate < 0.70
        and event_rate <= 0.0005
        and exposure <= 0.05
    ):
        return "CONDITIONAL_GO"
    if retry_rate < 0.40 and exposure > 0.05:
        return "NO_GO"
    return "INCONCLUSIVE"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compress_records(source: Path, destination: Path) -> None:
    with source.open("rb") as input_handle:
        with gzip.open(destination, "wb", compresslevel=6) as output_handle:
            shutil.copyfileobj(input_handle, output_handle)


def structural_summary(
    records_path: Path,
    query_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    expanded = {
        (int(row["query_id"]), float(row["beta"])):
            max(1, int(row["expanded_nodes"]))
        for row in query_rows
    }
    grouped: Dict[Tuple[float, str, str, int], List[Dict[str, int]]] = defaultdict(list)
    with records_path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            query_id = int(raw["query_id"])
            beta = float(raw["beta"])
            first_index = int(raw["first_expansion_index"])
            progress = first_index / float(expanded.get((query_id, beta), 1))
            if progress < 0.25:
                progress_bucket = "early_0_25"
            elif progress < 0.75:
                progress_bucket = "middle_25_75"
            else:
                progress_bucket = "late_75_100"
            degree = int(raw["candidate_degree"])
            if degree <= 8:
                degree_bucket = "degree_0_8"
            elif degree <= 16:
                degree_bucket = "degree_9_16"
            elif degree <= 32:
                degree_bucket = "degree_17_32"
            else:
                degree_bucket = "degree_33_plus"
            item = {
                "revisited": int(raw["revisited"]),
                "duplicates": int(raw["duplicate_encounters"]),
                "delay": int(raw["revisit_delay_expansions"]),
            }
            false_flag = int(raw["first_false_prune"])
            grouped[(beta, "search_progress", progress_bucket, false_flag)].append(item)
            grouped[(beta, "candidate_degree", degree_bucket, false_flag)].append(item)
    rows: List[Dict[str, Any]] = []
    for (beta, group_type, bucket, false_flag), items in sorted(grouped.items()):
        revisited = sum(item["revisited"] for item in items)
        delays = [item["delay"] for item in items if item["revisited"]]
        rows.append(
            {
                "beta": beta,
                "group_type": group_type,
                "bucket": bucket,
                "first_false_prune": false_flag,
                "sampled_first_pruned": len(items),
                "sampled_revisited": revisited,
                "sampled_revisit_rate": safe_ratio(revisited, len(items)),
                "sampled_second_encounter_only": sum(1 for item in items if item["duplicates"] == 1),
                "sampled_third_or_more_encounters": sum(1 for item in items if item["duplicates"] >= 2),
                "mean_revisit_delay_expansions": safe_ratio(sum(delays), len(delays)),
            }
        )
    return rows


def generate_figures(
    output: Path,
    summaries: Sequence[Mapping[str, Any]],
    records_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(
        [float(row["beta"]) for row in summaries],
        [100.0 * float(row["all_pruned_revisit_rate"]) for row in summaries],
        marker="o",
        label="all first-pruned",
    )
    axis.plot(
        [float(row["beta"]) for row in summaries],
        [100.0 * float(row["false_pruned_revisit_rate"]) for row in summaries],
        marker="o",
        label="false first-pruned",
    )
    axis.set_xlabel("Squared-distance scale beta")
    axis.set_ylabel("Potential revisit rate (%)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "revisit_probability_by_beta.png", dpi=160)
    plt.close(fig)

    delays: Dict[float, List[int]] = defaultdict(list)
    with records_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["revisited"]):
                delays[float(row["beta"])].append(int(row["revisit_delay_expansions"]))
    fig, axis = plt.subplots(figsize=(7, 5))
    for beta in sorted(delays):
        values = sorted(delays[beta])
        if values:
            axis.plot(values, [(i + 1) / len(values) for i in range(len(values))], label="beta={}".format(beta))
    axis.set_xlabel("Expansion delay to second encounter")
    axis.set_ylabel("CDF (sampled detailed records)")
    axis.grid(True, alpha=0.3)
    if axis.lines:
        axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "revisit_delay_cdf.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.bar(
        [str(row["beta"]) for row in summaries],
        [100.0 * float(row["unrevisited_false_query_exposure_rate"]) for row in summaries],
    )
    axis.set_xlabel("Squared-distance scale beta")
    axis.set_ylabel("Queries with unrevisited false prune (%)")
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "unrevisited_query_exposure.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-output-dir", type=Path, required=True)
    parser.add_argument("--stage1-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = args.runner_output_dir.resolve()
    output = args.output_dir.resolve()
    per_query_path = runner / "retry_shadow_per_query_raw.csv"
    records_path = runner / "retry_shadow_records.csv"
    summary_path = runner / "summary.json"
    metadata_path = runner / "metadata.json"
    for path in (per_query_path, records_path, summary_path, metadata_path, args.stage1_candidates):
        if not path.is_file():
            raise SystemExit("required input is missing: {}".format(path))
    if output.exists() and any(output.iterdir()) and not args.force:
        raise SystemExit("output directory is not empty; pass --force: {}".format(output))
    if output.exists() and args.force:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    runner_summary = load_json(summary_path)
    metadata = load_json(metadata_path)
    stage1 = load_json(args.stage1_candidates)
    rows = read_query_rows(per_query_path)
    failures = validate_query_rows(rows)
    expected_betas = sorted(float(value) for value in stage1.get("selected_betas", []))
    stage1_test_rows = stage1.get("test_rows", [])
    if stage1_test_rows:
        eligible_betas = {
            float(row["beta"])
            for row in stage1_test_rows
            if stage1_gate_passes(row)
        }
    else:
        eligible_betas = set(expected_betas)
    observed_betas = sorted({float(row["beta"]) for row in rows})
    if observed_betas != expected_betas:
        failures.append(
            "observed betas {} do not match Stage 1 candidates {}".format(observed_betas, expected_betas)
        )
    if runner_summary.get("status") != "valid":
        failures.append("runner summary status is not valid")
    if int(runner_summary.get("mismatch_queries", -1)) != 0:
        failures.append("retry shadow changed final top-K")
    if int(runner_summary.get("exact_distance_saved", -1)) != 0:
        failures.append("retry shadow unexpectedly saved exact distances")
    if metadata.get("mode") != "retry-shadow":
        failures.append("runner metadata mode is not retry-shadow")
    query_count = int(runner_summary.get("query_count", 0))
    if len(rows) != query_count * len(expected_betas):
        failures.append("per-query row count does not equal queries * betas")

    enriched_query_rows: List[Dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched["first_prune_rate"] = safe_ratio(int(row["first_pruned"]), int(row["eligible_first_visits"]))
        enriched["false_pruned_revisit_rate"] = safe_ratio(int(row["false_pruned_revisited"]), int(row["first_false_pruned"]))
        enriched["unrevisited_false_exposed"] = int(int(row["unrevisited_false_pruned"]) > 0)
        enriched_query_rows.append(enriched)
    summaries = aggregate_rows(rows)
    for row in summaries:
        row["stage1_eligible"] = float(row["beta"]) in eligible_betas
        row["gate2_classification"] = classify(row)
    write_csv(output / "retry_shadow_summary.csv", summaries)
    write_csv(output / "retry_shadow_per_query.csv", enriched_query_rows)
    structure_rows = structural_summary(records_path, rows)
    write_csv(output / "retry_shadow_structural_summary.csv", structure_rows)
    compress_records(records_path, output / "retry_shadow_records.csv.gz")

    classifications = [
        str(row["gate2_classification"])
        for row in summaries
        if bool(row["stage1_eligible"])
    ]
    if "GO" in classifications:
        overall = "GO"
    elif "CONDITIONAL_GO" in classifications:
        overall = "CONDITIONAL_GO"
    elif "NO_GO" in classifications and all(item == "NO_GO" for item in classifications):
        overall = "NO_GO"
    else:
        overall = "INCONCLUSIVE"
    correctness_passed = not failures
    document = {
        "format": "v0_margin_retry_candidates_with_retry",
        "format_version": 1,
        "stage1_candidate_source": str(args.stage1_candidates.resolve()),
        "selected_betas": expected_betas,
        "retry_shadow_rows": summaries,
        "correctness_gate_passed": correctness_passed,
        "correctness_failures": failures,
        "gate2_classification": overall,
        "baseline_path_upper_bound": True,
    }
    (output / "candidate_operating_points_with_retry.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not args.skip_figures:
        generate_figures(output, summaries, records_path)

    report = [
        "# Stage 2 Retry Upper-Bound Report",
        "",
        "- Correctness gate: **{}**".format("PASS" if correctness_passed else "FAIL"),
        "- Scientific classification: **{}**".format(overall),
        "- Queries: `{}`".format(query_count),
        "- Betas frozen from Stage 1: `{}`".format(expected_betas),
        "- Interpretation: baseline-path potential revisit upper bound; not active-pruning recovery.",
        "- Structural records: deterministic sample; full Gate denominators come from per-query summaries.",
        "",
        "| beta | first prune | false/pruned | all revisit | false revisit | unrevisited false/event | query exposure | class |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summaries:
        report.append(
            "| {beta:.2f} | {prune:.3%} | {false:.3%} | {all_retry:.3%} | {false_retry:.3%} | {event:.4%} | {exposure:.3%} | {classification} |".format(
                beta=float(row["beta"]),
                prune=float(row["first_prune_rate"]),
                false=float(row["false_per_pruned"]),
                all_retry=float(row["all_pruned_revisit_rate"]),
                false_retry=float(row["false_pruned_revisit_rate"]),
                event=float(row["unrevisited_false_event_rate"]),
                exposure=float(row["unrevisited_false_query_exposure_rate"]),
                classification=row["gate2_classification"],
            )
        )
    if failures:
        report.extend(["", "## Correctness failures", ""])
        report.extend("- {}".format(item) for item in failures)
    report.extend(
        [
            "",
            "Full denominators come from `retry_shadow_per_query_raw.csv`; the detailed candidate CSV may be sampled and is used only for structural/delay analysis.",
            "",
        ]
    )
    (output / "STAGE2_RETRY_UPPER_BOUND_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print("stage2_gate={} correctness={}".format(overall, "PASS" if correctness_passed else "FAIL"))
    if not correctness_passed or overall in ("NO_GO", "INCONCLUSIVE"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
