#!/usr/bin/env python3
"""Stage 1 offline replay for margin-calibrated V0 raw pruning."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence, Set, Tuple


REQUIRED_COLUMNS = {
    "query_id",
    "bound_status",
    "threshold",
    "approximate_squared_distance",
    "shadow_exact_squared_distance",
}

DEFAULT_BETAS = (
    1.00,
    1.25,
    1.30,
    1.35,
    1.40,
    1.45,
    1.50,
    1.55,
    1.60,
    1.65,
    1.70,
    1.80,
    2.00,
)


@dataclass
class Counts:
    valid_records: int = 0
    invalid_records: int = 0
    exact_negatives: int = 0
    first_pruned: int = 0
    true_pruned: int = 0
    false_pruned: int = 0
    near_threshold_records: int = 0
    near_threshold_pruned: int = 0
    near_threshold_false_pruned: int = 0
    false_prune_magnitudes: List[float] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def parse_betas(raw: str) -> List[float]:
    values = sorted({float(item.strip()) for item in raw.split(",") if item.strip()})
    if not values or any(not math.isfinite(item) or item < 1.0 for item in values):
        raise ValueError("all beta values must be finite and >= 1")
    if 1.0 not in values:
        values.insert(0, 1.0)
    return values


def squared_to_distance_margin(beta: float) -> float:
    return math.sqrt(beta) - 1.0


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


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


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def scan_query_ids(path: Path) -> Tuple[List[int], List[str], int]:
    query_ids: Set[int] = set()
    total_rows = 0
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = sorted(REQUIRED_COLUMNS.difference(columns))
        if missing:
            raise ValueError("input is missing columns: {}".format(", ".join(missing)))
        for row in reader:
            total_rows += 1
            query_ids.add(int(row["query_id"]))
    if not query_ids:
        raise ValueError("input contains no query IDs")
    return sorted(query_ids), list(columns), total_rows


def make_query_split(query_ids: Sequence[int], seed: int) -> Dict[int, str]:
    shuffled = list(query_ids)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * 0.60)
    validation_end = train_end + int(len(shuffled) * 0.20)
    split: Dict[int, str] = {}
    for query_id in shuffled[:train_end]:
        split[query_id] = "train"
    for query_id in shuffled[train_end:validation_end]:
        split[query_id] = "validation"
    for query_id in shuffled[validation_end:]:
        split[query_id] = "test"
    return split


def write_split_manifest(
    path: Path,
    query_ids: Sequence[int],
    split_by_query: Mapping[int, str],
    seed: int,
    input_path: Path,
    input_sha256: str,
) -> None:
    splits: Dict[str, List[int]] = {"train": [], "validation": [], "test": []}
    for query_id in query_ids:
        splits[split_by_query[query_id]].append(query_id)
    value = {
        "format": "v0_margin_retry_query_split",
        "format_version": 1,
        "method": "sorted query IDs shuffled with Python random.Random(seed), then exact 60/20/20 slicing",
        "seed": seed,
        "input_path": str(input_path),
        "input_sha256": input_sha256,
        "query_count": len(query_ids),
        "splits": splits,
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "pairwise_disjoint": not (
            set(splits["train"]) & set(splits["validation"])
            or set(splits["train"]) & set(splits["test"])
            or set(splits["validation"]) & set(splits["test"])
        ),
    }
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_event(row: Mapping[str, str]) -> Tuple[bool, float, float, float]:
    try:
        status = int(row["bound_status"])
        threshold = float(row["threshold"])
        estimate = float(row["approximate_squared_distance"])
        exact = float(row["shadow_exact_squared_distance"])
    except (TypeError, ValueError):
        return False, 0.0, 0.0, 0.0
    valid = (
        status == 0
        and math.isfinite(threshold)
        and math.isfinite(estimate)
        and math.isfinite(exact)
        and threshold > 0.0
        and estimate >= 0.0
        and exact >= 0.0
    )
    return valid, threshold, estimate, exact


def evaluate_splits(
    input_path: Path,
    split_by_query: Mapping[int, str],
    betas_by_split: Mapping[str, Sequence[float]],
) -> Tuple[Dict[Tuple[str, float], Counts], Dict[Tuple[str, float, int], Counts]]:
    aggregate: Dict[Tuple[str, float], Counts] = {}
    per_query: Dict[Tuple[str, float, int], Counts] = {}
    for split, betas in betas_by_split.items():
        for beta in betas:
            aggregate[(split, beta)] = Counts()

    with open_text(input_path) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            query_id = int(row["query_id"])
            split = split_by_query[query_id]
            betas = betas_by_split.get(split, ())
            if not betas:
                continue
            valid, threshold, estimate, exact = parse_event(row)
            for beta in betas:
                total = aggregate[(split, beta)]
                query = per_query.setdefault((split, beta, query_id), Counts())
                if not valid:
                    total.invalid_records += 1
                    query.invalid_records += 1
                    continue
                exact_negative = exact > threshold
                first_prune = estimate > beta * threshold
                false_prune = first_prune and not exact_negative
                near_threshold = abs(exact - threshold) <= 0.05 * threshold
                for counts in (total, query):
                    counts.valid_records += 1
                    counts.exact_negatives += int(exact_negative)
                    counts.first_pruned += int(first_prune)
                    counts.true_pruned += int(first_prune and exact_negative)
                    counts.false_pruned += int(false_prune)
                    counts.near_threshold_records += int(near_threshold)
                    counts.near_threshold_pruned += int(near_threshold and first_prune)
                    counts.near_threshold_false_pruned += int(near_threshold and false_prune)
                if false_prune:
                    magnitude = max(0.0, (threshold - exact) / threshold)
                    total.false_prune_magnitudes.append(magnitude)
                    query.false_prune_magnitudes.append(magnitude)
    return aggregate, per_query


def summary_row(
    split: str,
    beta: float,
    counts: Counts,
    query_counts: Sequence[Counts],
) -> Dict[str, Any]:
    exposed = sum(1 for item in query_counts if item.false_pruned > 0)
    query_count = len(query_counts)
    prune_ci = wilson_interval(counts.first_pruned, counts.valid_records)
    false_ci = wilson_interval(counts.false_pruned, counts.first_pruned)
    exposure_ci = wilson_interval(exposed, query_count)
    return {
        "split": split,
        "beta": beta,
        "distance_margin_a": squared_to_distance_margin(beta),
        "valid_records": counts.valid_records,
        "invalid_records": counts.invalid_records,
        "exact_negatives": counts.exact_negatives,
        "first_pruned": counts.first_pruned,
        "true_pruned": counts.true_pruned,
        "false_pruned": counts.false_pruned,
        "first_prune_rate": safe_ratio(counts.first_pruned, counts.valid_records),
        "first_prune_ci_low": prune_ci[0],
        "first_prune_ci_high": prune_ci[1],
        "false_per_pruned": safe_ratio(counts.false_pruned, counts.first_pruned),
        "false_per_pruned_ci_low": false_ci[0],
        "false_per_pruned_ci_high": false_ci[1],
        "exact_negative_coverage": safe_ratio(counts.true_pruned, counts.exact_negatives),
        "prune_precision": safe_ratio(counts.true_pruned, counts.first_pruned),
        "query_count": query_count,
        "query_false_prune_exposure_count": exposed,
        "query_false_prune_exposure_rate": safe_ratio(exposed, query_count),
        "query_exposure_ci_low": exposure_ci[0],
        "query_exposure_ci_high": exposure_ci[1],
        "near_threshold_records": counts.near_threshold_records,
        "near_threshold_pruned": counts.near_threshold_pruned,
        "near_threshold_false_pruned": counts.near_threshold_false_pruned,
        "false_prune_magnitude_p50": quantile(counts.false_prune_magnitudes, 0.50),
        "false_prune_magnitude_p95": quantile(counts.false_prune_magnitudes, 0.95),
        "false_prune_magnitude_max": max(counts.false_prune_magnitudes, default=0.0),
    }


def rows_from_counts(
    aggregate: Mapping[Tuple[str, float], Counts],
    per_query: Mapping[Tuple[str, float, int], Counts],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    event_rows: List[Dict[str, Any]] = []
    query_rows: List[Dict[str, Any]] = []
    for split, beta in sorted(aggregate, key=lambda item: (item[0], item[1])):
        matching = [
            (query_id, counts)
            for (query_split, query_beta, query_id), counts in per_query.items()
            if query_split == split and query_beta == beta
        ]
        matching.sort(key=lambda item: item[0])
        event_rows.append(summary_row(split, beta, aggregate[(split, beta)], [c for _, c in matching]))
        for query_id, counts in matching:
            query_rows.append(
                {
                    "split": split,
                    "beta": beta,
                    "query_id": query_id,
                    "valid_records": counts.valid_records,
                    "invalid_records": counts.invalid_records,
                    "exact_negatives": counts.exact_negatives,
                    "first_pruned": counts.first_pruned,
                    "false_pruned": counts.false_pruned,
                    "first_prune_rate": safe_ratio(counts.first_pruned, counts.valid_records),
                    "false_per_pruned": safe_ratio(counts.false_pruned, counts.first_pruned),
                    "false_prune_exposed": int(counts.false_pruned > 0),
                }
            )
    return event_rows, query_rows


def qualifies_gate(row: Mapping[str, Any]) -> bool:
    prune_rate = float(row["first_prune_rate"])
    false_rate = float(row["false_per_pruned"])
    return (prune_rate >= 0.20 and false_rate <= 0.01) or (
        prune_rate >= 0.35 and false_rate <= 0.02
    )


def choose_candidates(validation_rows: Sequence[Mapping[str, Any]]) -> List[float]:
    qualified = sorted(
        float(row["beta"])
        for row in validation_rows
        if float(row["beta"]) > 1.0 and qualifies_gate(row)
    )
    if len(qualified) <= 3:
        return qualified
    indices = (0, (len(qualified) - 1) // 2, len(qualified) - 1)
    return sorted({qualified[index] for index in indices})


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_figures(
    output_dir: Path,
    event_rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[Tuple[str, float], Counts],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    validation = [row for row in event_rows if row["split"] == "validation"]

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(
        [100.0 * float(row["first_prune_rate"]) for row in validation],
        [100.0 * float(row["false_per_pruned"]) for row in validation],
        marker="o",
    )
    for row in validation:
        axis.annotate(str(row["beta"]), (100.0 * float(row["first_prune_rate"]), 100.0 * float(row["false_per_pruned"])))
    axis.set_xlabel("First-prune rate (%)")
    axis.set_ylabel("False / pruned (%)")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "prune_vs_false_prune.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(
        [100.0 * float(row["first_prune_rate"]) for row in validation],
        [100.0 * float(row["query_false_prune_exposure_rate"]) for row in validation],
        marker="o",
    )
    for row in validation:
        axis.annotate(str(row["beta"]), (100.0 * float(row["first_prune_rate"]), 100.0 * float(row["query_false_prune_exposure_rate"])))
    axis.set_xlabel("First-prune rate (%)")
    axis.set_ylabel("Queries exposed to >=1 false prune (%)")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "prune_vs_query_exposure.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    for row in validation:
        beta = float(row["beta"])
        values = sorted(aggregate[("validation", beta)].false_prune_magnitudes)
        if not values:
            continue
        y = [(index + 1) / float(len(values)) for index in range(len(values))]
        axis.plot(values, y, label="beta={}".format(beta))
    axis.set_xlabel("(threshold - exact) / threshold")
    axis.set_ylabel("CDF")
    axis.grid(True, alpha=0.3)
    if axis.lines:
        axis.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(figures / "false_prune_magnitude_cdf.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260818)
    parser.add_argument("--betas", default=",".join(str(value) for value in DEFAULT_BETAS))
    parser.add_argument("--expected-valid-records", type=int, default=211836)
    parser.add_argument("--expected-exact-negatives", type=int, default=185205)
    parser.add_argument("--expected-query-count", type=int, default=1000)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output = args.output_dir.resolve()
    if not input_path.is_file():
        raise SystemExit("input does not exist: {}".format(input_path))
    if output.exists() and any(output.iterdir()) and not args.force:
        raise SystemExit("output directory is not empty; pass --force: {}".format(output))
    if output.exists() and args.force:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    betas = parse_betas(args.betas)
    query_ids, columns, total_input_rows = scan_query_ids(input_path)
    input_sha256 = sha256_file(input_path)
    split_by_query = make_query_split(query_ids, args.split_seed)
    write_split_manifest(
        output / "query_split_manifest.json",
        query_ids,
        split_by_query,
        args.split_seed,
        input_path,
        input_sha256,
    )

    development_aggregate, development_queries = evaluate_splits(
        input_path,
        split_by_query,
        {"train": betas, "validation": betas},
    )
    development_rows, development_query_rows = rows_from_counts(
        development_aggregate, development_queries
    )
    validation_rows = [row for row in development_rows if row["split"] == "validation"]
    selected_betas = choose_candidates(validation_rows)

    test_aggregate: Dict[Tuple[str, float], Counts] = {}
    test_queries: Dict[Tuple[str, float, int], Counts] = {}
    if selected_betas:
        test_aggregate, test_queries = evaluate_splits(
            input_path,
            split_by_query,
            {"test": selected_betas},
        )
    test_rows, test_query_rows = rows_from_counts(test_aggregate, test_queries)
    all_event_rows = development_rows + test_rows
    all_query_rows = development_query_rows + test_query_rows
    write_csv(output / "margin_sweep_event_summary.csv", all_event_rows)
    write_csv(output / "margin_sweep_query_summary.csv", all_query_rows)

    full_reference_aggregate, _ = evaluate_splits(
        input_path,
        {query_id: "all" for query_id in query_ids},
        {"all": [1.0]},
    )
    reference = full_reference_aggregate[("all", 1.0)]
    correctness_checks = [
        {
            "name": "valid_records",
            "actual": reference.valid_records,
            "expected": args.expected_valid_records,
            "passed": reference.valid_records == args.expected_valid_records,
        },
        {
            "name": "exact_negatives",
            "actual": reference.exact_negatives,
            "expected": args.expected_exact_negatives,
            "passed": reference.exact_negatives == args.expected_exact_negatives,
        },
        {
            "name": "query_count",
            "actual": len(query_ids),
            "expected": args.expected_query_count,
            "passed": len(query_ids) == args.expected_query_count,
        },
        {
            "name": "query_split_disjoint",
            "actual": len(split_by_query),
            "expected": len(query_ids),
            "passed": len(split_by_query) == len(query_ids),
        },
    ]
    test_gate_rows = [row for row in test_rows if qualifies_gate(row)]
    scientific_gate_passed = bool(test_gate_rows)
    correctness_gate_passed = all(item["passed"] for item in correctness_checks)

    candidate_document = {
        "format": "v0_margin_retry_candidate_operating_points",
        "format_version": 1,
        "selection_split": "validation",
        "selection_rule": "qualifying beta values; if >3 choose lowest, middle, highest beta deterministically",
        "gate_rule": "prune>=20% and false/pruned<=1%, OR prune>=35% and false/pruned<=2%",
        "reference_beta_excluded_from_selection": 1.0,
        "selected_betas": selected_betas,
        "validation_rows": [row for row in validation_rows if float(row["beta"]) in selected_betas],
        "test_rows": test_rows,
        "test_evaluation_passes": 1 if selected_betas else 0,
        "gate1_correctness_passed": correctness_gate_passed,
        "gate1_scientific_passed": scientific_gate_passed,
        "gate1_passed": correctness_gate_passed and scientific_gate_passed,
    }
    (output / "candidate_operating_points.json").write_text(
        json.dumps(candidate_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    input_audit = {
        "format": "v0_margin_retry_stage1_input_audit",
        "format_version": 1,
        "input_path": str(input_path),
        "input_sha256": input_sha256,
        "columns": columns,
        "total_input_rows": total_input_rows,
        "valid_records": reference.valid_records,
        "invalid_records": reference.invalid_records,
        "exact_negatives": reference.exact_negatives,
        "exact_negative_rate": safe_ratio(reference.exact_negatives, reference.valid_records),
        "beta_1_first_pruned": reference.first_pruned,
        "beta_1_first_prune_rate": safe_ratio(reference.first_pruned, reference.valid_records),
        "beta_1_false_pruned": reference.false_pruned,
        "beta_1_false_per_pruned": safe_ratio(reference.false_pruned, reference.first_pruned),
        "correctness_checks": correctness_checks,
    }
    (output / "input_audit.json").write_text(
        json.dumps(input_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    combined_aggregate = dict(development_aggregate)
    combined_aggregate.update(test_aggregate)
    if not args.skip_figures:
        generate_figures(output, all_event_rows, combined_aggregate)

    report = [
        "# Stage 1 Margin Frontier Report",
        "",
        "- Correctness gate: **{}**".format("PASS" if correctness_gate_passed else "FAIL"),
        "- Scientific viability gate: **{}**".format("PASS" if scientific_gate_passed else "FAIL"),
        "- Overall Gate 1: **{}**".format("PASS" if correctness_gate_passed and scientific_gate_passed else "FAIL"),
        "- Input SHA-256: `{}`".format(input_sha256),
        "- Queries: `{}` (train/validation/test = `{}/{}/{}`)".format(
            len(query_ids),
            sum(1 for value in split_by_query.values() if value == "train"),
            sum(1 for value in split_by_query.values() if value == "validation"),
            sum(1 for value in split_by_query.values() if value == "test"),
        ),
        "- Valid records: `{}`".format(reference.valid_records),
        "- Invalid records: `{}`".format(reference.invalid_records),
        "- Exact negatives: `{}` ({:.6%})".format(
            reference.exact_negatives,
            safe_ratio(reference.exact_negatives, reference.valid_records),
        ),
        "- Selected beta values: `{}`".format(selected_betas),
        "",
        "## Held-out test operating points",
        "",
        "| beta | distance margin a | first prune | false/pruned | query exposure | Gate |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in test_rows:
        report.append(
            "| {beta:.2f} | {distance_margin_a:.4f} | {prune:.3%} | {false:.3%} | {exposure:.3%} | {gate} |".format(
                beta=float(row["beta"]),
                distance_margin_a=float(row["distance_margin_a"]),
                prune=float(row["first_prune_rate"]),
                false=float(row["false_per_pruned"]),
                exposure=float(row["query_false_prune_exposure_rate"]),
                gate="PASS" if qualifies_gate(row) else "FAIL",
            )
        )
    if not test_rows:
        report.append("| — | — | — | — | — | FAIL |")
    report.extend(
        [
            "",
            "Candidate-level false/prune rates and query-level exposure are intentionally reported separately. Test queries were evaluated only for the beta values frozen from validation.",
            "",
        ]
    )
    (output / "STAGE1_MARGIN_FRONTIER_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(
        "stage1_gate={} selected_betas={}".format(
            "PASS" if correctness_gate_passed and scientific_gate_passed else "FAIL",
            ",".join(str(item) for item in selected_betas),
        )
    )
    return 0 if correctness_gate_passed and scientific_gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
