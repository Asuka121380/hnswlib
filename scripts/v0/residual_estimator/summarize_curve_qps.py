"""Summarize paired QPS curves at five frozen recall anchors."""

import argparse
import math
import statistics
from pathlib import Path

from common import read_json, sha256, write_json
from summarize_active import paired_interval


def summarize(qps_path: str) -> dict:
    qps = read_json(qps_path)
    if qps.get("stage") != "qps":
        raise ValueError("expected QPS manifest")
    selected_path = qps["config"]
    selected = read_json(selected_path)
    selected_complete = read_json(Path(selected_path).with_suffix(".complete.json"))
    if selected_complete["output_sha256"] != sha256(selected_path):
        raise ValueError("curve QPS selection changed")
    for source in selected["source_quality"].values():
        if sha256(source["manifest"]) != source["sha256"]:
            raise ValueError("source quality manifest changed")
    if qps["companion_sha256"] != selected["companion_sha256"]:
        raise ValueError("QPS used another companion")
    complete = read_json(Path(qps_path).parent / "complete.json")
    if complete["outputs"]["manifest"] != sha256(qps_path) or \
            complete["inputs"]["config"] != sha256(selected_path):
        raise ValueError("QPS completion hash mismatch")

    cases = selected["matched_cases"]
    keys = [tuple(sorted(case.items())) for case in cases]
    if len(keys) != len(set(keys)) or not cases or any(
            not case["matched"] or
            abs(case["quality_recall"] - case["target_recall"]) >
            selected["recall_tolerance"] + 1e-12 for case in cases):
        raise ValueError("invalid recall-matched curve cases")
    grouped = {key: {} for key in keys}
    blocks = selected["blocks"]
    for run in qps["runs"]:
        case, block = run["case"], run["block"]
        key = tuple(sorted(case.items()))
        if key not in grouped or not 0 <= block < blocks or block in grouped[key]:
            raise ValueError("unexpected or repeated curve QPS case/block")
        if sha256(run["result"]) != run["sha256"]:
            raise ValueError("curve QPS result SHA mismatch")
        result = read_json(run["result"])
        if (result["method"] != case["method"] or
                result["ef_search"] != case["ef"] or
                ("theta" in case and result["theta"] != case["theta"]) or
                ("beta" in case and result["beta"] != case["beta"]) or
                result["query_count"] != selected["query_count"] or
                result["repeats"] != selected["repeats"] or
                not math.isfinite(result["qps"]) or result["qps"] <= 0):
            raise ValueError("curve QPS result differs from selected case")
        grouped[key][block] = result
    if len(qps["runs"]) != len(cases) * blocks or any(
            set(values) != set(range(blocks)) for values in grouped.values()):
        raise ValueError("incomplete paired curve QPS matrix")

    rows = []
    for anchor in selected["anchors"]:
        anchor_ef = anchor["baseline"]["ef"]
        anchor_cases = [case for case in cases if case["anchor_ef"] == anchor_ef]
        baseline = next(case for case in anchor_cases if case["method"] == "baseline")
        pq = next(case for case in anchor_cases if case["method"] == "approx-no-retry")
        baseline_qps = [grouped[tuple(sorted(baseline.items()))][block]["qps"]
                        for block in range(blocks)]
        pq_qps = [grouped[tuple(sorted(pq.items()))][block]["qps"]
                  for block in range(blocks)]
        baseline_median = statistics.median(baseline_qps)
        pq_median = statistics.median(pq_qps)
        for case in anchor_cases:
            results = [grouped[tuple(sorted(case.items()))][block]
                       for block in range(blocks)]
            qps_values = [result["qps"] for result in results]
            median = statistics.median(qps_values)
            rows.append({
                "anchor_ef": anchor_ef,
                "target_recall": case["target_recall"],
                "quality_recall": case["quality_recall"],
                "method": case["method"], "ef": case["ef"],
                "beta": case.get("beta"), "theta": case.get("theta"),
                "qps_median": median,
                "speedup_vs_baseline": median / baseline_median,
                "speedup_vs_baseline_ci95": paired_interval(qps_values, baseline_qps),
                "speedup_vs_pq": median / pq_median,
                "speedup_vs_pq_ci95": paired_interval(qps_values, pq_qps),
                "latency_p95_ns_median": statistics.median(
                    result["latency_p95_ns"] for result in results),
                "latency_p99_ns_median": statistics.median(
                    result["latency_p99_ns"] for result in results),
            })
    best_threshold = {
        anchor_ef: max((row for row in rows if row["anchor_ef"] == anchor_ef and
                        row["method"] == "residual-threshold"),
                       key=lambda row: row["qps_median"])
        for anchor_ef in sorted({row["anchor_ef"] for row in rows})}
    for row in rows:
        row["curve_point"] = (row["method"] != "residual-threshold" or
                              row is best_threshold[row["anchor_ef"]])
    order = {"baseline": 0, "approx-no-retry": 1, "residual-direct": 2,
             "residual-threshold": 3}
    rows.sort(key=lambda row: (row["anchor_ef"], order[row["method"]],
                               -row["qps_median"]))
    return {"schema_version": 1, "qps_manifest_sha256": sha256(qps_path),
            "selection_sha256": sha256(selected_path), "blocks": blocks,
            "interval_method": "paired-block bootstrap, 2000 resamples",
            "results": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qps", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = summarize(args.qps)
    write_json(args.output, report)
    for row in report["results"]:
        print(f"anchor={row['target_recall']:.5f} {row['method']} "
              f"ef={row['ef']} theta={row['theta']} "
              f"recall={row['quality_recall']:.5f} "
              f"QPS={row['qps_median']:.2f} "
              f"vs_baseline={row['speedup_vs_baseline']:.3f} "
              f"curve_point={row['curve_point']}")


if __name__ == "__main__":
    main()
