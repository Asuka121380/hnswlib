"""Summarize paired baseline, PQ and residual QPS near baseline recall."""

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
        raise ValueError("targeted QPS selection changed")
    for source in selected["source_quality"].values():
        if sha256(source["manifest"]) != source["sha256"]:
            raise ValueError("source quality manifest changed")
    if qps["companion_sha256"] != selected["companion_sha256"]:
        raise ValueError("QPS used another residual companion")
    complete = read_json(Path(qps_path).parent / "complete.json")
    if complete["outputs"]["manifest"] != sha256(qps_path) or \
            complete["inputs"]["config"] != sha256(selected_path):
        raise ValueError("QPS completion hash mismatch")
    cases = selected["matched_cases"]
    baseline = [case for case in cases if case["method"] == "baseline"]
    if len(baseline) != 1 or len(cases) != 5 or any(
            not case["matched"] or abs(case["quality_recall"] -
                    selected["baseline_recall"]) > selected["recall_tolerance"] + 1e-12
            for case in cases):
        raise ValueError("invalid recall-window QPS cases")
    grouped = {(case["method"], case["ef"], case.get("theta", 1.0)): {}
               for case in cases}
    if len(grouped) != len(cases):
        raise ValueError("duplicate targeted QPS cases")
    blocks = selected["blocks"]
    for run in qps["runs"]:
        case, block = run["case"], run["block"]
        key = case["method"], case["ef"], case.get("theta", 1.0)
        if key not in grouped or case != next(
                item for item in cases if (item["method"], item["ef"],
                                           item.get("theta", 1.0)) == key) or \
                not 0 <= block < blocks or block in grouped[key]:
            raise ValueError("unexpected or repeated QPS case/block")
        if sha256(run["result"]) != run["sha256"]:
            raise ValueError("QPS result SHA mismatch")
        result = read_json(run["result"])
        if result["method"] != case["method"] or \
                result["ef_search"] != case["ef"] or \
                ("theta" in case and result["theta"] != case["theta"]) or \
                ("beta" in case and result["beta"] != case["beta"]) or \
                result["query_count"] != selected["query_count"] or \
                result["repeats"] != selected["repeats"] or \
                not math.isfinite(result["qps"]) or result["qps"] <= 0:
            raise ValueError("QPS result differs from selected case")
        grouped[key][block] = result
    if len(qps["runs"]) != len(cases) * blocks or any(
            set(values) != set(range(blocks)) for values in grouped.values()):
        raise ValueError("incomplete paired QPS matrix")
    baseline_key = ("baseline", baseline[0]["ef"], 1.0)
    baseline_qps = [grouped[baseline_key][block]["qps"] for block in range(blocks)]
    baseline_median = statistics.median(baseline_qps)
    pq_case = next((case for case in cases if case["method"] == "approx-no-retry"), None)
    if pq_case is None:
        raise ValueError("missing PQ QPS case")
    pq_key = ("approx-no-retry", pq_case["ef"], 1.0)
    pq_qps = [grouped[pq_key][block]["qps"] for block in range(blocks)]
    pq_median = statistics.median(pq_qps)
    rows = []
    for case in cases:
        key = case["method"], case["ef"], case.get("theta", 1.0)
        results = [grouped[key][block] for block in range(blocks)]
        qps_values = [result["qps"] for result in results]
        rows.append({
            "method": case["method"], "ef": case["ef"],
            "theta": case.get("theta", 1.0),
            "beta": case.get("beta"),
            "quality_recall": case["quality_recall"],
            "qps_median": statistics.median(qps_values),
            "speedup_vs_baseline": statistics.median(qps_values) / baseline_median,
            "speedup_vs_baseline_ci95": paired_interval(qps_values, baseline_qps),
            "speedup_vs_pq": statistics.median(qps_values) / pq_median,
            "speedup_vs_pq_ci95": paired_interval(qps_values, pq_qps),
            "latency_p95_ns_median": statistics.median(
                result["latency_p95_ns"] for result in results),
            "latency_p99_ns_median": statistics.median(
                result["latency_p99_ns"] for result in results),
        })
    rows.sort(key=lambda row: (-row["qps_median"], row["ef"], row["theta"]))
    return {"schema_version": 1, "baseline_recall": selected["baseline_recall"],
            "recall_tolerance": selected["recall_tolerance"],
            "recall_window": selected["recall_window"],
            "baseline_ef": baseline[0]["ef"], "qps_manifest_sha256": sha256(qps_path),
            "selection_sha256": sha256(selected_path), "blocks": blocks,
            "interval_method": "paired-block bootstrap, 2000 resamples",
            "threshold_for_25_percent_gain_qps": baseline_median * 1.25,
            "results": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qps", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = summarize(args.qps)
    write_json(args.output, report)
    print(f"baseline recall={report['baseline_recall']:.6f}; "
          f"25% QPS target={report['threshold_for_25_percent_gain_qps']:.2f}")
    for row in report["results"]:
        print(f"{row['method']} ef={row['ef']} theta={row['theta']:.2f} "
              f"recall={row['quality_recall']:.6f} QPS={row['qps_median']:.2f} "
              f"vs_baseline={row['speedup_vs_baseline']:.3f} "
              f"vs_pq={row['speedup_vs_pq']:.3f}")


if __name__ == "__main__":
    main()
