"""Summarize complete paired QPS blocks for recall-qualified residual cases."""

import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path

from common import read_json, sha256, write_json
from summarize_active import paired_interval


def case_key(case: dict) -> tuple[str, int, float]:
    return case["method"], case["ef"], case["theta"]


def summarize(quality_path: str, qps_path: str) -> dict:
    quality = read_json(quality_path)
    qps = read_json(qps_path)
    selected = read_json(qps["config"])
    if quality.get("stage") != "quality" or qps.get("stage") != "qps":
        raise ValueError("expected quality and QPS manifests")
    if selected.get("source_quality_sha256") != sha256(quality_path):
        raise ValueError("quality manifest differs from residual selection")
    if qps["companion_sha256"] != quality["companion_sha256"]:
        raise ValueError("quality and QPS used different residual companions")
    complete = read_json(Path(qps_path).parent / "complete.json")
    if complete["outputs"]["manifest"] != sha256(qps_path) or \
            complete["inputs"]["config"] != sha256(qps["config"]):
        raise ValueError("QPS completion hash mismatch")
    cases = selected["matched_cases"]
    floor = selected["recall_floor"]
    expected = {case_key(case): case for case in cases}
    if len(expected) != len(cases) or not cases or any(
            not case.get("matched") or case["quality_recall"] < floor
            for case in cases):
        raise ValueError("invalid or duplicate recall-qualified cases")
    anchors = [key for key in expected if key[0] == "residual-direct"]
    anchor_ef = read_json(quality["config"])["direct_anchor_ef"]
    if anchors != [("residual-direct", anchor_ef, 1.0)] or any(
            key[0] not in ("residual-direct", "residual-threshold")
            for key in expected):
        raise ValueError("residual tuning requires one direct anchor and no PQ")
    blocks = selected["blocks"]
    grouped = defaultdict(dict)
    for run in qps["runs"]:
        case = run["case"]
        key = case_key(case)
        block = run["block"]
        if key not in expected or case != expected[key] or \
                not 0 <= block < blocks or block in grouped[key]:
            raise ValueError("unexpected or duplicate QPS case/block")
        if sha256(run["result"]) != run["sha256"]:
            raise ValueError("QPS result SHA mismatch")
        result = read_json(run["result"])
        if result["method"] != case["method"] or \
                result["ef_search"] != case["ef"] or \
                result["theta"] != case["theta"] or \
                result["query_count"] != selected["query_count"] or \
                result["repeats"] != selected["repeats"] or \
                not math.isfinite(result["qps"]) or result["qps"] <= 0:
            raise ValueError("QPS result does not match its frozen case")
        grouped[key][block] = result
    if len(qps["runs"]) != len(cases) * blocks or any(
            set(grouped[key]) != set(range(blocks)) for key in expected):
        raise ValueError("incomplete paired QPS matrix")
    anchor_key = anchors[0]
    anchor_qps = [grouped[anchor_key][block]["qps"] for block in range(blocks)]
    rows = []
    for key, case in expected.items():
        results = [grouped[key][block] for block in range(blocks)]
        qps_values = [result["qps"] for result in results]
        rows.append({
            "method": key[0], "ef": key[1], "theta": key[2],
            "quality_recall": case["quality_recall"], "blocks": blocks,
            "qps_median": statistics.median(qps_values),
            "speedup_vs_direct": statistics.median(qps_values) /
                statistics.median(anchor_qps),
            "speedup_vs_direct_ci95": paired_interval(qps_values, anchor_qps),
            "latency_p95_ns_median": statistics.median(
                result["latency_p95_ns"] for result in results),
            "latency_p99_ns_median": statistics.median(
                result["latency_p99_ns"] for result in results),
        })
    rows.sort(key=lambda row: (-row["qps_median"],
                               row["latency_p95_ns_median"], row["ef"],
                               row["theta"]))
    return {"schema_version": 1, "recall_floor": floor,
            "quality_manifest_sha256": sha256(quality_path),
            "qps_manifest_sha256": sha256(qps_path),
            "companion_sha256": qps["companion_sha256"],
            "interval_method": "paired-block bootstrap, 2000 resamples",
            "winner": rows[0], "results": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--qps", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = summarize(args.quality, args.qps)
    write_json(args.output, report)
    winner = report["winner"]
    print(f"winner method={winner['method']} ef={winner['ef']} "
          f"theta={winner['theta']} recall={winner['quality_recall']:.6f} "
          f"qps={winner['qps_median']:.2f}")


if __name__ == "__main__":
    main()
