"""Summarize randomized block-paired QPS at quality-selected recall points."""

import argparse
import random
import statistics
from collections import defaultdict

from common import read_json, sha256, write_json


def paired_interval(numerator: list[float], denominator: list[float],
                    seed: int = 20260919) -> tuple[float, float]:
    if len(numerator) != len(denominator) or len(numerator) < 2:
        raise ValueError("paired interval requires at least two complete blocks")
    rng = random.Random(seed)
    samples = []
    for _ in range(2000):
        indices = [rng.randrange(len(numerator)) for _ in numerator]
        samples.append(statistics.median(numerator[i] for i in indices) /
                       statistics.median(denominator[i] for i in indices))
    samples.sort()
    return samples[49], samples[1949]


def summarize(manifest: dict) -> list[dict]:
    if manifest["stage"] != "qps":
        raise ValueError("expected QPS manifest")
    grouped = defaultdict(lambda: defaultdict(dict))
    matched = {}
    for run in manifest["runs"]:
        if sha256(run["result"]) != run["sha256"]:
            raise ValueError("QPS result SHA mismatch")
        case = run["case"]
        target, method, block = case["target_recall"], case["method"], run["block"]
        if block in grouped[target][method]:
            raise ValueError("duplicate method in randomized block")
        data = read_json(run["result"])
        if data["qps"] <= 0:
            raise ValueError("nonpositive measured QPS")
        grouped[target][method][block] = data
        if target in matched and matched[target] != case["matched"]:
            raise ValueError("inconsistent recall matching")
        matched[target] = case["matched"]
    rows = []
    for target, methods in sorted(grouped.items()):
        if "baseline" not in methods or "approx-no-retry" not in methods:
            raise ValueError("missing baseline or PQ in QPS matrix")
        block_set = set(next(iter(methods.values())))
        if len(block_set) < 2 or any(set(values) != block_set for values in methods.values()):
            raise ValueError("QPS matrix does not contain complete paired blocks")
        ordered = sorted(block_set)
        exact = [methods["baseline"][block]["qps"] for block in ordered]
        pq = [methods["approx-no-retry"][block]["qps"] for block in ordered]
        for method, values in sorted(methods.items()):
            qps = [values[block]["qps"] for block in ordered]
            rows.append({
                "target_recall": target, "method": method,
                "matched": matched[target], "blocks": len(ordered),
                "qps_median": statistics.median(qps),
                "speedup_vs_exact": statistics.median(qps) / statistics.median(exact),
                "speedup_vs_exact_ci95": paired_interval(qps, exact),
                "speedup_vs_pq": statistics.median(qps) / statistics.median(pq),
                "speedup_vs_pq_ci95": paired_interval(qps, pq),
                "latency_p95_ns_median": statistics.median(
                    values[block]["latency_p95_ns"] for block in ordered),
                "latency_p99_ns_median": statistics.median(
                    values[block]["latency_p99_ns"] for block in ordered),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--qps", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    qps = read_json(args.qps)
    matched_cases = read_json(qps["config"])
    if matched_cases.get("source_quality_sha256") != sha256(args.quality):
        raise ValueError("quality manifest differs from matched-recall selection")
    output = {"schema_version": 1, "quality_manifest_sha256": sha256(args.quality),
              "qps_manifest_sha256": sha256(args.qps),
              "results": summarize(qps),
              "interval_method": "paired-block bootstrap, 2000 resamples",
              "unmatched_not_claimed": True}
    write_json(args.output, output)


if __name__ == "__main__":
    main()
