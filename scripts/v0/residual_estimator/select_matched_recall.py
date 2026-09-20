"""Select measured quality points for the finite paired QPS matrix."""

import argparse
import csv
import math
from pathlib import Path

from common import read_json, sha256, write_json


def nearest(candidates: list[dict], target: float) -> dict:
    if not candidates:
        raise ValueError("missing method in quality matrix")
    return min(candidates, key=lambda item: (abs(item["recall"] - target),
                                              item["case"]["ef"]))


def selection_recall(result: str, expected_ids: set[int]) -> float:
    metrics = Path(result).parent / "query_metrics.csv"
    observed: dict[int, float] = {}
    with metrics.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            query_id = int(row["query_id"])
            if query_id not in expected_ids:
                continue
            if query_id in observed:
                raise ValueError(f"repeated selection query {query_id}: {metrics}")
            recall = float(row["v0_recall_at_k"])
            if not math.isfinite(recall) or not 0.0 <= recall <= 1.0:
                raise ValueError(f"invalid selection recall: {metrics}")
            observed[query_id] = recall
    if set(observed) != expected_ids:
        raise ValueError(f"missing selection queries in {metrics}")
    return sum(observed.values()) / len(expected_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest, config = read_json(args.quality), read_json(args.config)
    if manifest["stage"] != "quality":
        raise ValueError("expected active quality manifest")
    contract = read_json(manifest["contract"])
    split = read_json(contract["split_path"])
    expected_ids = set(split["development"]) | set(split["selection"])
    if not expected_ids or expected_ids & set(split["audit"]):
        raise ValueError("invalid quality selection split")
    points = []
    for run in manifest["runs"]:
        points.append({"case": run["case"],
                       "recall": selection_recall(run["result"], expected_ids),
                       "result": run["result"]})
    selected = []
    for target in config["target_recalls"]:
        exact = nearest([p for p in points if p["case"]["method"] == "baseline"], target)
        pq = nearest([p for p in points if p["case"]["method"] == "approx-no-retry"], target)
        residual = nearest([p for p in points if p["case"]["method"] == "residual-direct"], target)
        spread = max(p["recall"] for p in (exact, pq, residual)) - min(
            p["recall"] for p in (exact, pq, residual))
        selected.append({"target_recall": target,
                         "matched": spread <= config["match_tolerance"],
                         "actual_spread": spread,
                         "points": [exact, pq, residual]})
    frozen = dict(config)
    frozen["matched_cases"] = [
        {**point["case"], "target_recall": row["target_recall"],
         "quality_recall": point["recall"], "matched": row["matched"]}
        for row in selected for point in row["points"]
    ]
    frozen["quality_selection"] = selected
    frozen["quality_selection_scope"] = "development+selection"
    frozen["quality_selection_query_count"] = len(expected_ids)
    frozen["source_quality_sha256"] = sha256(args.quality)
    write_json(args.output, frozen)
    write_json(Path(args.output).with_suffix(".complete.json"),
               {"quality_sha256": sha256(args.quality),
                "output_sha256": sha256(args.output)})


if __name__ == "__main__":
    main()
