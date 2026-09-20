"""Freeze every recall-qualified residual case for the tuning QPS run."""

import argparse
import csv
import math
from pathlib import Path

from common import read_json, sha256, write_json
from run_active_matrix import quality_cases


def measured_recall(result: str, expected_ids: set[int]) -> float:
    metrics = Path(result).parent / "query_metrics.csv"
    observed = {}
    with metrics.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            query_id = int(row["query_id"])
            if query_id not in expected_ids or query_id in observed:
                raise ValueError(f"unexpected or repeated query ID in {metrics}")
            recall = float(row["v0_recall_at_k"])
            if not math.isfinite(recall) or not 0.0 <= recall <= 1.0:
                raise ValueError(f"invalid recall in {metrics}")
            observed[query_id] = recall
    if set(observed) != expected_ids:
        raise ValueError(f"missing quality query IDs in {metrics}")
    return sum(observed.values()) / len(expected_ids)


def freeze(quality_path: str, config_path: str) -> dict:
    quality = read_json(quality_path)
    config = read_json(config_path)
    if quality.get("stage") != "quality" or \
            quality.get("quality_selection_scope") != "development+selection":
        raise ValueError("expected development+selection quality manifest")
    complete = read_json(Path(quality_path).parent / "complete.json")
    if complete["outputs"]["manifest"] != sha256(quality_path):
        raise ValueError("quality completion hash mismatch")
    contract = read_json(quality["contract"])
    split = read_json(contract["split_path"])
    expected_ids = set(split["development"]) | set(split["selection"])
    if not expected_ids or expected_ids & set(split["audit"]):
        raise ValueError("invalid frozen quality split")
    if quality["query_id_file_sha256"] != sha256(
            Path(quality_path).parent / "quality-query-ids.txt"):
        raise ValueError("quality query ID hash mismatch")
    expected_cases = quality_cases(read_json(quality["config"]), {})
    runs = quality["runs"]
    if [run["case"] for run in runs] != expected_cases:
        raise ValueError("quality runs differ from the frozen residual grid")
    floor = config.get("recall_floor")
    if not isinstance(floor, (int, float)) or isinstance(floor, bool) or \
            not math.isfinite(floor) or not 0.0 < floor <= 1.0:
        raise ValueError("invalid recall floor")
    points = []
    for run in runs:
        if sha256(run["result"]) != run["sha256"]:
            raise ValueError("quality result SHA mismatch")
        if read_json(run["result"]).get("status") != "valid":
            raise ValueError("invalid quality result")
        points.append({"case": run["case"],
                       "recall": measured_recall(run["result"], expected_ids),
                       "result": run["result"]})
    anchor = [point for point in points
              if point["case"]["method"] == "residual-direct"]
    if len(anchor) != 1 or anchor[0]["recall"] < floor:
        raise ValueError("residual-direct anchor does not meet the recall floor")
    feasible = [point for point in points if point["recall"] >= floor]
    if len(feasible) == 1:
        raise ValueError("no threshold case meets the recall floor")
    frozen = dict(config)
    frozen["matched_cases"] = [
        {**point["case"], "quality_recall": point["recall"],
         "target_recall": floor, "matched": True}
        for point in feasible]
    frozen["quality_points"] = points
    frozen["quality_selection_scope"] = "development+selection"
    frozen["quality_selection_query_count"] = len(expected_ids)
    frozen["source_quality_manifest"] = str(Path(quality_path).resolve())
    frozen["source_quality_sha256"] = sha256(quality_path)
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frozen = freeze(args.quality, args.config)
    write_json(args.output, frozen)
    write_json(Path(args.output).with_suffix(".complete.json"), {
        "quality_sha256": sha256(args.quality),
        "config_sha256": sha256(args.config),
        "output_sha256": sha256(args.output),
    })
    print(f"qualified residual cases: {len(frozen['matched_cases'])}")


if __name__ == "__main__":
    main()
