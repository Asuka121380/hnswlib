"""Select measured quality points for the finite paired QPS matrix."""

import argparse
from pathlib import Path

from common import read_json, sha256, write_json


def nearest(candidates: list[dict], target: float) -> dict:
    if not candidates:
        raise ValueError("missing method in quality matrix")
    return min(candidates, key=lambda item: (abs(item["recall"] - target),
                                              item["case"]["ef"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest, config = read_json(args.quality), read_json(args.config)
    if manifest["stage"] != "quality":
        raise ValueError("expected active quality manifest")
    points = []
    for run in manifest["runs"]:
        summary = read_json(run["result"])
        points.append({"case": run["case"],
                       "recall": summary["mean_v0_recall_at_k"],
                       "result": run["result"]})
    selected = []
    for target in config["target_recalls"]:
        exact = nearest([p for p in points if p["case"]["method"] == "baseline"], target)
        pq = nearest([p for p in points if p["case"]["method"] == "approx-no-retry"], target)
        residual = nearest([p for p in points if p["case"]["method"].startswith("residual")], target)
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
    frozen["source_quality_sha256"] = sha256(args.quality)
    write_json(args.output, frozen)
    write_json(Path(args.output).with_suffix(".complete.json"),
               {"quality_sha256": sha256(args.quality),
                "output_sha256": sha256(args.output)})


if __name__ == "__main__":
    main()
