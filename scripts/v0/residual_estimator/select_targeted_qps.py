"""Freeze PQ and residual cases within baseline recall tolerance."""

import argparse
import math
from pathlib import Path

from common import read_json, sha256, write_json
from run_active_matrix import quality_cases
from select_residual_tuning import measured_recall


def checked_quality(path: str) -> tuple[dict, dict, dict]:
    manifest = read_json(path)
    directory = Path(path).parent
    complete = read_json(directory / "complete.json")
    if manifest.get("stage") != "quality" or \
            manifest.get("quality_selection_scope") != "development+selection" or \
            complete["outputs"]["manifest"] != sha256(path) or \
            complete["inputs"]["config"] != sha256(manifest["config"]):
        raise ValueError(f"incomplete frozen quality manifest: {path}")
    contract = read_json(manifest["contract"])
    split = read_json(contract["split_path"])
    expected = set(split["development"]) | set(split["selection"])
    if not expected or expected & set(split["audit"]):
        raise ValueError("invalid frozen quality split")
    query_hash = sha256(directory / "quality-query-ids.txt")
    if manifest["query_id_file_sha256"] != query_hash:
        raise ValueError("quality query IDs changed")
    if (directory / "quality-query-ids.txt").read_text(encoding="utf-8") != \
            "".join(f"{query_id}\n" for query_id in sorted(expected)):
        raise ValueError("quality query IDs differ from frozen split")
    cases = quality_cases(read_json(manifest["config"]), {})
    if [run["case"] for run in manifest["runs"]] != cases:
        raise ValueError("quality cases differ from their config")
    points = {}
    for run in manifest["runs"]:
        if sha256(run["result"]) != run["sha256"] or \
                read_json(run["result"]).get("status") != "valid":
            raise ValueError("invalid quality result or SHA")
        key = tuple(sorted(run["case"].items()))
        if key in points:
            raise ValueError("duplicate quality case")
        points[key] = measured_recall(run["result"], expected)
    identity = {"contract_sha256": sha256(manifest["contract"]),
                "query_ids_sha256": query_hash,
                "companion_sha256": manifest["companion_sha256"],
                "selection_sha256": sha256(manifest["selection"])}
    return manifest, points, identity


def freeze(active_path: str, supplement_path: str, fine_path: str,
           config_path: str) -> dict:
    config = read_json(config_path)
    sources = {name: path for name, path in
               (("active", active_path), ("supplement", supplement_path),
                ("fine", fine_path))}
    loaded = {name: checked_quality(path) for name, path in sources.items()}
    identities = [value[2] for value in loaded.values()]
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("quality runs used different split, companion, or contract")
    baseline_case = {"method": "baseline", "ef": config["baseline_ef"]}
    baseline_key = tuple(sorted(baseline_case.items()))
    baseline_recall = loaded["active"][1][baseline_key]
    tolerance = config["recall_tolerance"]
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool) or \
            not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("invalid recall tolerance")
    chosen = [("active", baseline_case, baseline_recall)]
    for candidate in config["candidates"]:
        source = candidate["source"]
        if source not in sources:
            raise ValueError("invalid quality candidate source")
        case = {key: value for key, value in candidate.items()
                if key != "source"}
        if case.get("method") not in ("approx-no-retry", "residual-threshold",
                                      "residual-direct"):
            raise ValueError("invalid QPS candidate method")
        recall = loaded[source][1][tuple(sorted(case.items()))]
        if abs(recall - baseline_recall) > tolerance + 1e-12:
            raise ValueError(f"candidate outside baseline recall window: {case}")
        chosen.append((source, case, recall))
    if len(chosen) != 5 or len({tuple(sorted(case.items()))
                                 for _, case, _ in chosen}) != 5 or \
            sorted(case["method"] for _, case, _ in chosen) != sorted(
                ["baseline", "approx-no-retry", "residual-direct",
                 "residual-threshold", "residual-threshold"]):
        raise ValueError("targeted QPS requires baseline, PQ, direct and two thresholds")
    frozen = dict(config)
    frozen["matched_cases"] = [
        {**case, "target_recall": baseline_recall,
         "quality_recall": recall, "matched": True}
        for _, case, recall in chosen]
    frozen["baseline_recall"] = baseline_recall
    frozen["recall_window"] = [baseline_recall - tolerance,
                               baseline_recall + tolerance]
    frozen["quality_query_ids_sha256"] = identities[0]["query_ids_sha256"]
    frozen["companion_sha256"] = identities[0]["companion_sha256"]
    frozen["source_quality"] = {
        name: {"manifest": str(Path(path).resolve()), "sha256": sha256(path)}
        for name, path in sources.items()}
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-quality", required=True)
    parser.add_argument("--supplement-quality", required=True)
    parser.add_argument("--fine-quality", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frozen = freeze(args.active_quality, args.supplement_quality,
                    args.fine_quality, args.config)
    write_json(args.output, frozen)
    write_json(Path(args.output).with_suffix(".complete.json"), {
        "output_sha256": sha256(args.output),
        "source_quality": frozen["source_quality"],
    })
    print(f"target baseline recall={frozen['baseline_recall']:.6f}; "
          f"QPS cases={len(frozen['matched_cases'])}")


if __name__ == "__main__":
    main()
