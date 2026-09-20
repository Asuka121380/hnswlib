"""Freeze recall-matched cases at five baseline anchors for paired QPS."""

import argparse
import math
from pathlib import Path

from common import read_json, sha256, write_json
from select_targeted_qps import checked_quality


def freeze(run_root: str, config_path: str) -> dict:
    config = read_json(config_path)
    tolerance = config.get("recall_tolerance")
    if (not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool)
            or not math.isfinite(tolerance) or not 0 < tolerance < 1):
        raise ValueError("invalid recall tolerance")
    sources = config.get("quality_sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("missing curve quality sources")
    root = Path(run_root)
    paths = {}
    for name, directory in sources.items():
        if not isinstance(directory, str) or Path(directory).name != directory:
            raise ValueError("quality source must be a run-root directory name")
        paths[name] = root / directory / "manifest.json"
    loaded = {name: checked_quality(str(path)) for name, path in paths.items()}
    identities = [source[2] for source in loaded.values()]
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("curve quality sources used different frozen inputs")
    first = next(iter(loaded.values()))[0]
    contract = read_json(first["contract"])
    split = read_json(contract["split_path"])
    expected_count = len(set(split["development"]) | set(split["selection"]))
    if expected_count != config.get("quality_query_count"):
        raise ValueError("unexpected quality query count")

    anchors = config.get("anchors")
    if not isinstance(anchors, list) or len(anchors) != 5:
        raise ValueError("curve requires five baseline anchors")
    selected = []
    anchor_efs = set()
    for anchor in anchors:
        baseline_spec = anchor["baseline"]
        if baseline_spec.get("method") != "baseline" or \
                set(baseline_spec) != {"source", "method", "ef"}:
            raise ValueError("invalid baseline anchor")
        baseline_ef = baseline_spec["ef"]
        if (not isinstance(baseline_ef, int) or isinstance(baseline_ef, bool)
                or baseline_ef <= 0 or baseline_ef in anchor_efs):
            raise ValueError("duplicate or invalid baseline ef")
        anchor_efs.add(baseline_ef)

        def lookup(spec: dict) -> tuple[dict, float]:
            source = spec["source"]
            if source not in loaded:
                raise ValueError("unknown curve quality source")
            case = {key: value for key, value in spec.items() if key != "source"}
            key = tuple(sorted(case.items()))
            if key not in loaded[source][1]:
                raise ValueError(f"missing curve quality case: {spec}")
            return case, loaded[source][1][key]

        baseline, target = lookup(baseline_spec)
        candidates = anchor.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("missing curve candidates")
        cases = [(baseline, target)] + [lookup(spec) for spec in candidates]
        methods = [case["method"] for case, _ in cases]
        if (methods.count("baseline") != 1 or
                methods.count("approx-no-retry") != 1 or
                methods.count("residual-direct") != 1 or
                methods.count("residual-threshold") < 1 or
                len({tuple(sorted(case.items())) for case, _ in cases}) != len(cases)):
            raise ValueError("curve anchor needs unique baseline, PQ, direct and threshold")
        for case, recall in cases:
            if abs(recall - target) > tolerance + 1e-12:
                raise ValueError(f"case outside recall window at ef={baseline_ef}: {case}")
            selected.append({**case, "anchor_ef": baseline_ef,
                             "target_recall": target,
                             "quality_recall": recall, "matched": True})

    frozen = dict(config)
    frozen["matched_cases"] = selected
    frozen["quality_query_ids_sha256"] = identities[0]["query_ids_sha256"]
    frozen["companion_sha256"] = identities[0]["companion_sha256"]
    frozen["source_quality"] = {
        name: {"manifest": str(path.resolve()), "sha256": sha256(path)}
        for name, path in paths.items()}
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frozen = freeze(args.run_root, args.config)
    write_json(args.output, frozen)
    write_json(Path(args.output).with_suffix(".complete.json"), {
        "output_sha256": sha256(args.output),
        "source_quality": frozen["source_quality"],
    })
    print(f"curve QPS: {len(frozen['anchors'])} anchors, "
          f"{len(frozen['matched_cases'])} matched cases")
    for case in frozen["matched_cases"]:
        print(f"anchor_ef={case['anchor_ef']} method={case['method']} "
              f"ef={case['ef']} recall={case['quality_recall']:.5f}")


if __name__ == "__main__":
    main()
