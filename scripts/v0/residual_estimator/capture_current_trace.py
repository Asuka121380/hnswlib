"""Capture all eligible first-visit events for fixed development query IDs."""

import argparse
import hashlib
import subprocess
from pathlib import Path

from common import mark_complete, read_json, sha256, write_json


def spans(ids: list[int]) -> list[tuple[int, int]]:
    result = []
    for query_id in sorted(ids):
        if result and query_id == result[-1][0] + result[-1][1]:
            first, length = result[-1]
            result[-1] = (first, length + 1)
        else:
            result.append((query_id, 1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config, contract = read_json(args.config), read_json(args.contract)
    assets = contract["assets"]
    if "dataset_config" not in assets:
        raise ValueError("A2 requires an explicit dataset_config asset")
    development = read_json(contract["split_path"])["development"]
    ordered = sorted(development, key=lambda value: hashlib.sha256(
        f"{contract['config']['split_seed']}:{value}".encode()).digest())
    selected = sorted(ordered[:config["development_queries"]])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for ef in config["ef_search"]:
        for first, count in spans(selected):
            target = output / f"ef{ef}-q{first}-{first + count - 1}"
            command = [args.runner, "--dataset-config", assets["dataset_config"]["path"],
                       "--index-path", assets["index"]["path"],
                       "--sidecar-path", assets["sidecar"]["path"],
                       "--output-dir", str(target), "--mode", "shadow",
                       "--query-start", str(first), "--query-count", str(count),
                       "--k", str(config["k"]), "--ef-search", str(ef),
                       "--shadow-sample-modulus", "1",
                       "--shadow-sample-remainder", "0"]
            subprocess.run(command, check=True)
            summary = read_json(target / "summary.json")
            if summary.get("status") != "valid":
                raise ValueError(f"invalid shadow run: {target}")
            runs.append({"ef": ef, "query_start": first, "query_count": count,
                         "output": str(target.resolve()),
                         "summary_sha256": sha256(target / "summary.json")})
    manifest = {"schema_version": 1, "query_ids": selected,
                "stage_sequence_available": False, "runs": runs}
    write_json(output / "manifest.json", manifest)
    mark_complete(output, {"config": sha256(args.config),
                           "contract": sha256(args.contract)},
                  {"manifest": sha256(output / "manifest.json")})


if __name__ == "__main__":
    main()
