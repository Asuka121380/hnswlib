"""Run finite active quality or timed QPS cases with explicit manifests."""

import argparse
import random
import struct
import subprocess
from pathlib import Path

from common import mark_complete, read_json, sha256, write_json


def quality_cases(config: dict, selection: dict) -> list[dict]:
    cases = []
    for ef in config["ef_search"]:
        cases.append({"method": "baseline", "ef": ef})
        cases.extend({"method": "approx-no-retry", "ef": ef, "beta": beta}
                     for beta in config["beta_values"])
        cases.append({"method": "residual-direct", "ef": ef, "theta": 1.0})
        chosen = next(row for row in selection["choices"]
                      if row["bits"] == selection["selected_bits"][0])
        cases.append({"method": "residual-threshold", "ef": ef,
                      "theta": chosen["primary_theta"]})
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--companion", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", required=True, choices=("quality", "qps"))
    args = parser.parse_args()
    config, contract, selection = (read_json(args.config), read_json(args.contract),
                                   read_json(args.selection))
    assets = contract["assets"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    quality_query_ids = None
    if args.stage == "quality":
        if "dataset_config" not in assets:
            raise ValueError("quality stage requires dataset_config")
        split = read_json(contract["split_path"])
        development = set(split["development"])
        selection_ids = set(split["selection"])
        audit = set(split["audit"])
        if not development or not selection_ids or development & selection_ids or \
                (development | selection_ids) & audit:
            raise ValueError("invalid quality query split")
        quality_query_ids = output / "quality-query-ids.txt"
        quality_query_ids.write_text(
            "".join(f"{query_id}\n" for query_id in sorted(development | selection_ids)),
            encoding="utf-8")
        cases = quality_cases(config, selection)
    else:
        cases = config.get("matched_cases", [])
        if not cases:
            raise ValueError("QPS requires frozen matched_cases from quality selection")
    with Path(args.companion).open("rb") as source:
        header = source.read(24)
    if len(header) != 24 or header[:8] != b"V0RES001":
        raise ValueError("invalid companion header")
    companion_bits = struct.unpack_from("<I", header, 20)[0]
    if args.stage == "quality" and companion_bits != selection["selected_bits"][0]:
        raise ValueError("companion k differs from frozen primary selection")
    runs = []
    schedule = []
    blocks = 1 if args.stage == "quality" else config["blocks"]
    for block in range(blocks):
        order = list(enumerate(cases))
        if args.stage == "qps":
            random.Random(20260919 + block).shuffle(order)
        schedule.extend((block, case_id, case) for case_id, case in order)
    for block, case_id, case in schedule:
        case_dir = output / f"block-{block:02d}-case-{case_id:03d}"
        case_dir.mkdir(exist_ok=True)
        method = case["method"]
        if args.stage == "quality":
            mode = "correctness" if method == "baseline" else method
            command = [args.runner, "--dataset-config", assets["dataset_config"]["path"],
                       "--index-path", assets["index"]["path"],
                       "--sidecar-path", assets["sidecar"]["path"],
                       "--output-dir", str(case_dir), "--mode", mode,
                       "--query-id-file", str(quality_query_ids),
                       "--ef-search", str(case["ef"]), "--k", str(config["k"])]
            if method == "approx-no-retry":
                command += ["--approx-beta", str(case["beta"])]
            if method.startswith("residual"):
                command += ["--companion-path", args.companion,
                            "--residual-theta", str(case["theta"])]
            result_path = case_dir / "summary.json"
        else:
            result_path = case_dir / "qps.json"
            command = [args.runner, "--method", method,
                       "--index-path", assets["index"]["path"],
                       "--query-path", assets["queries"]["path"],
                       "--dimension", str(contract["config"]["dimension"]),
                       "--output", str(result_path),
                       "--latency-records-output", str(case_dir / "latency.csv"),
                       "--result-records-output", str(case_dir / "results.csv"),
                       "--query-start", str(config["query_start"]),
                       "--query-count", str(config["query_count"]),
                       "--ef-search", str(case["ef"]), "--k", str(config["k"]),
                       "--warmup-queries", str(config["warmup_queries"]),
                       "--repeats", str(config["repeats"])]
            if method != "baseline":
                command += ["--sidecar-path", assets["sidecar"]["path"]]
            if method == "approx-no-retry":
                command += ["--beta", str(case["beta"])]
            if method.startswith("residual"):
                command += ["--companion-path", args.companion,
                            "--theta", str(case["theta"])]
        subprocess.run(command, check=True)
        result = read_json(result_path)
        if args.stage == "quality" and result.get("status") != "valid":
            raise ValueError(f"invalid quality case: {case_dir}")
        runs.append({"block": block, "case": case, "result": str(result_path.resolve()),
                     "sha256": sha256(result_path), "command": command})
    manifest = {"schema_version": 1, "stage": args.stage,
                "config": str(Path(args.config).resolve()),
                "contract": str(Path(args.contract).resolve()),
                "selection": str(Path(args.selection).resolve()),
                "companion_sha256": sha256(args.companion), "runs": runs}
    if quality_query_ids is not None:
        manifest["quality_selection_scope"] = "development+selection"
        manifest["query_id_file_sha256"] = sha256(quality_query_ids)
    write_json(output / "manifest.json", manifest)
    mark_complete(output, {"config": sha256(args.config),
                           "selection": sha256(args.selection)},
                  {"manifest": sha256(output / "manifest.json")})


if __name__ == "__main__":
    main()
