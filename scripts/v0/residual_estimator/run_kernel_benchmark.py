"""Run paired residual setup, correction, metadata and exact cost blocks."""

import argparse
import statistics
import subprocess
from pathlib import Path

from common import mark_complete, read_json, sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config, selection = read_json(args.config), read_json(args.selection)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": 1, "cases": []}
    for bits in selection["selected_bits"]:
        for working_set, edges in (("l1", 256), ("larger_than_l3", 65536)):
            target = output / f"k{bits}-{working_set}.json"
            subprocess.run([args.runner, "--output", str(target),
                            "--dimension", str(config["dimension"]),
                            "--bits", str(bits), "--edges", str(edges),
                            "--queries", "100", "--blocks", str(config["blocks"])],
                           check=True)
            data = read_json(target)
            blocks = data["blocks"]
            if len(blocks) != config["blocks"]:
                raise ValueError("missing benchmark block")
            case = {"bits": bits, "working_set": working_set,
                    "edges": edges, "stride": data["stride"],
                    "raw_sha256": sha256(target)}
            for field in ("setup_ns", "correction_total_ns",
                          "metadata_total_ns", "exact_total_ns"):
                values = [block[field] / (edges if field != "setup_ns" else 1)
                          for block in blocks]
                case[field.replace("total_", "") + "_median"] = statistics.median(values)
                case[field.replace("total_", "") + "_mean"] = statistics.mean(values)
            summary["cases"].append(case)
    summary["reference_warning"] = (
        "Synthetic working sets isolate components; active matched-recall QPS "
        "is the end-to-end performance measurement.")
    write_json(output / "manifest.json", summary)
    mark_complete(output, {"config": sha256(args.config),
                           "selection": sha256(args.selection)},
                  {"manifest": sha256(output / "manifest.json")})


if __name__ == "__main__":
    main()
