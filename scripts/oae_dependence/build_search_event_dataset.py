#!/usr/bin/env python3
"""Bind a baseline dco_trace.csv to its frozen original learn-query IDs."""

from __future__ import annotations

import argparse
import csv
import json
from array import array
from pathlib import Path

import numpy as np

from oae_dependence_core import require, sha256_file, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-csv", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"refusing to overwrite: {args.output}")
    manifest = json.loads(args.query_manifest.read_text(encoding="utf-8"))
    offset, count = int(manifest["source_offset"]), int(manifest["count"])
    query_id, current, neighbor = array("q"), array("q"), array("q")
    with args.trace_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"query_id", "current_node_label", "neighbor_label", "geometry_valid"}
        require(reader.fieldnames is not None and required.issubset(reader.fieldnames), "trace CSV lacks required columns")
        for row in reader:
            if row["geometry_valid"] not in ("1", "true", "True"):
                continue
            local = int(row["query_id"])
            require(0 <= local < count, "trace query id exceeds frozen local split")
            c, v = int(row["current_node_label"]), int(row["neighbor_label"])
            if c == v:
                continue
            query_id.append(offset + local)
            current.append(c)
            neighbor.append(v)
    q = np.frombuffer(query_id, dtype=np.int64).copy()
    c = np.frombuffer(current, dtype=np.int64).copy()
    v = np.frombuffer(neighbor, dtype=np.int64).copy()
    require(len(q) > 0, "no valid search events were collected")
    observed = np.unique(q)
    expected = np.arange(offset, offset + count)
    require(np.array_equal(observed, expected), "one or more frozen queries has no valid search event")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, query_id=q, current_label=c, neighbor_label=v)
    report = {"status": "PASS", "event_count": len(q), "query_count": count,
              "query_offset": offset, "trace_csv_sha256": sha256_file(args.trace_csv),
              "dataset_sha256": sha256_file(args.output)}
    write_json(Path(str(args.output) + ".manifest.json"), report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
