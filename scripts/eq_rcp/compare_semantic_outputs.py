#!/usr/bin/env python3
"""Compare Stage-0 feature-off and skeleton-on semantic outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


TIMING_COLUMNS = {
    "baseline_query_latency_ns",
    "trace_query_latency_ns",
    "exact_distance_time_ns",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def normalized_query_stats(path: Path) -> bytes:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        fields = [field for field in reader.fieldnames if field not in TIMING_COLUMNS]
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            writer.writerow({field: row[field] for field in fields})
    return buffer.getvalue().encode("utf-8")


def normalized_metadata(path: Path) -> bytes:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    value.pop("build_latency_ns", None)
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def compare(
    name: str,
    left: bytes,
    right: bytes,
    records: List[Dict[str, object]],
) -> bool:
    equal = left == right
    records.append(
        {
            "artifact": name,
            "equal": equal,
            "off_sha256": sha256(left),
            "on_sha256": sha256(right),
            "off_bytes": len(left),
            "on_bytes": len(right),
        }
    )
    return equal


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--off-probe", type=Path, required=True)
    parser.add_argument("--on-probe", type=Path, required=True)
    parser.add_argument("--off-trace-dir", type=Path, required=True)
    parser.add_argument("--on-trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    records: List[Dict[str, object]] = []
    equal = True
    equal &= compare(
        "semantic_probe",
        read_bytes(args.off_probe),
        read_bytes(args.on_probe),
        records,
    )
    equal &= compare(
        "metadata_without_build_latency",
        normalized_metadata(args.off_trace_dir / "metadata.json"),
        normalized_metadata(args.on_trace_dir / "metadata.json"),
        records,
    )
    equal &= compare(
        "query_stats_without_timing",
        normalized_query_stats(args.off_trace_dir / "query_stats.csv"),
        normalized_query_stats(args.on_trace_dir / "query_stats.csv"),
        records,
    )
    for filename in ("dco_trace.csv", "edge_directions.csv"):
        equal &= compare(
            filename,
            read_bytes(args.off_trace_dir / filename),
            read_bytes(args.on_trace_dir / filename),
            records,
        )

    result = {
        "format": "eq_rcp_stage0_semantic_comparison_v1",
        "status": "PASS" if equal else "FAIL",
        "timing_columns_excluded": sorted(TIMING_COLUMNS),
        "artifacts": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not equal:
        raise SystemExit("Stage-0 semantic outputs differ")
    print("eq_rcp_stage0_semantic_comparison_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
