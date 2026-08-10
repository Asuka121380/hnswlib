#!/usr/bin/env python3
"""Freeze a query-grouped manifest for an existing operational trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


REQUIRED_COLUMNS = (
    "query_id",
    "current_node_id",
    "candidate_id",
    "current_squared_distance",
    "threshold",
    "shadow_exact_squared_distance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--source-metadata", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    command = [
        "git",
        "-c",
        f"safe.directory={repo.as_posix()}",
        "-C",
        str(repo),
        *args,
    ]
    return subprocess.check_output(command, text=True).strip()


def query_bucket(seed: int, query_id: int) -> float:
    payload = f"{seed}:{query_id}".encode("ascii")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return value / float(1 << 64)


def main() -> int:
    args = parse_args()
    if not args.records.is_file():
        raise SystemExit(f"records file does not exist: {args.records}")
    if not args.repo.is_dir():
        raise SystemExit(f"repo does not exist: {args.repo}")
    if not (0.0 < args.train_fraction < 1.0):
        raise SystemExit("train-fraction must be in (0,1)")
    if not (0.0 < args.validation_fraction < 1.0):
        raise SystemExit("validation-fraction must be in (0,1)")
    if args.train_fraction + args.validation_fraction >= 1.0:
        raise SystemExit("train + validation fractions must be <1")

    parquet = pq.ParquetFile(args.records)
    names = set(parquet.schema.names)
    missing = sorted(set(REQUIRED_COLUMNS) - names)
    if missing:
        raise SystemExit(f"records are missing required columns: {missing}")
    table = pq.read_table(args.records, columns=list(REQUIRED_COLUMNS))
    arrays = {name: table[name].to_numpy(zero_copy_only=False) for name in REQUIRED_COLUMNS}
    query_ids = np.asarray(arrays["query_id"], dtype=np.int64)
    unique_queries = sorted(int(value) for value in np.unique(query_ids))

    split = {"train": [], "validation": [], "test": []}
    for query_id in unique_queries:
        bucket = query_bucket(args.seed, query_id)
        if bucket < args.train_fraction:
            split["train"].append(query_id)
        elif bucket < args.train_fraction + args.validation_fraction:
            split["validation"].append(query_id)
        else:
            split["test"].append(query_id)

    finite_columns = (
        "current_squared_distance",
        "threshold",
        "shadow_exact_squared_distance",
    )
    finite_mask = np.ones(table.num_rows, dtype=bool)
    for name in finite_columns:
        finite_mask &= np.isfinite(np.asarray(arrays[name], dtype=np.float64))
    nonnegative_mask = (
        np.asarray(arrays["current_squared_distance"], dtype=np.float64) >= 0.0
    ) & (np.asarray(arrays["threshold"], dtype=np.float64) >= 0.0) & (
        np.asarray(arrays["shadow_exact_squared_distance"], dtype=np.float64) >= 0.0
    )
    eligible = finite_mask & nonnegative_mask
    oracle = np.asarray(arrays["shadow_exact_squared_distance"], dtype=np.float64) > np.asarray(
        arrays["threshold"], dtype=np.float64
    )

    status = git(args.repo, "status", "--porcelain")
    manifest = {
        "format": "etv1_frozen_operational_manifest",
        "format_version": 1,
        "records": {
            "path": str(args.records.resolve()),
            "sha256": sha256(args.records),
            "bytes": args.records.stat().st_size,
            "rows": table.num_rows,
            "row_groups": parquet.metadata.num_row_groups,
            "schema": parquet.schema.names,
            "required_columns": list(REQUIRED_COLUMNS),
            "finite_nonnegative_rows": int(np.count_nonzero(eligible)),
            "oracle_prunable_rows": int(np.count_nonzero(eligible & oracle)),
            "oracle_prunable_fraction": float(np.mean(oracle[eligible])) if np.any(eligible) else math.nan,
        },
        "queries": {
            "count": len(unique_queries),
            "minimum": min(unique_queries) if unique_queries else None,
            "maximum": max(unique_queries) if unique_queries else None,
        },
        "repo": {
            "path": str(args.repo.resolve()),
            "commit": git(args.repo, "rev-parse", "HEAD"),
            "branch": git(args.repo, "branch", "--show-current"),
            "working_tree_dirty": bool(status),
            "status_porcelain": status.splitlines(),
        },
        "source_metadata": None,
        "fresh_query_status": "development_only_existing_queries",
    }
    if args.source_metadata:
        manifest["source_metadata"] = {
            "path": str(args.source_metadata.resolve()),
            "sha256": sha256(args.source_metadata),
            "json": json.loads(args.source_metadata.read_text(encoding="utf-8")),
        }

    split_manifest = {
        "format": "etv1_query_split_manifest",
        "format_version": 1,
        "split_unit": "query_id",
        "seed": args.seed,
        "algorithm": "sha256(seed:query_id) first uint64 / 2^64",
        "fractions": {
            "train": args.train_fraction,
            "validation": args.validation_fraction,
            "test": 1.0 - args.train_fraction - args.validation_fraction,
        },
        "query_ids": split,
        "counts": {name: len(values) for name, values in split.items()},
        "records_sha256": manifest["records"]["sha256"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "frozen_baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "query_split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": manifest, "split_counts": split_manifest["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

