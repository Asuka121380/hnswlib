#!/usr/bin/env python3
"""Validate and aggregate independent real-data trace shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-queries", required=True, type=int)
    parser.add_argument("--expected-shards", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard_root = args.run_dir / "raw" / "shards"
    shard_dirs = sorted(path for path in shard_root.glob("part-*") if path.is_dir())
    if len(shard_dirs) != args.expected_shards:
        raise RuntimeError(f"Expected {args.expected_shards} shards, found {len(shard_dirs)}")

    metadata: list[dict] = []
    query_frames: list[pd.DataFrame] = []
    dco_paths: list[Path] = []
    edge_paths: list[Path] = []
    for shard in shard_dirs:
        required = [
            shard / "complete.json",
            shard / "metadata.json",
            shard / "query_stats.csv",
            shard / "dco_trace.csv",
            shard / "edge_directions.csv",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"Incomplete shard {shard.name}: {missing}")
        with (shard / "complete.json").open(encoding="utf-8") as handle:
            completion = json.load(handle)
        if completion.get("status") != "complete":
            raise RuntimeError(f"Shard {shard.name} is not complete")
        with (shard / "metadata.json").open(encoding="utf-8") as handle:
            metadata.append(json.load(handle))
        query_frames.append(pd.read_csv(shard / "query_stats.csv"))
        dco_paths.append(shard / "dco_trace.csv")
        edge_paths.append(shard / "edge_directions.csv")

    run_ids = {item["run_id"] for item in metadata}
    datasets = {item["dataset"] for item in metadata}
    parameter_keys = (
        "k", "ef_search", "M", "ef_construction", "seed",
        "dco_sample_modulus", "dco_sample_remainder",
    )
    parameter_sets = {tuple(item[key] for key in parameter_keys) for item in metadata}
    if len(run_ids) != 1 or len(datasets) != 1 or len(parameter_sets) != 1:
        raise RuntimeError("Shard metadata does not describe one consistent experiment")

    query = pd.concat(query_frames, ignore_index=True).sort_values("query_id")
    if len(query) != args.expected_queries:
        raise RuntimeError(f"Expected {args.expected_queries} query rows, found {len(query)}")
    if query["query_id"].duplicated().any():
        raise RuntimeError("Duplicate query IDs found across shards")
    expected_ids = list(range(int(query["query_id"].min()), int(query["query_id"].min()) + args.expected_queries))
    if query["query_id"].astype(int).tolist() != expected_ids:
        raise RuntimeError("Query IDs are not a complete contiguous range")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    query.to_csv(args.output_dir / "query_stats.csv", index=False)
    query.to_parquet(args.output_dir / "query_stats.parquet", index=False)

    dco_count = 0
    per_query_counts: dict[int, int] = {}
    dco_writer: pq.ParquetWriter | None = None
    try:
        for path in dco_paths:
            reader = pacsv.open_csv(path, read_options=pacsv.ReadOptions(block_size=16 * 1024 * 1024))
            for batch in reader:
                table = pa.Table.from_batches([batch])
                if dco_writer is None:
                    dco_writer = pq.ParquetWriter(args.output_dir / "dco_trace.parquet", table.schema, compression="zstd")
                dco_writer.write_table(table)
                dco_count += len(table)
                modulus = int(metadata[0]["dco_sample_modulus"])
                remainder = int(metadata[0]["dco_sample_remainder"])
                if modulus > 1:
                    query_ids = table.column("query_id").to_numpy(zero_copy_only=False).astype(np.uint64)
                    dco_indices = table.column("dco_index").to_numpy(zero_copy_only=False).astype(np.uint64)
                    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
                    constant = np.uint64(0x9E3779B97F4A7C15)
                    values = np.uint64(int(metadata[0]["seed"])) ^ ((query_ids + constant) & mask)
                    values ^= (dco_indices + constant + ((values << np.uint64(6)) & mask) + (values >> np.uint64(2))) & mask
                    values = (values + constant) & mask
                    values = ((values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & mask
                    values = ((values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & mask
                    values ^= values >> np.uint64(31)
                    if not np.all(values % np.uint64(modulus) == np.uint64(remainder % modulus)):
                        raise RuntimeError(f"DCO rows in {path} violate deterministic sampling")
                if int(metadata[0]["dco_sample_modulus"]) == 1:
                    ids = table.column("query_id").to_numpy(zero_copy_only=False).astype(np.int64)
                    unique, counts = np.unique(ids, return_counts=True)
                    for query_id, count in zip(unique, counts):
                        per_query_counts[int(query_id)] = per_query_counts.get(int(query_id), 0) + int(count)
    finally:
        if dco_writer is not None:
            dco_writer.close()
    if dco_writer is None:
        raise RuntimeError("No DCO rows were found")
    expected_sampled_rows = int(query["trace_records_written"].sum())
    if dco_count != expected_sampled_rows:
        raise RuntimeError(
            f"Sampled DCO rows do not reconcile with query summaries: {dco_count} != {expected_sampled_rows}"
        )
    if int(metadata[0]["dco_sample_modulus"]) == 1:
        actual = np.array([per_query_counts.get(int(query_id), 0) for query_id in query["query_id"]])
        if not np.array_equal(actual, query["n_dist"].to_numpy()):
            raise RuntimeError("Full-trace DCO rows do not reconcile with query summaries")

    edge_count = 0
    edge_writer: pq.ParquetWriter | None = None
    try:
        for path in edge_paths:
            reader = pacsv.open_csv(path, read_options=pacsv.ReadOptions(block_size=16 * 1024 * 1024))
            for batch in reader:
                table = pa.Table.from_batches([batch])
                if edge_writer is None:
                    edge_writer = pq.ParquetWriter(args.output_dir / "edge_directions.parquet", table.schema, compression="zstd")
                edge_writer.write_table(table)
                edge_count += len(table)
    finally:
        if edge_writer is not None:
            edge_writer.close()
    if edge_writer is None:
        raise RuntimeError("No edge samples were found")
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **metadata[0],
                "query_count": args.expected_queries,
                "shard_count": args.expected_shards,
                "dco_record_count": dco_count,
                "edge_sample_count": edge_count,
            },
            handle,
            indent=2,
        )
    print(f"aggregate_shards_ok queries={len(query)} dco={dco_count} edges={edge_count}")


if __name__ == "__main__":
    main()
