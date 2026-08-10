#!/usr/bin/env python3
"""Operational pivot-distance audit using raw fvecs and frozen trace IDs.

The script is ready for Gate B but intentionally fails closed when the raw
vectors are unavailable.  It does not require an HNSW index or V0 sidecar.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


class Fvecs:
    def __init__(self, path: Path, internal_to_label: np.ndarray | None = None):
        self.path = path
        raw = np.memmap(path, dtype=np.int32, mode="r")
        if raw.size == 0:
            raise ValueError(f"empty fvecs file: {path}")
        self.dimension = int(raw[0])
        if self.dimension <= 0:
            raise ValueError(f"invalid fvecs dimension: {self.dimension}")
        stride = self.dimension + 1
        if raw.size % stride:
            raise ValueError(f"truncated fvecs file: {path}")
        self.count = raw.size // stride
        records = raw.reshape(self.count, stride)
        if not np.all(records[:, 0] == self.dimension):
            raise ValueError(f"inconsistent dimensions in fvecs file: {path}")
        self._vectors = records[:, 1:].view(np.float32)
        self.internal_to_label = internal_to_label

    def take(self, ids: np.ndarray) -> np.ndarray:
        if self.internal_to_label is not None:
            if np.any(ids < 0) or np.any(ids >= self.internal_to_label.size):
                raise IndexError("internal HNSW ID is out of range")
            ids = self.internal_to_label[ids]
        if np.any(ids < 0) or np.any(ids >= self.count):
            raise IndexError(f"fvecs id outside [0,{self.count}): {self.path}")
        return np.asarray(self._vectors[ids], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--base-fvecs", type=Path, required=True)
    parser.add_argument("--query-fvecs", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--internal-to-label", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser.parse_args()


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        label: float(np.quantile(values, quantile))
        for label, quantile in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9), ("p99", 0.99))
    }


def main() -> int:
    args = parse_args()
    for path in (args.records, args.base_fvecs, args.query_fvecs, args.split_manifest, args.internal_to_label):
        if not path.is_file():
            raise SystemExit(f"required input is unavailable: {path}")
    internal_to_label = np.load(args.internal_to_label, allow_pickle=False).astype(np.int64, copy=False)
    if internal_to_label.ndim != 1:
        raise SystemExit("internal-to-label mapping must be one-dimensional")
    base = Fvecs(args.base_fvecs, internal_to_label)
    queries = Fvecs(args.query_fvecs)
    if base.dimension != queries.dimension:
        raise SystemExit("base/query dimensions differ")
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    train_ids = np.asarray(split["query_ids"]["train"], dtype=np.int64)
    query_mean = np.mean(queries.take(train_ids), axis=0)

    parquet = pq.ParquetFile(args.records)
    columns = ["query_id", "current_node_id", "edge_length", "direction_error"]
    strategies = {"zero": np.zeros(base.dimension), "query_mean": query_mean}
    ratios: dict[str, list[np.ndarray]] = {name: [] for name in strategies}
    current_norms: list[np.ndarray] = []
    rows = 0
    for batch in parquet.iter_batches(batch_size=args.batch_size, columns=columns):
        data = batch.to_pydict()
        qids = np.asarray(data["query_id"], dtype=np.int64)
        cids = np.asarray(data["current_node_id"], dtype=np.int64)
        q = queries.take(qids)
        c = base.take(cids)
        current = np.linalg.norm(q - c, axis=1)
        valid = np.isfinite(current) & (current > 0.0)
        current_norms.append(current[valid])
        for name, pivot in strategies.items():
            pivot_distance = np.linalg.norm(q - pivot, axis=1)
            ratios[name].append(pivot_distance[valid] / current[valid])
        rows += int(np.count_nonzero(valid))

    report = {
        "format": "etv1_pivot_audit",
        "format_version": 1,
        "rows": rows,
        "dimension": base.dimension,
        "base_count": int(base.count),
        "query_count": int(queries.count),
        "current_q_minus_c_norm": quantiles(np.concatenate(current_norms)),
        "pivot_factor": {
            name: quantiles(np.concatenate(chunks)) for name, chunks in ratios.items()
        },
        "interpretation": "factor below 1 improves the query-side radius; factor above 1 worsens it",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
