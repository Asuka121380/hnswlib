from __future__ import annotations
import csv
import math
import struct
from pathlib import Path
import numpy as np
from .common import load, verify_file, digest

def read_ids(path, count=None):
    words = Path(path).read_text(encoding="utf-8").split()
    if not words or any(not w.isascii() or not w.isdigit() for w in words):
        raise ValueError("query IDs must be a nonempty list of unsigned integers")
    ids = [int(w) for w in words]
    if len(set(ids)) != len(ids) or any(i >= (1 << 63) or (count is not None and i >= count) for i in ids):
        raise ValueError("duplicate or out-of-range query ID")
    return ids

def write_ids(path, ids):
    ids = list(ids)
    if not ids or len(set(ids)) != len(ids) or any(int(i) != i or i < 0 for i in ids):
        raise ValueError("invalid query IDs")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(f"{int(i)}\n" for i in ids), encoding="ascii")

def read_vecs(path, kind="f", dimension=None):
    path = Path(path)
    with path.open("rb") as stream:
        first = stream.read(4)
    if len(first) != 4:
        raise ValueError("empty vector file")
    width = struct.unpack("<i", first)[0]
    if width <= 0 or (dimension is not None and width != dimension) or path.stat().st_size % (4*(width+1)):
        raise ValueError("vector file shape mismatch")
    raw = np.memmap(path, dtype="<i4", mode="r").reshape(-1, width+1)
    for start in range(0, len(raw), 65536):
        block = raw[start:start+65536]
        if not np.all(block[:, 0] == width):
            raise ValueError("inconsistent per-row dimension")
        if kind == "f" and not np.isfinite(block[:, 1:].view("<f4")).all():
            raise ValueError("nonfinite vector")
    return raw[:, 1:].view("<f4") if kind == "f" else raw[:, 1:]

def write_vecs(path, rows, kind="f"):
    rows = np.asarray(rows, dtype="<f4" if kind == "f" else "<i4")
    if rows.ndim != 2 or not rows.shape[0] or not rows.shape[1]:
        raise ValueError("empty vector matrix")
    with Path(path).open("wb") as out:
        for first in range(0, len(rows), 4096):
            block = rows[first:first+4096]
            packed = np.empty((len(block), rows.shape[1]+1), dtype="<i4")
            packed[:, 0] = rows.shape[1]
            packed[:, 1:] = block.view("<i4")
            packed.tofile(out)

def read_truth(path, ids, k, rows_path=None):
    matrix = read_vecs(path, "i")
    if matrix.shape[1] < k:
        raise ValueError("GT width is smaller than k")
    mapping = load(rows_path) if rows_path else {str(i): i for i in ids}
    if rows_path and len(set(mapping.values())) != len(mapping):
        raise ValueError("GT row mapping is not one-to-one")
    truth = {}
    for q in ids:
        row = mapping.get(str(q))
        if isinstance(row, bool) or not isinstance(row, int) or row < 0 or row >= len(matrix):
            raise ValueError(f"GT row missing/out of range for query {q}")
        labels = list(map(int, matrix[row, :k]))
        if min(labels) < 0 or len(set(labels)) != k:
            raise ValueError("invalid GT labels")
        truth[q] = set(labels)
    return truth

def evaluate_results(path, truth, k, repeats=1, allowed_labels=None):
    expected = set(truth)
    result = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            r, q, rank = int(row.get("repeat_id", 0)), int(row["query_id"]), int(row["rank"])
            label, distance = int(row["label"]), float(row["distance"])
            if r < 0 or r >= repeats or q not in expected or rank < 0 or rank >= k:
                raise ValueError("unexpected repeat/query/rank")
            if label < 0 or (allowed_labels is not None and label not in allowed_labels):
                raise ValueError("returned label outside canonical base labels")
            if not math.isfinite(distance) or distance < 0:
                raise ValueError("invalid returned squared distance")
            values = result.setdefault((r, q), {})
            if rank in values:
                raise ValueError("duplicate result rank")
            values[rank] = (label, distance)
    if set(result) != {(r, q) for r in range(repeats) for q in expected}:
        raise ValueError("returned query/repeat coverage mismatch")
    per_query = []
    stable = True
    for q in sorted(expected):
        reference = None
        scores = []
        for r in range(repeats):
            rows = result[r, q]
            if set(rows) != set(range(k)):
                raise ValueError("missing result rank")
            ordered = [rows[i] for i in range(k)]
            if len({x[0] for x in ordered}) != k:
                raise ValueError("duplicate result label")
            if any(ordered[i][1] > ordered[i+1][1] for i in range(k-1)):
                raise ValueError("unsorted returned distances")
            if reference is None:
                reference = ordered
            elif ordered != reference:
                stable = False
            scores.append(len({x[0] for x in ordered} & truth[q])/k)
        per_query.append({"query_id": q, "recall": float(np.mean(scores)), "topk_sha256": digest(reference)})
    return {"recall_at_10" if k == 10 else f"recall_at_{k}": float(np.mean([v["recall"] for v in per_query])),
            "recall_at_k": float(np.mean([v["recall"] for v in per_query])),
            "per_query": per_query, "results_stable_across_repeats": stable,
            "result_query_count": len(expected), "result_repeat_count": repeats}

def split_contract(dataset, split, k, verify=True):
    spec = dataset["splits"][split]
    if verify:
        for entry in (dataset["query_pool"], spec["ids"], spec["gt"], spec["gt_rows"]):
            verify_file(entry)
    ids = read_ids(spec["ids"]["path"], dataset["n_queries"])
    truth = read_truth(spec["gt"]["path"], ids, k, spec["gt_rows"]["path"])
    return ids, truth
