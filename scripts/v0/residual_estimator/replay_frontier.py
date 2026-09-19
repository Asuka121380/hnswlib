"""Compare frozen beta and residual policies on identical sampled events."""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from common import mark_complete, read_json, sha256, write_json


def fvecs(path: str, dimension: int) -> np.ndarray:
    raw = np.memmap(path, dtype="u1", mode="r")
    stride = 4 + 4 * dimension
    if raw.size % stride:
        raise ValueError("query fvecs length mismatch")
    records = raw.reshape((-1, stride))
    headers = records[:, :4].copy().view("<i4").reshape(-1)
    if not np.all(headers == dimension):
        raise ValueError("query fvecs dimension mismatch")
    return records[:, 4:].copy().view("<f4").reshape((-1, dimension))


def evaluate(decision: np.ndarray, far: np.ndarray, ids: np.ndarray,
             split: str, policy: str, threshold: float,
             seed: int, bits: int) -> tuple[dict, list[dict]]:
    total = len(decision)
    tp_mask = decision & far
    fp_mask = decision & ~far
    tp = int(np.count_nonzero(tp_mask))
    fp = int(np.count_nonzero(fp_mask))
    near = int(np.count_nonzero(~far))
    unique_ids, inverse = np.unique(ids, return_inverse=True)
    query_tp = np.bincount(inverse, weights=tp_mask, minlength=len(unique_ids))
    query_fp = np.bincount(inverse, weights=fp_mask, minlength=len(unique_ids))
    query_n = np.bincount(inverse, minlength=len(unique_ids))
    row = {"split": split, "policy": policy, "threshold": threshold,
           "seed": seed, "bits": bits, "N": total, "TP": tp, "FP": fp,
           "FN": int(np.count_nonzero(far)) - tp,
           "TP_per_N": tp / total, "FP_per_N": fp / total,
           "FP_per_pruned": fp / max(1, tp + fp),
           "FP_per_near": fp / max(1, near),
           "sampled_query_exposure": float(np.mean(query_fp > 0))}
    per_query = [
        {"split": split, "policy": policy, "threshold": threshold,
         "seed": seed, "bits": bits, "query_id": int(query_id),
         "N": int(query_n[i]), "TP": int(query_tp[i]), "FP": int(query_fp[i])}
        for i, query_id in enumerate(unique_ids)
    ]
    return row, per_query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sketch", required=True)
    parser.add_argument("--split", default="development,selection")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dataset, sketch = read_json(args.dataset), read_json(args.sketch)
    contract = read_json(dataset["contract"])
    split_ids = read_json(contract["split_path"])
    dimension = contract["config"]["dimension"]
    queries = fvecs(contract["assets"]["queries"]["path"], dimension)
    events = pd.read_parquet(dataset["events_parquet"])
    if not events["query_id"].between(0, len(queries) - 1).all():
        raise ValueError("event query ID outside frozen query file")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frontier, query_counts = [], []
    splits = args.split.split(",")
    if not splits or any(value not in split_ids for value in splits):
        raise ValueError("unknown split")
    beta_grid = np.unique(np.r_[np.arange(1.0, 2.0001, 0.02),
                                [1.30, 1.40, 1.45, 1.50, 1.55]])
    theta_grid = np.arange(0.80, 1.4001, 0.02)
    for split in splits:
        subset = events.loc[events["query_id"].isin(split_ids[split])]
        if subset.empty:
            raise ValueError(f"no events in {split}")
        ids = subset["query_id"].to_numpy(dtype=np.int64)
        edge_indices = subset["edge_index"].to_numpy(dtype=np.int64)
        raw = subset["Dhat_pq_cpp"].to_numpy(dtype=np.float64)
        exact = subset["D_exact_cpp"].to_numpy(dtype=np.float64)
        tau = subset["tau"].to_numpy(dtype=np.float64)
        eligible = subset["edge_eligible"].to_numpy(dtype=bool)
        if not np.all(np.isfinite(exact) & np.isfinite(tau) & (tau > 0)):
            raise ValueError("nonfinite exact distance or threshold")
        far = exact > tau
        for beta in beta_grid:
            decision = eligible & (raw > beta * tau)
            row, detail = evaluate(decision, far, ids, split,
                                   "pq-beta", float(beta), -1, 0)
            frontier.append(row); query_counts.extend(detail)
        row, detail = evaluate(far, far, ids, split,
                               "correction-oracle", 1.0, -1, 0)
        frontier.append(row); query_counts.extend(detail)
        for item in sketch["sketches"]:
            directory = Path(item["directory"])
            bits, seed = item["bits"], item["seed"]
            matrix = np.fromfile(directory / "matrix.f32", dtype="<f4")
            matrix = matrix.reshape((bits, dimension)).astype(np.float64)
            projected = queries.astype(np.float64) @ matrix.T
            signs = np.load(directory / "signs.npy", mmap_mode="r")
            scales = np.load(directory / "scales.npy", mmap_mode="r")
            offsets = np.load(directory / "offsets.npy", mmap_mode="r")
            valid = np.load(directory / "valid.npy", mmap_mode="r")
            correction = np.empty(len(subset), dtype=np.float64)
            for start in range(0, len(subset), 50000):
                end = min(start + 50000, len(subset))
                unpacked = np.unpackbits(signs[edge_indices[start:end]],
                                          axis=1, bitorder="little")[:, :bits]
                signed = np.where(unpacked, 1.0, -1.0)
                dot = np.einsum("ij,ij->i", signed, projected[ids[start:end]])
                correction[start:end] = (
                    scales[edge_indices[start:end]] * dot -
                    offsets[edge_indices[start:end]])
            corrected = raw - correction
            usable = eligible & (valid[edge_indices] == 1) & np.isfinite(corrected)
            row, detail = evaluate(usable & (corrected > tau), far, ids,
                                   split, "residual-direct", 1.0, seed, bits)
            frontier.append(row); query_counts.extend(detail)
            for theta in theta_grid:
                theta = round(float(theta), 2)
                decision = usable & (corrected > theta * tau)
                row, detail = evaluate(decision, far, ids, split,
                                       "residual-threshold", theta, seed, bits)
                frontier.append(row); query_counts.extend(detail)
    pd.DataFrame(frontier).to_csv(output / "frontier.csv", index=False)
    pd.DataFrame(query_counts).to_csv(output / "query_counts.csv", index=False)
    manifest = {"schema_version": 1, "dataset": str(Path(args.dataset).resolve()),
                "sketch": str(Path(args.sketch).resolve()), "splits": splits,
                "frontier_sha256": sha256(output / "frontier.csv"),
                "query_counts_sha256": sha256(output / "query_counts.csv"),
                "sampled_exposure_only": True}
    write_json(output / "manifest.json", manifest)
    mark_complete(output, {"dataset": sha256(args.dataset), "sketch": sha256(args.sketch)},
                  {"manifest": sha256(output / "manifest.json")})


if __name__ == "__main__":
    main()
