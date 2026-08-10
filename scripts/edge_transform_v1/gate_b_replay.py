#!/usr/bin/env python3
"""Same-budget ETV1 Gate-B replay on frozen operational records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from etv1_quantizer import fit_opq, fit_pq, random_orthogonal


class Fvecs:
    def __init__(self, path: Path, internal_to_label: np.ndarray | None = None):
        raw = np.memmap(path, dtype=np.int32, mode="r")
        if raw.size == 0:
            raise ValueError(f"empty fvecs file: {path}")
        self.dimension = int(raw[0])
        stride = self.dimension + 1
        if self.dimension <= 0 or raw.size % stride:
            raise ValueError(f"invalid fvecs file: {path}")
        self.count = raw.size // stride
        records = raw.reshape(self.count, stride)
        if not np.all(records[:, 0] == self.dimension):
            raise ValueError(f"inconsistent dimensions: {path}")
        self._vectors = records[:, 1:].view(np.float32)
        self.internal_to_label = internal_to_label

    def take(self, ids: np.ndarray) -> np.ndarray:
        if self.internal_to_label is not None:
            if np.any(ids < 0) or np.any(ids >= self.internal_to_label.size):
                raise IndexError("internal HNSW ID is out of range")
            ids = self.internal_to_label[ids]
        if np.any(ids < 0) or np.any(ids >= self.count):
            raise IndexError("fvecs ID is out of range")
        return np.asarray(self._vectors[ids], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--base-fvecs", type=Path, required=True)
    parser.add_argument("--query-fvecs", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--internal-to-label", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ksub", type=int, default=256)
    parser.add_argument("--train-records", type=int, default=50_000)
    parser.add_argument("--eval-records-per-split", type=int, default=0)
    parser.add_argument("--opq-iterations", type=int, default=3)
    parser.add_argument("--kmeans-iterations", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def deterministic_sample(indices: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or indices.size <= limit:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=limit, replace=False))


def materialize(
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    base: Fvecs,
    queries: Fvecs,
) -> dict[str, np.ndarray]:
    q = queries.take(arrays["query_id"][indices].astype(np.int64))
    c = base.take(arrays["current_node_id"][indices].astype(np.int64))
    v = base.take(arrays["candidate_id"][indices].astype(np.int64))
    edge = v.astype(np.float64) - c.astype(np.float64)
    length = np.linalg.norm(edge, axis=1)
    valid = np.isfinite(length) & (length > 0.0)
    if not np.all(valid):
        q, c, v, edge, length, indices = (
            value[valid] for value in (q, c, v, edge, length, indices)
        )
    direction = np.asarray(edge / length[:, None], dtype=np.float32)
    return {
        "indices": indices,
        "q": q,
        "c": c,
        "v": v,
        "u": direction,
        "length": length,
    }


def quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {name: float("nan") for name in ("p10", "p50", "p90", "p99")}
    return {
        name: float(np.quantile(values, point))
        for name, point in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9), ("p99", 0.99))
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_query_bootstrap(
    left: dict[str, float], right: dict[str, float], seed: int, repetitions: int = 10_000
) -> dict[str, float]:
    common = np.asarray(sorted(set(left) & set(right)), dtype=str)
    if common.size == 0:
        return {"mean_difference": float("nan"), "ci95_lower": float("nan"), "ci95_upper": float("nan")}
    differences = np.asarray([left[qid] - right[qid] for qid in common], dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 256):
        count = min(256, repetitions - start)
        samples = rng.integers(0, differences.size, size=(count, differences.size))
        bootstrap[start : start + count] = np.mean(differences[samples], axis=1)
    return {
        "queries": int(common.size),
        "mean_difference": float(np.mean(differences)),
        "ci95_lower": float(np.quantile(bootstrap, 0.025)),
        "ci95_upper": float(np.quantile(bootstrap, 0.975)),
    }


def evaluate_model(
    model,
    pivot: np.ndarray | None,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    base: Fvecs,
    queries: Fvecs,
    batch_size: int,
) -> dict[str, object]:
    prune: list[np.ndarray] = []
    current_prune: list[np.ndarray] = []
    violation: list[np.ndarray] = []
    gamma: list[np.ndarray] = []
    query_ids: list[np.ndarray] = []
    gaps: list[np.ndarray] = []
    oracle: list[np.ndarray] = []
    raw_prune: list[np.ndarray] = []
    effective_radius: list[np.ndarray] = []
    edge_mse: list[np.ndarray] = []
    trace_exact_delta: list[np.ndarray] = []
    defect = float(np.linalg.norm(np.eye(model.dimension) - model.transform.T @ model.transform, ord=2))
    for start in range(0, indices.size, batch_size):
        selected = indices[start : start + batch_size]
        batch = materialize(arrays, selected, base, queries)
        ids = batch["indices"]
        q = batch["q"].astype(np.float64)
        c = batch["c"].astype(np.float64)
        v = batch["v"].astype(np.float64)
        u = batch["u"]
        ell = batch["length"]
        local_pivot = c if pivot is None else np.broadcast_to(pivot, c.shape)
        x = q - local_pivot
        w = x @ model.transform.T
        y = u.astype(np.float64) @ model.transform.T
        yhat = model.reconstruct_transformed(u).astype(np.float64)
        selected_support = np.zeros(ids.size, dtype=np.float64)
        block_radius = np.zeros(ids.size, dtype=np.float64)
        eps_global_sq = np.zeros(ids.size, dtype=np.float64)
        for block in range(model.m):
            sl = slice(block * model.dsub, (block + 1) * model.dsub)
            wb, yb, yhb = w[:, sl], y[:, sl], yhat[:, sl]
            wn = np.linalg.norm(wb, axis=1)
            tn = np.linalg.norm(yb, axis=1)
            eps = np.linalg.norm(yb - yhb, axis=1)
            norm_support = wn * tn
            reconstruction_support = np.sum(wb * yhb, axis=1) + wn * eps
            selected_support += np.minimum(norm_support, reconstruction_support)
            block_radius += wn * eps
            eps_global_sq += eps**2
        offset = np.zeros(ids.size) if pivot is None else np.sum((local_pivot - c) * u, axis=1)
        defect_padding = defect * np.linalg.norm(x, axis=1)
        support = offset + selected_support + defect_padding
        current = np.sum((q - c) ** 2, axis=1)
        exact = np.sum((q - v) ** 2, axis=1)
        lower = np.maximum(0.0, current + ell**2 - 2.0 * ell * support)
        trace_current = arrays["current_lb"][ids].astype(np.float64)
        safe = np.maximum(trace_current, lower)
        threshold = arrays["threshold"][ids].astype(np.float64)
        current_radius = 2.0 * ell * np.linalg.norm(q - c, axis=1) * arrays["direction_error"][ids]
        new_radius = 2.0 * ell * block_radius
        denominator = np.maximum(current_radius, np.finfo(np.float64).tiny)
        gamma.append(new_radius / denominator)
        prune.append(safe > threshold)
        current_prune.append(trace_current > threshold)
        violation.append(lower - exact)
        query_ids.append(arrays["query_id"][ids].astype(np.int64))
        gaps.append(safe - threshold)
        oracle.append(exact > threshold)
        raw_prune.append(arrays["approximate_squared_distance"][ids].astype(np.float64) > threshold)
        effective_radius.append(new_radius)
        edge_mse.append(np.mean((y - yhat) ** 2, axis=1))
        trace_exact_delta.append(np.abs(exact - arrays["shadow_exact_squared_distance"][ids].astype(np.float64)))

    pruned = np.concatenate(prune)
    current = np.concatenate(current_prune)
    violations = np.concatenate(violation)
    qids = np.concatenate(query_ids)
    gamma_values = np.concatenate(gamma)
    gap_values = np.concatenate(gaps)
    oracle_values = np.concatenate(oracle)
    raw_values = np.concatenate(raw_prune)
    radius_values = np.concatenate(effective_radius)
    mse_values = np.concatenate(edge_mse)
    exact_delta = np.concatenate(trace_exact_delta)
    unique = np.unique(qids)
    per_query = np.asarray([np.mean(pruned[qids == qid]) for qid in unique])
    per_query_map = {str(int(qid)): float(value) for qid, value in zip(unique, per_query)}
    oracle_coverage = float(np.mean(oracle_values))
    return {
        "records": int(pruned.size),
        "queries": int(unique.size),
        "deterministic_saving": float(np.mean(pruned)),
        "current_saving": float(np.mean(current)),
        "additional_saving": float(np.mean(pruned & ~current)),
        "exact_oracle_opportunity": oracle_coverage,
        "raw_estimate_coverage": float(np.mean(raw_values)),
        "oracle_recovery_ratio": float(np.mean(pruned) / oracle_coverage) if oracle_coverage else 0.0,
        "queries_with_nonzero_saving_fraction": float(np.mean(per_query > 0.0)),
        "per_query_saving": quantiles(per_query),
        "per_query_saving_by_query": per_query_map,
        "gamma": quantiles(gamma_values),
        "effective_radius": quantiles(radius_values),
        "safe_gap": quantiles(gap_values),
        "edge_direction_mse": float(np.mean(mse_values)),
        "trace_exact_max_abs_delta": float(np.max(exact_delta)),
        "max_new_lb_minus_exact": float(np.max(violations)),
        "new_lb_violations_at_1e_8": int(np.count_nonzero(violations > 1e-8)),
        "orthogonality_defect": defect,
    }


def main() -> int:
    args = parse_args()
    for path in (args.records, args.base_fvecs, args.query_fvecs, args.split_manifest, args.internal_to_label):
        if not path.is_file():
            raise SystemExit(f"required Gate-B input is unavailable: {path}")
    internal_to_label = np.load(args.internal_to_label, allow_pickle=False)
    if internal_to_label.ndim != 1 or not np.issubdtype(internal_to_label.dtype, np.integer):
        raise SystemExit("internal-to-label mapping must be a one-dimensional integer .npy array")
    internal_to_label = internal_to_label.astype(np.int64, copy=False)
    base, queries = Fvecs(args.base_fvecs, internal_to_label), Fvecs(args.query_fvecs)
    if base.dimension != queries.dimension or base.dimension % args.m:
        raise SystemExit("incompatible vector dimensions or PQ m")
    columns = [
        "query_id",
        "current_node_id",
        "candidate_id",
        "threshold",
        "direction_error",
        "current_lb",
        "shadow_exact_squared_distance",
        "approximate_squared_distance",
    ]
    table = pq.read_table(args.records, columns=columns)
    arrays = {name: table[name].to_numpy(zero_copy_only=False) for name in columns}
    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    split_sets = {
        name: np.asarray(values, dtype=np.int64)
        for name, values in split_manifest["query_ids"].items()
    }
    split_indices = {
        name: np.flatnonzero(np.isin(arrays["query_id"], values))
        for name, values in split_sets.items()
    }
    train_indices = deterministic_sample(split_indices["train"], args.train_records, args.seed)
    train = materialize(arrays, train_indices, base, queries)
    train_q = train["q"].astype(np.float64)
    train_c = train["c"].astype(np.float64)
    train_v = train["v"].astype(np.float64)
    train_exact = np.sum((train_q - train_v) ** 2, axis=1)
    train_threshold = arrays["threshold"][train["indices"]].astype(np.float64)
    margin_scale = max(float(np.quantile(np.abs(train_exact - train_threshold), 0.1)), 1e-6)
    weights = train["length"] ** 2 / (np.abs(train_exact - train_threshold) + margin_scale)
    weights = np.minimum(weights, np.quantile(weights, 0.99))
    query_mean = np.mean(queries.take(split_sets["train"]).astype(np.float64), axis=0)

    methods = {
        "identity_pq": fit_pq(
            train["u"], m=args.m, ksub=args.ksub, seed=args.seed,
            max_iter=args.kmeans_iterations, method="identity_pq"
        ),
        "random_jq": fit_pq(
            train["u"], m=args.m, ksub=args.ksub, seed=args.seed + 100,
            transform=random_orthogonal(base.dimension, args.seed + 100),
            max_iter=args.kmeans_iterations, method="random_jq"
        ),
        "edge_opq": fit_opq(
            train["u"], m=args.m, ksub=args.ksub, seed=args.seed + 200,
            iterations=args.opq_iterations, max_iter=args.kmeans_iterations,
            method="edge_opq"
        ),
        "etv1_weighted_opq": fit_opq(
            train["u"], m=args.m, ksub=args.ksub, seed=args.seed + 300,
            iterations=args.opq_iterations, max_iter=args.kmeans_iterations,
            sample_weight=weights, method="etv1_weighted_opq"
        ),
    }

    report: dict[str, object] = {
        "format": "etv1_gate_b_replay",
        "format_version": 1,
        "records_sha256": sha256_file(args.records),
        "internal_to_label_sha256": sha256_file(args.internal_to_label),
        "dimension": base.dimension,
        "m": args.m,
        "ksub": args.ksub,
        "code_bytes": int(np.ceil(args.m * np.ceil(np.log2(args.ksub)) / 8.0)),
        "train_records": int(train["indices"].size),
        "development_only": True,
        "methods": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for method_name, model in methods.items():
        model.save(str(args.output_dir / f"{method_name}.npz"))
        method_report: dict[str, object] = {}
        for split_name in ("validation", "test"):
            indices = deterministic_sample(
                split_indices[split_name], args.eval_records_per_split, args.seed + 1000
            )
            method_report[split_name] = {
                "query_mean_pivot": evaluate_model(
                    model, query_mean, arrays, indices, base, queries, args.batch_size
                ),
                "p_equals_c_oracle": evaluate_model(
                    model, None, arrays, indices, base, queries, args.batch_size
                ),
            }
        report["methods"][method_name] = method_report

    test_pivot = "query_mean_pivot"
    etv1 = report["methods"]["etv1_weighted_opq"]["test"][test_pivot]
    edge_opq = report["methods"]["edge_opq"]["test"][test_pivot]
    comparison = paired_query_bootstrap(
        etv1["per_query_saving_by_query"], edge_opq["per_query_saving_by_query"], args.seed + 9000
    )
    base_conditions = {
        "zero_deterministic_violations": etv1["new_lb_violations_at_1e_8"] == 0,
        "saving_at_least_10_percent": etv1["deterministic_saving"] >= 0.10,
        "nonzero_saving_queries_at_least_75_percent": etv1["queries_with_nonzero_saving_fraction"] >= 0.75,
        "median_per_query_saving_at_least_5_percent": etv1["per_query_saving"]["p50"] >= 0.05,
    }
    pruning_aware_conditions = {
        "etv1_gain_at_least_2_percentage_points": comparison["mean_difference"] >= 0.02,
        "bootstrap_ci95_lower_above_zero": comparison["ci95_lower"] > 0.0,
    }
    base_pass = all(base_conditions.values())
    pruning_aware_pass = base_pass and all(pruning_aware_conditions.values())
    report["gate_b"] = {
        "evaluated_method": "etv1_weighted_opq/query_mean_pivot/test",
        "development_only": True,
        "base_conditions": base_conditions,
        "pruning_aware_conditions": pruning_aware_conditions,
        "etv1_minus_edge_opq_query_bootstrap": comparison,
        "base_certificate_pass": base_pass,
        "pruning_aware_transform_pass": pruning_aware_pass,
        "decision": (
            "GO_ETV1" if pruning_aware_pass else
            "GO_EDGE_OPQ" if base_pass else
            "NO_GO_CPP"
        ),
    }

    output = args.output_dir / "gate_b_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    violations = sum(
        branch["new_lb_violations_at_1e_8"]
        for method in report["methods"].values()
        for split in method.values()
        for branch in split.values()
    )
    return 0 if violations == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
