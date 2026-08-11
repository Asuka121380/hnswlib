#!/usr/bin/env python3
"""Training-only convergence diagnostic for the global OAE metric A.

The analysis never reads validation events.  It estimates query-balanced
per-query diagonal and frozen-probe covariance statistics, then compares
nested query counts, independent split halves, and the induced ranking of a
fixed sample of V0 residuals.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from oae_dependence_core import (
    Fvecs, SearchEvents, edge_geometry, reconstruct_pq, require,
    semantic_sha256, sha256_file, unique_edge_pairs, write_json,
)


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    require(denominator > 0.0, "reference statistic has zero norm")
    return float(np.linalg.norm(candidate - reference) / denominator)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    require(denominator > 0.0, "cannot compare zero vectors")
    return float(np.dot(left.ravel(), right.ravel()) / denominator)


def rank_values(values: np.ndarray) -> np.ndarray:
    """Deterministic average ranks without a SciPy dependency."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    return float(np.dot(left_centered, right_centered) / denominator) if denominator else 1.0


def prediction_comparison(candidate: np.ndarray, reference: np.ndarray, top_fraction: float) -> dict[str, float]:
    require(candidate.shape == reference.shape and candidate.ndim == 1, "prediction arrays must agree")
    require(0.0 < top_fraction < 1.0, "top fraction must be in (0,1)")
    denominator = float(np.linalg.norm(reference))
    nrmse = float(np.linalg.norm(candidate - reference) / denominator) if denominator else 0.0
    top_count = max(1, int(math.ceil(len(reference) * top_fraction)))
    reference_top = np.argpartition(reference, -top_count)[-top_count:]
    candidate_top = np.argpartition(candidate, -top_count)[-top_count:]
    overlap = len(np.intersect1d(reference_top, candidate_top, assume_unique=False)) / top_count
    return {
        "pearson": correlation(candidate, reference),
        "spearman": correlation(rank_values(candidate), rank_values(reference)),
        "normalized_rmse": nrmse,
        "top_fraction": top_fraction,
        "top_set_overlap": float(overlap),
    }


def top_subspace_angle_degrees(candidate: np.ndarray, reference: np.ndarray, rank: int) -> dict[str, float]:
    rank = min(rank, candidate.shape[0])
    _, candidate_vectors = np.linalg.eigh(candidate)
    _, reference_vectors = np.linalg.eigh(reference)
    singular = np.linalg.svd(candidate_vectors[:, -rank:].T @ reference_vectors[:, -rank:], compute_uv=False)
    angles = np.degrees(np.arccos(np.clip(singular, -1.0, 1.0)))
    return {"mean_degrees": float(angles.mean()), "maximum_degrees": float(angles.max()), "rank": rank}


def per_query_statistics(events: SearchEvents, base: Fvecs, learn: Fvecs,
                         probe: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    require(bool(np.all(events.query_id[:-1] <= events.query_id[1:])),
            "training events must be grouped in nondecreasing query order")
    query_ids, starts, counts = np.unique(events.query_id, return_index=True, return_counts=True)
    diagonal = np.zeros((len(query_ids), base.dimension), dtype=np.float64)
    probe_covariance = np.zeros((len(query_ids), probe.shape[1], probe.shape[1]), dtype=np.float64)
    for query_index, (query_id, start, count) in enumerate(zip(query_ids, starts, counts)):
        query = learn.rows(np.asarray([query_id], dtype=np.int64))[0].astype(np.float32)
        stop = int(start + count)
        for batch_start in range(int(start), stop, batch_size):
            batch_stop = min(stop, batch_start + batch_size)
            current = events.current_label[batch_start:batch_stop]
            x = query[None, :] - base.rows(current).astype(np.float32)
            diagonal[query_index] += np.sum(np.square(x, dtype=np.float64), axis=0)
            projected = x @ probe
            probe_covariance[query_index] += projected.T @ projected
        diagonal[query_index] /= count
        probe_covariance[query_index] /= count
        if (query_index + 1) % 50 == 0 or query_index + 1 == len(query_ids):
            print(f"per-query statistics: {query_index + 1}/{len(query_ids)}", flush=True)
    return query_ids, counts, diagonal, probe_covariance


def residual_sample(events: SearchEvents, base: Fvecs, codebook: Any,
                    sample_count: int, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    current, neighbor = unique_edge_pairs(events)
    total = len(current)
    if sample_count > 0 and total > sample_count:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(total, size=sample_count, replace=False))
        current, neighbor = current[chosen], neighbor[chosen]
    directions, lengths = edge_geometry(base, current, neighbor)
    header = codebook.header
    centroids = np.asarray(codebook.centroids, dtype=np.float32).reshape(header.M, header.ksub, header.dsub)
    reconstructed = reconstruct_pq(directions, centroids) * lengths[:, None]
    residual = directions * lengths[:, None] - reconstructed
    return residual.astype(np.float32), np.stack([current, neighbor], axis=1), total


def residual_predictions(diagonal: np.ndarray, covariance: np.ndarray,
                         residual: np.ndarray, probe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diagonal_prediction = np.square(residual, dtype=np.float64) @ diagonal
    projected = residual @ probe
    probe_prediction = np.einsum("ni,ij,nj->n", projected, covariance, projected, optimize=True)
    return diagonal_prediction, probe_prediction


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"minimum": float(array.min()), "median": float(np.median(array)),
            "p90": float(np.quantile(array, 0.9)), "maximum": float(array.max())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--training-events", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir, output_dir = args.run_dir.resolve(), args.output_dir.resolve()
    require(not output_dir.exists() or not any(output_dir.iterdir()),
            f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    input_manifest = json.loads((run_dir / "input_manifest.json").read_text(encoding="utf-8"))
    events = SearchEvents.load(args.training_events)
    expected_ids = input_manifest["query_splits"]["training"]["original_query_ids"]
    observed_ids = np.unique(events.query_id)
    require(np.array_equal(observed_ids, np.asarray(expected_ids)), "training event query IDs disagree with frozen input")
    validation_ids = set(input_manifest["query_splits"]["validation"]["original_query_ids"])
    require(set(observed_ids).isdisjoint(validation_ids), "validation query leaked into stability analysis")

    dimension = int(input_manifest["dimension"])
    base = Fvecs(Path(input_manifest["artifacts"]["base"]["path"]), dimension)
    learn = Fvecs(Path(input_manifest["artifacts"]["learn"]["path"]), dimension)
    probe = np.load(run_dir / "probe_matrix.npy", allow_pickle=False)
    require(probe.shape == (dimension, int(config["probe_count"])), "frozen probe shape mismatch")
    query_ids, event_counts, per_query_diagonal, per_query_covariance = per_query_statistics(
        events, base, learn, probe, int(config["batch_size"]))

    query_counts = [int(value) for value in config["query_counts"]]
    require(query_counts == sorted(set(query_counts)), "query counts must be unique and sorted")
    require(query_counts[-1] == len(query_ids), "largest query count must equal all training queries")
    seed = int(config["seed"])
    permutation = np.random.default_rng(seed).permutation(len(query_ids))
    full_diagonal = per_query_diagonal.mean(axis=0)
    full_covariance = per_query_covariance.mean(axis=0)

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "scripts" / "v0"))
    from v0pq_format import read_v0pq
    codebook_path = Path(input_manifest["artifacts"]["v0_codebook"]["path"])
    codebook = read_v0pq(codebook_path)
    require(codebook.header.dimension == dimension, "V0 codebook dimension mismatch")
    residual, edge_pairs, unique_edge_count = residual_sample(
        events, base, codebook, int(config["residual_edge_sample_count"]), seed)
    reference_diag_prediction, reference_probe_prediction = residual_predictions(
        full_diagonal, full_covariance, residual, probe)

    convergence: dict[str, Any] = {}
    for count in query_counts:
        selected = permutation[:count]
        diagonal = per_query_diagonal[selected].mean(axis=0)
        covariance = per_query_covariance[selected].mean(axis=0)
        diag_prediction, probe_prediction = residual_predictions(diagonal, covariance, residual, probe)
        convergence[str(count)] = {
            "query_ids_sha256": semantic_sha256(sorted(int(query_ids[index]) for index in selected)),
            "diagonal_relative_l2": relative_l2(diagonal, full_diagonal),
            "diagonal_cosine_similarity": cosine_similarity(diagonal, full_diagonal),
            "probe_covariance_relative_frobenius": relative_l2(covariance, full_covariance),
            "probe_covariance_cosine_similarity": cosine_similarity(covariance, full_covariance),
            "probe_top_subspace_angle": top_subspace_angle_degrees(
                covariance, full_covariance, int(config["top_eigenspace_rank"])),
            "diagonal_residual_prediction": prediction_comparison(
                diag_prediction, reference_diag_prediction, float(config["top_fraction"])),
            "probe_residual_prediction": prediction_comparison(
                probe_prediction, reference_probe_prediction, float(config["top_fraction"])),
        }

    repetitions = int(config["split_half_repetitions"])
    half_size = int(config["split_half_query_count"])
    require(2 * half_size <= len(query_ids), "split halves exceed training query count")
    rng = np.random.default_rng(seed + 1)
    split_values: dict[str, list[float]] = {
        "diagonal_relative_l2": [], "probe_covariance_relative_frobenius": [],
        "diagonal_prediction_pearson_to_full": [], "diagonal_prediction_nrmse_to_full": [],
        "probe_prediction_pearson_to_full": [], "probe_prediction_nrmse_to_full": [],
    }
    for _ in range(repetitions):
        split = rng.permutation(len(query_ids))
        first, second = split[:half_size], split[half_size:2 * half_size]
        first_diag, second_diag = per_query_diagonal[first].mean(axis=0), per_query_diagonal[second].mean(axis=0)
        first_cov, second_cov = per_query_covariance[first].mean(axis=0), per_query_covariance[second].mean(axis=0)
        split_values["diagonal_relative_l2"].append(
            float(np.linalg.norm(first_diag - second_diag) / np.linalg.norm(full_diagonal)))
        split_values["probe_covariance_relative_frobenius"].append(
            float(np.linalg.norm(first_cov - second_cov) / np.linalg.norm(full_covariance)))
        for diagonal, covariance in ((first_diag, first_cov), (second_diag, second_cov)):
            diag_prediction, probe_prediction = residual_predictions(diagonal, covariance, residual, probe)
            diag_metrics = prediction_comparison(diag_prediction, reference_diag_prediction, float(config["top_fraction"]))
            probe_metrics = prediction_comparison(probe_prediction, reference_probe_prediction, float(config["top_fraction"]))
            split_values["diagonal_prediction_pearson_to_full"].append(diag_metrics["pearson"])
            split_values["diagonal_prediction_nrmse_to_full"].append(diag_metrics["normalized_rmse"])
            split_values["probe_prediction_pearson_to_full"].append(probe_metrics["pearson"])
            split_values["probe_prediction_nrmse_to_full"].append(probe_metrics["normalized_rmse"])
    split_half = {name: quantiles(values) for name, values in split_values.items()}

    decision_count = str(int(config["decision_query_count"]))
    require(decision_count in convergence, "decision query count is not in the convergence curve")
    selected = convergence[decision_count]
    thresholds = config["thresholds"]
    checks = {
        "diagonal_matrix_converged": selected["diagonal_relative_l2"] <= float(thresholds["matrix_relative_error_maximum"]),
        "probe_matrix_converged": selected["probe_covariance_relative_frobenius"] <= float(thresholds["matrix_relative_error_maximum"]),
        "split_half_diagonal_converged": split_half["diagonal_relative_l2"]["median"] <= float(thresholds["split_half_relative_error_maximum"]),
        "split_half_probe_converged": split_half["probe_covariance_relative_frobenius"]["median"] <= float(thresholds["split_half_relative_error_maximum"]),
        "diagonal_prediction_correlation": selected["diagonal_residual_prediction"]["pearson"] >= float(thresholds["prediction_correlation_minimum"]),
        "probe_prediction_correlation": selected["probe_residual_prediction"]["pearson"] >= float(thresholds["prediction_correlation_minimum"]),
        "diagonal_prediction_nrmse": selected["diagonal_residual_prediction"]["normalized_rmse"] <= float(thresholds["prediction_nrmse_maximum"]),
        "probe_prediction_nrmse": selected["probe_residual_prediction"]["normalized_rmse"] <= float(thresholds["prediction_nrmse_maximum"]),
        "diagonal_top_set_overlap": selected["diagonal_residual_prediction"]["top_set_overlap"] >= float(thresholds["top_set_overlap_minimum"]),
        "probe_top_set_overlap": selected["probe_residual_prediction"]["top_set_overlap"] >= float(thresholds["top_set_overlap_minimum"]),
    }
    status = "STABLE_AT_DECISION_QUERY_COUNT" if all(checks.values()) else "NOT_STABLE_AT_DECISION_QUERY_COUNT"
    report = {
        "schema": "oae_global_a_stability_report_v1", "status": status,
        "interpretation_scope": "training-sample adequacy only; not OAE performance",
        "training_query_count": len(query_ids), "training_event_count": events.size,
        "training_event_count_per_query": {"minimum": int(event_counts.min()),
                                             "median": float(np.median(event_counts)),
                                             "maximum": int(event_counts.max())},
        "validation_events_read": False, "decision_query_count": int(decision_count),
        "residual_edge_sample_count": len(residual), "training_unique_edge_count": unique_edge_count,
        "convergence": convergence, "split_half": split_half, "criteria": checks,
        "thresholds": thresholds,
    }
    report["semantic_sha256"] = semantic_sha256(report)
    np.savez(output_dir / "per_query_global_a_statistics.npz", query_id=query_ids,
             event_count=event_counts, diagonal=per_query_diagonal,
             probe_covariance=per_query_covariance, canonical_permutation=permutation,
             residual_edge_pairs=edge_pairs)
    shutil.copyfile(args.config, output_dir / "stability_contract.json")
    write_json(output_dir / "global_a_stability_report.json", report)
    artifacts = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "analysis_manifest.json":
            artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "schema": "oae_global_a_stability_analysis_v1", "status": "COMPLETE",
        "report_status": status, "input_manifest_semantic_sha256": input_manifest["semantic_sha256"],
        "training_events_sha256": sha256_file(args.training_events),
        "v0_codebook_sha256": sha256_file(codebook_path), "artifacts": artifacts,
        "declarations": {"training_queries_only": True, "validation_events_read": False,
                         "quantizer_trained": False, "hnsw_trace_rerun": False},
    }
    manifest["semantic_sha256"] = semantic_sha256(manifest)
    write_json(output_dir / "analysis_manifest.json", manifest)
    base.close()
    learn.close()
    print(json.dumps({"status": "COMPLETE", "stability": status,
                      "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
