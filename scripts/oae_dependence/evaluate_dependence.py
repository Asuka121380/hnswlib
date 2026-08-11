#!/usr/bin/env python3
"""Fit training-only edge groups and evaluate both preregistered OAE tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from oae_dependence_core import (
    Fvecs, SearchEvents, evaluate_primary, evaluate_secondary, fit_model,
    semantic_sha256, sha256_file, write_json, write_metrics_parquet,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--training-events", type=Path, required=True)
    parser.add_argument("--validation-events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    input_manifest = json.loads((args.input_dir / "input_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((args.input_dir / "experiment_contract.json").read_text(encoding="utf-8"))
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if out != args.input_dir.resolve():
        for name in ("input_manifest.json", "experiment_contract.json", "training_query_manifest.json", "validation_query_manifest.json"):
            (out / name).write_bytes((args.input_dir / name).read_bytes())
    if (out / "run_manifest.json").exists():
        raise RuntimeError(f"refusing to overwrite completed run: {out}")

    dimension = int(contract["dimension"])
    base = Fvecs(Path(input_manifest["artifacts"]["base"]["path"]), dimension)
    learn = Fvecs(Path(input_manifest["artifacts"]["learn"]["path"]), dimension)
    training = SearchEvents.load(args.training_events)
    validation = SearchEvents.load(args.validation_events)
    if not set(np.unique(training.query_id)).isdisjoint(set(np.unique(validation.query_id))):
        raise RuntimeError("training and validation event queries overlap")
    model_cfg, gates = contract["model"], contract["gates"]
    model = fit_model(
        training, base, learn,
        probe_count=int(model_cfg["probe_count"]), direction_clusters=int(model_cfg["direction_clusters"]),
        length_bins=int(model_cfg["length_bins"]), shrinkage=float(model_cfg["shrinkage_pseudocount"]),
        probe_seed=int(model_cfg["probe_seed"]), group_seed=int(model_cfg["group_seed"]),
        kmeans_iterations=int(model_cfg["kmeans_iterations"]),
        group_fit_edge_cap=int(model_cfg["group_fit_edge_cap"]), batch_size=int(model_cfg["batch_size"]),
    )
    np.save(out / "probe_matrix.npy", model.probe, allow_pickle=False)
    np.savez(out / "edge_group_model.npz", direction_centers=model.direction_centers,
             length_boundaries=model.length_boundaries)
    np.savez(out / "group_statistics.npz", mu_global=model.mu_global, mu_group=model.mu_group,
             a_global=model.a_global, a_group=model.a_group,
             group_effective_count=model.group_effective_count)

    primary_metrics, primary_report = evaluate_primary(
        validation, base, learn, model, batch_size=int(model_cfg["batch_size"]),
        bootstrap_repetitions=int(model_cfg["bootstrap_repetitions"]),
        bootstrap_seed=int(model_cfg["bootstrap_seed"]),
        relative_gate=float(gates["primary_relative_improvement_minimum"]),
        minimum_noninferior_probes=int(gates["minimum_noninferior_probes"]),
        maximum_query_contribution=float(gates["maximum_single_query_contribution"]),
    )
    write_metrics_parquet(out / "primary_per_query_metrics.parquet", primary_metrics)
    primary_report["semantic_sha256"] = semantic_sha256(primary_report)
    write_json(out / "primary_report.json", primary_report)

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "scripts" / "v0"))
    from v0pq_format import read_v0pq
    codebook_path = Path(input_manifest["artifacts"]["v0_codebook"]["path"])
    codebook = read_v0pq(codebook_path)
    h = codebook.header
    if h.dimension != dimension:
        raise RuntimeError("V0 codebook and experiment dimensions differ")
    centroids = np.asarray(codebook.centroids, dtype=np.float32).reshape(h.M, h.ksub, h.dsub)
    secondary_metrics, secondary_report = evaluate_secondary(
        validation, base, learn, model, centroids, batch_size=int(model_cfg["secondary_batch_size"]),
        bootstrap_repetitions=int(model_cfg["bootstrap_repetitions"]),
        bootstrap_seed=int(model_cfg["bootstrap_seed"]),
    )
    write_metrics_parquet(out / "v0_residual_per_query_metrics.parquet", secondary_metrics)
    secondary_report["v0_codebook_sha256"] = sha256_file(codebook_path)
    secondary_report["residual_definition"] = "edge_length_times_standard_pq_reconstruction_of_unit_edge_direction"
    secondary_report["semantic_sha256"] = semantic_sha256(secondary_report)
    write_json(out / "v0_residual_report.json", secondary_report)

    if primary_report["status"] == "PASS_EDGE_QUERY_DEPENDENCE":
        decision = ("GO_SMALL_OAE_QUANTIZER_THEORY" if secondary_report["status"] == "PASS_V0_RESIDUAL_RELEVANCE"
                    else "CONDITIONAL_STRUCTURE_NOT_YET_QUANTIZATION_RELEVANT")
    elif primary_report["status"] == "INCONCLUSIVE_EDGE_QUERY_DEPENDENCE":
        decision = "INCONCLUSIVE_NO_SCALE_UP"
    else:
        decision = "STOP_EDGE_GEOMETRY_CONDITIONED_OAE"
    artifacts = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    run_manifest = {
        "schema": "oae_edge_query_dependence_run_v1", "decision": decision,
        "primary_status": primary_report["status"], "secondary_status": secondary_report["status"],
        "training_event_dataset_sha256": sha256_file(args.training_events),
        "validation_event_dataset_sha256": sha256_file(args.validation_events),
        "input_semantic_sha256": input_manifest["semantic_sha256"], "artifacts": artifacts,
        "declarations": {"validation_parameters_frozen": True, "new_quantizer_trained": False,
                         "hnsw_control_flow_modified": False},
    }
    run_manifest["semantic_sha256"] = semantic_sha256(run_manifest)
    write_json(out / "run_manifest.json", run_manifest)
    print(json.dumps({"status": "COMPLETE", "decision": decision, "output_dir": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
