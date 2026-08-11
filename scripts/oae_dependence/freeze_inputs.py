#!/usr/bin/env python3
"""Freeze query ranges and immutable artifacts before OAE dependence tracing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from oae_dependence_core import Fvecs, require, semantic_sha256, sha256_file, write_json


def resolved(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


def artifact(path: Path) -> dict[str, object]:
    require(path.is_file(), f"missing frozen artifact: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_split(source: Fvecs, offset: int, count: int, path: Path) -> None:
    require(0 <= offset and count > 0 and offset + count <= source.count, "query split is out of range")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(source._raw[offset:offset + count]).tofile(path)  # exact fvec records, including headers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    out = args.output_dir.resolve()
    require(not out.exists() or not any(out.iterdir()), f"refusing to overwrite non-empty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    dimension = int(config["dimension"])
    train = config["query_splits"]["training"]
    validation = config["query_splits"]["validation"]
    train_ids = set(range(int(train["offset"]), int(train["offset"]) + int(train["count"])))
    validation_ids = set(range(int(validation["offset"]), int(validation["offset"]) + int(validation["count"])))
    consumed = set()
    for interval in config["excluded_query_ranges"]:
        consumed.update(range(int(interval["offset"]), int(interval["offset"]) + int(interval["count"])))
    require(train_ids.isdisjoint(validation_ids), "training and validation queries overlap")
    require(train_ids.isdisjoint(consumed) and validation_ids.isdisjoint(consumed), "query split overlaps an excluded range")
    require(len(train_ids) + len(validation_ids) <= int(config["resource_caps"]["learn_queries_total"]),
            "query split exceeds preregistered cap")

    paths = {name: resolved(value) for name, value in config["paths"].items()}
    require(paths["learn"].name == "gist_learn.fvecs", "only gist_learn.fvecs may supply experimental queries")
    require("query" not in paths["learn"].name, "official query file is forbidden")
    learn = Fvecs(paths["learn"], dimension)
    role_manifests = {}
    for role, split in (("training", train), ("validation", validation)):
        offset, count = int(split["offset"]), int(split["count"])
        query_path = out / f"{role}_queries.fvecs"
        truth_path = out / f"{role}_trace_placeholder_groundtruth.ivecs"
        write_split(learn, offset, count, query_path)
        placeholder = np.zeros((count, int(config["trace"]["k"]) + 1), dtype="<i4")
        placeholder[:, 0] = int(config["trace"]["k"])
        placeholder.tofile(truth_path)
        dataset_path = out / f"{role}_trace_dataset.json"
        write_json(dataset_path, {
            "dataset": config["dataset"], "distance_kind": "squared_l2_float32",
            "base_path": str(paths["base"]), "query_path": str(query_path),
            "ground_truth_path": str(truth_path), "base_format": "fvecs",
            "query_format": "fvecs", "ground_truth_format": "ivecs",
            "dimension": dimension, "n_base": int(config["n_base"]), "n_query": count,
            "ground_truth_k": int(config["trace"]["k"]),
            "ground_truth_semantics": "runner_compatibility_placeholder_not_used_by_experiment",
        })
        role_manifest = {
            "role": role, "source": str(paths["learn"]), "source_offset": offset,
            "count": count, "original_query_ids": list(range(offset, offset + count)),
            "query_file": artifact(query_path), "trace_dataset_config": artifact(dataset_path),
            "placeholder_ground_truth": artifact(truth_path),
        }
        write_json(out / f"{role}_query_manifest.json", role_manifest)
        role_manifests[role] = role_manifest

    frozen = {name: artifact(path) for name, path in paths.items()}
    manifest = {
        "schema": "oae_edge_query_dependence_input_v1", "dataset": config["dataset"],
        "dimension": dimension, "n_base": int(config["n_base"]), "artifacts": frozen,
        "query_splits": role_manifests, "excluded_query_ranges": config["excluded_query_ranges"],
        "trace": config["trace"], "model": config["model"], "gates": config["gates"],
        "resource_caps": config["resource_caps"],
        "declarations": {
            "official_gist_query_not_used": True,
            "queries_are_responses_not_predictor_features": True,
            "no_new_quantizer_is_trained": True,
        },
    }
    # Keep machine-specific absolute paths available for execution, but exclude
    # them from the semantic identity of the experiment.
    semantic_payload = {
        "schema": manifest["schema"], "dataset": manifest["dataset"],
        "dimension": manifest["dimension"], "n_base": manifest["n_base"],
        "artifacts": {name: {"bytes": value["bytes"], "sha256": value["sha256"]}
                      for name, value in frozen.items()},
        "query_splits": {
            role: {"source_offset": value["source_offset"], "count": value["count"],
                   "original_query_ids": value["original_query_ids"],
                   "query_file_sha256": value["query_file"]["sha256"]}
            for role, value in role_manifests.items()
        },
        "excluded_query_ranges": manifest["excluded_query_ranges"], "trace": manifest["trace"],
        "model": manifest["model"], "gates": manifest["gates"],
        "resource_caps": manifest["resource_caps"], "declarations": manifest["declarations"],
    }
    manifest["semantic_payload"] = semantic_payload
    manifest["semantic_sha256"] = semantic_sha256(semantic_payload)
    write_json(out / "input_manifest.json", manifest)
    write_json(out / "experiment_contract.json", config)
    print(json.dumps({"status": "FROZEN", "output_dir": str(out), "semantic_sha256": manifest["semantic_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
