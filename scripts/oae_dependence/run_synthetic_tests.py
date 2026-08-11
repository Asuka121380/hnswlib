#!/usr/bin/env python3
"""Deterministic null and positive controls for conditional prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from oae_dependence_core import bootstrap_mean_ci, semantic_sha256, write_json


def trial(*, positive: bool, seed: int) -> dict[str, float | list[float]]:
    rng = np.random.default_rng(seed)
    train_queries, validation_queries, events_per_query, probes = 200, 200, 32, 8
    train_group = rng.integers(0, 2, size=(train_queries, events_per_query))
    validation_group = rng.integers(0, 2, size=(validation_queries, events_per_query))
    means = np.zeros((2, probes))
    if positive:
        means[0, 0], means[1, 1] = 3.0, 3.0
    train_y = rng.normal(size=(train_queries, events_per_query, probes)) + means[train_group]
    validation_y = rng.normal(size=(validation_queries, events_per_query, probes)) + means[validation_group]
    global_mean = train_y.mean(axis=(0, 1))
    group_mean = np.stack([train_y[train_group == group].mean(axis=0) for group in range(2)])
    global_error = np.square(validation_y - global_mean).mean(axis=(1, 2))
    conditional_error = np.square(validation_y - group_mean[validation_group]).mean(axis=(1, 2))
    improvement = global_error - conditional_error
    low, high = bootstrap_mean_ci(improvement, 2000, seed + 1)
    relative = float((global_error.mean() - conditional_error.mean()) / global_error.mean())
    return {"relative_improvement": relative, "mean_improvement_ci95": [low, high]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    null, positive = trial(positive=False, seed=20260811), trial(positive=True, seed=20260811)
    null_pass = null["relative_improvement"] < 0.02 or null["mean_improvement_ci95"][0] <= 0.0
    positive_pass = positive["relative_improvement"] >= 0.02 and positive["mean_improvement_ci95"][0] > 0.0
    report = {"status": "PASS" if null_pass and positive_pass else "FAIL", "seed": 20260811,
              "null": {**null, "passed": null_pass}, "positive": {**positive, "passed": positive_pass}}
    report["semantic_sha256"] = semantic_sha256(report)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
