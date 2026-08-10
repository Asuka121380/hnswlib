#!/usr/bin/env python3
"""Vectorized Phase-A property stress test for ETV1."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.trials <= 0 or args.batch_size <= 0:
        raise SystemExit("trials and batch-size must be positive")
    if args.dimension <= 1 or args.blocks <= 0 or args.dimension % args.blocks != 0:
        raise SystemExit("dimension must be >1 and divisible by blocks")

    rng = np.random.default_rng(args.seed)
    d = args.dimension
    width = d // args.blocks
    tolerance = 1e-10
    counters = {
        "lb_violations": 0,
        "progressive_monotonicity_violations": 0,
        "progressive_safety_violations": 0,
        "block_dominance_violations": 0,
        "interval_length_violations": 0,
        "nonorthogonal_transform_violations": 0,
    }
    maxima = {
        "lb_minus_exact": -math.inf,
        "progressive_drop": 0.0,
        "progressive_lb_minus_exact": -math.inf,
        "block_minus_global_radius": -math.inf,
        "interval_lb_minus_exact": -math.inf,
        "nonorthogonal_lb_minus_exact": -math.inf,
    }

    completed = 0
    while completed < args.trials:
        n = min(args.batch_size, args.trials - completed)
        q = rng.normal(size=(n, d))
        c = rng.normal(size=(n, d))
        p = rng.normal(size=(n, d))
        u = rng.normal(size=(n, d))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        ell = 10.0 ** rng.uniform(-3.0, 2.0, size=n)
        v = c + ell[:, None] * u
        w = q - p
        noise_scale = 10.0 ** rng.uniform(-4.0, -0.1, size=(n, 1))
        y_hat = u - rng.normal(size=(n, d)) * noise_scale
        offset = np.sum((p - c) * u, axis=1)
        offset_upper = offset + np.abs(rng.normal(scale=1e-12, size=n))

        norm_supports = np.empty((n, args.blocks), dtype=np.float64)
        selected_supports = np.empty_like(norm_supports)
        eps = np.empty_like(norm_supports)
        for block in range(args.blocks):
            sl = slice(block * width, (block + 1) * width)
            wb = w[:, sl]
            yb = u[:, sl]
            yhb = y_hat[:, sl]
            wn = np.linalg.norm(wb, axis=1)
            tn = np.linalg.norm(yb, axis=1) * (1.0 + rng.random(n) * 1e-12)
            eb = np.linalg.norm(yb - yhb, axis=1) * (1.0 + rng.random(n) * 1e-12)
            norm_supports[:, block] = wn * tn
            reconstruction = np.sum(wb * yhb, axis=1) + wn * eb
            selected_supports[:, block] = np.minimum(norm_supports[:, block], reconstruction)
            eps[:, block] = eb

        current = np.sum((q - c) ** 2, axis=1)
        exact = np.sum((q - v) ** 2, axis=1)
        support = offset_upper + np.sum(selected_supports, axis=1)
        lower = np.maximum(0.0, current + ell**2 - 2.0 * ell * support)
        delta = lower - exact
        counters["lb_violations"] += int(np.count_nonzero(delta > tolerance))
        maxima["lb_minus_exact"] = max(maxima["lb_minus_exact"], float(np.max(delta)))

        progressive_support = offset_upper + np.sum(norm_supports, axis=1)
        previous = np.maximum(0.0, current + ell**2 - 2.0 * ell * progressive_support)
        for block in range(args.blocks):
            progressive_support += selected_supports[:, block] - norm_supports[:, block]
            refined = np.maximum(0.0, current + ell**2 - 2.0 * ell * progressive_support)
            drop = previous - refined
            unsafe = refined - exact
            counters["progressive_monotonicity_violations"] += int(np.count_nonzero(drop > tolerance))
            counters["progressive_safety_violations"] += int(np.count_nonzero(unsafe > tolerance))
            maxima["progressive_drop"] = max(maxima["progressive_drop"], float(np.max(drop)))
            maxima["progressive_lb_minus_exact"] = max(
                maxima["progressive_lb_minus_exact"], float(np.max(unsafe))
            )
            previous = refined

        block_radius = np.zeros(n, dtype=np.float64)
        for block in range(args.blocks):
            sl = slice(block * width, (block + 1) * width)
            block_radius += np.linalg.norm(w[:, sl], axis=1) * eps[:, block]
        global_radius = np.linalg.norm(w, axis=1) * np.linalg.norm(eps, axis=1)
        dominance = block_radius - global_radius
        counters["block_dominance_violations"] += int(np.count_nonzero(dominance > tolerance))
        maxima["block_minus_global_radius"] = max(
            maxima["block_minus_global_radius"], float(np.max(dominance))
        )

        lo = ell * (1.0 - 0.01 * rng.random(n))
        hi = ell * (1.0 + 0.01 * rng.random(n))
        minimizer = np.clip(support, lo, hi)
        interval_lower = np.maximum(0.0, current + minimizer**2 - 2.0 * minimizer * support)
        interval_delta = interval_lower - exact
        counters["interval_length_violations"] += int(np.count_nonzero(interval_delta > tolerance))
        maxima["interval_lb_minus_exact"] = max(
            maxima["interval_lb_minus_exact"], float(np.max(interval_delta))
        )

        # A diagonal, deliberately non-orthogonal R.  The operator-norm
        # defect certificate must close the lost inner-product term.
        scales = np.linspace(0.85, 1.15, d, dtype=np.float64)
        defect = float(np.max(np.abs(1.0 - scales**2)))
        rw = w * scales
        ry = u * scales
        transformed_inner = np.sum(rw * ry, axis=1)
        nonorth_support = offset_upper + transformed_inner + defect * np.linalg.norm(w, axis=1)
        nonorth_lower = np.maximum(0.0, current + ell**2 - 2.0 * ell * nonorth_support)
        nonorth_delta = nonorth_lower - exact
        counters["nonorthogonal_transform_violations"] += int(
            np.count_nonzero(nonorth_delta > tolerance)
        )
        maxima["nonorthogonal_lb_minus_exact"] = max(
            maxima["nonorthogonal_lb_minus_exact"], float(np.max(nonorth_delta))
        )
        completed += n

    violation_total = sum(counters.values())
    report = {
        "format": "etv1_property_test",
        "format_version": 1,
        "trials": args.trials,
        "batch_size": args.batch_size,
        "dimension": args.dimension,
        "blocks": args.blocks,
        "seed": args.seed,
        "tolerance": tolerance,
        "counters": counters,
        "maxima": maxima,
        "naive_length_rounding_counterexample": {
            "support_upper": -1.0,
            "length_interval": [1.0, 2.0],
            "naive_term": 1.0 - 2.0 * 2.0 * -1.0,
            "true_term_at_length_1": 1.0 - 2.0 * 1.0 * -1.0,
        },
        "status": "pass" if violation_total == 0 else "fail",
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0 if violation_total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

