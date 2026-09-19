"""Join offline decision and component-cost evidence without an early gate."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import read_json, sha256, write_json


def bootstrap_delta(base: pd.DataFrame, candidate: pd.DataFrame,
                    draws: int = 2000) -> dict:
    joined = base.merge(candidate, on="query_id", suffixes=("_base", "_candidate"),
                        validate="one_to_one")
    if len(joined) != len(base) or len(joined) != len(candidate):
        raise ValueError("paired query IDs differ")
    rng = np.random.default_rng(20260919)
    sampled = rng.integers(0, len(joined), size=(draws, len(joined)))
    n = joined["N_base"].to_numpy()[sampled].sum(axis=1)
    if not np.all(n > 0):
        raise ValueError("empty bootstrap query sample")
    result = {}
    for metric in ("TP", "FP"):
        delta = (joined[f"{metric}_candidate"].to_numpy()[sampled].sum(axis=1) -
                 joined[f"{metric}_base"].to_numpy()[sampled].sum(axis=1)) / n
        result[metric + "_per_N_delta"] = {
            "estimate": float((joined[f"{metric}_candidate"].sum() -
                                joined[f"{metric}_base"].sum()) /
                               joined["N_base"].sum()),
            "ci95": [float(value) for value in np.quantile(delta, [0.025, 0.975])]
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    replay, selection, kernel = (read_json(args.replay), read_json(args.selection),
                                 read_json(args.kernel))
    counts = pd.read_csv(Path(args.replay).parent / "query_counts.csv")
    counts = counts.loc[counts["split"] == "selection"]
    baseline = counts.loc[(counts["policy"] == "pq-beta") &
                          np.isclose(counts["threshold"],
                                     selection["beta_development"])]
    results = []
    for choice in selection["choices"]:
        for seed_row in choice["seed_results"]:
            seed = seed_row["seed"]
            direct = counts.loc[(counts["policy"] == "residual-direct") &
                                (counts["bits"] == choice["bits"]) &
                                (counts["seed"] == seed)]
            if len(direct) != len(baseline):
                raise ValueError("missing selected query counts")
            results.append({"bits": choice["bits"], "seed": seed,
                            "direct_vs_beta": bootstrap_delta(baseline, direct)})
    output = {"schema_version": 1, "selected_bits": selection["selected_bits"],
              "bootstrap_draws": 2000, "selection_query_results": results,
              "kernel_cases": kernel["cases"],
              "interpretation": "A/B diagnostics; C/D active results determine final QPS conclusion",
              "replay_sha256": sha256(args.replay),
              "selection_sha256": sha256(args.selection),
              "kernel_sha256": sha256(args.kernel)}
    write_json(args.output, output)


if __name__ == "__main__":
    main()
