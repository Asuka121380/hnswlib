"""Freeze a finite active matrix using development thresholds and selection data."""

import argparse
from pathlib import Path

import pandas as pd

from common import read_json, sha256, write_json


def best_at_anchor(rows: pd.DataFrame, anchor: float) -> pd.Series:
    feasible = rows.loc[rows["FP_per_N"] <= anchor]
    if feasible.empty:
        return rows.sort_values(["FP_per_N", "TP_per_N"], ascending=[True, False]).iloc[0]
    return feasible.sort_values(["TP_per_N", "FP_per_N"],
                                ascending=[False, True]).iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fp-anchor", type=float, default=0.001)
    args = parser.parse_args()
    manifest = read_json(args.input)
    if set(manifest["splits"]) != {"development", "selection"}:
        raise ValueError("candidate selection requires development and selection only")
    frame = pd.read_csv(Path(args.input).parent / "frontier.csv")
    development = frame.loc[frame["split"] == "development"]
    selection = frame.loc[frame["split"] == "selection"]
    base_dev = best_at_anchor(development.loc[development["policy"] == "pq-beta"],
                              args.fp_anchor)
    base = selection.loc[(selection["policy"] == "pq-beta") &
                         (selection["threshold"] == base_dev["threshold"])].iloc[0]
    choices = []
    for bits in sorted(development.loc[development["bits"] > 0, "bits"].unique()):
        seed_rows = []
        for seed in sorted(development.loc[development["bits"] == bits, "seed"].unique()):
            direct = selection.loc[(selection["policy"] == "residual-direct") &
                                   (selection["bits"] == bits) &
                                   (selection["seed"] == seed)].iloc[0]
            theta_dev = best_at_anchor(development.loc[
                (development["policy"] == "residual-threshold") &
                (development["bits"] == bits) &
                (development["seed"] == seed)], args.fp_anchor)
            theta = selection.loc[(selection["policy"] == "residual-threshold") &
                                  (selection["bits"] == bits) &
                                  (selection["seed"] == seed) &
                                  (selection["threshold"] == theta_dev["threshold"])].iloc[0]
            seed_rows.append({"seed": int(seed),
                              "direct_tp_delta": float(direct["TP_per_N"] - base["TP_per_N"]),
                              "direct_fp_delta": float(direct["FP_per_N"] - base["FP_per_N"]),
                              "theta": float(theta_dev["threshold"]),
                              "threshold_tp_delta": float(theta["TP_per_N"] - base["TP_per_N"]),
                              "threshold_fp_delta": float(theta["FP_per_N"] - base["FP_per_N"])})
        seed42 = next(row for row in seed_rows if row["seed"] == 42)
        median_direct = sorted(row["direct_tp_delta"] for row in seed_rows)[1]
        choices.append({"bits": int(bits), "seed_results": seed_rows,
                        "primary_theta": seed42["theta"],
                        "median_direct_tp_delta": median_direct})
    choices.sort(key=lambda row: row["median_direct_tp_delta"], reverse=True)
    selected = [row["bits"] for row in choices[:2]]
    if not any(row["median_direct_tp_delta"] > 0 for row in choices):
        selected = [64]
    result = {"schema_version": 1, "source_manifest": str(Path(args.input).resolve()),
              "fp_anchor": args.fp_anchor,
              "beta_development": float(base_dev["threshold"]),
              "beta_selection_tp_per_N": float(base["TP_per_N"]),
              "primary_seed": 42, "selected_bits": selected,
              "active_policies": ["residual-direct", "residual-threshold"],
              "choices": choices,
              "selection_is_not_new_heldout": True}
    write_json(args.output, result)
    write_json(Path(args.output).with_suffix(".complete.json"),
               {"input_sha256": sha256(args.input), "output_sha256": sha256(args.output)})


if __name__ == "__main__":
    main()
