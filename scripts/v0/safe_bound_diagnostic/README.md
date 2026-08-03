# V0 safe-bound radius diagnostic

This package implements milestone M0/M1 of the safe-bound supplementary
experiment plan. It adds schema-v2 radius-component records without changing
the default V0 lower bound or enabling real pruning.

## Files

- `SCHEMA.md`: schema-v2 field definitions and closure equations.
- `analyze_radius_components.py`: validates records, derives candidate metrics,
  writes Parquet/CSV/JSON/Markdown, and creates diagnostic figures.
- `test_analyze_radius_components.py`: positive and negative synthetic tests.
- `write_run_manifest.py`: records commit, inputs, hashes, build flags and Slurm
  identity in a machine-readable manifest.
- `run_safe_bound_diagnostic.slurm`: build, C++/Python gates, 100-query run, and
  optional 1000-query follow-up.
- `IMPLEMENTATION_STATUS.md`: implemented, verified and gated work.

## Local or login-node dependency test

Use the Python environment that contains NumPy, pandas, Matplotlib and PyArrow:

```bash
export ANALYSIS_PYTHON="$HOME/IndividualProject/envs/baseline-trace-py312-v2/bin/python"
"$ANALYSIS_PYTHON" \
  scripts/v0/safe_bound_diagnostic/test_analyze_radius_components.py
```

## Cluster submission

Run Stage A only first:

```bash
export PATH="$HOME/IndividualProject/envs/v0-pq/bin:$PATH"
export ANALYSIS_PYTHON="$HOME/IndividualProject/envs/baseline-trace-py312-v2/bin/python"
export RUN_ROOT="$HOME/IndividualProject/results/v0_safe_bound_diagnostic/gist1m/<run-id>"
export RUN_STAGE_B=0

sbatch --parsable --export=ALL \
  scripts/v0/safe_bound_diagnostic/run_safe_bound_diagnostic.slurm
```

After Stage A passes, use a new `RUN_ROOT` and set `RUN_STAGE_B=1` to include
the 1000-query run. The script refuses to overwrite a completed stage.

The candidate columns for cap, blockwise and representation-level `LB*` are
reserved but empty in this milestone. They must only be populated after their
uncertainty sets and directed-rounding rules are proven and tested.
