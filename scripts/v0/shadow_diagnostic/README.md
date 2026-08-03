# V0 minimal shadow diagnostic implementation

This package implements the two-stage GIST1M diagnostic described in
`GIST1M_MINIMAL_SHADOW_DIAGNOSTIC_PLAN_20260803.md`.

## Files

- `run_minimal_shadow_diagnostic.slurm`: builds an observe-only shadow runner,
  runs the 100-query gate, and only then runs the 1000-query diagnostic.
- `analyze_shadow_diagnostic.py`: validates closure/safety/results, derives
  candidate metrics, writes Parquet/CSV/JSON/Markdown, and creates five PNG/PDF
  figure pairs.
- `test_analyze_shadow_diagnostic.py`: dependency and regression smoke test
  using synthetic shadow data.

## Before submitting

Install/copy this directory to:

```text
$HOME/IndividualProject/src/hnswlib/scripts/v0/shadow_diagnostic
```

Then run the synthetic test in the same Python environment intended for the
cluster experiment:

```bash
python3 scripts/v0/shadow_diagnostic/test_analyze_shadow_diagnostic.py
```

The Python environment must provide `numpy`, `pandas`, `matplotlib`, `pyarrow`,
and a Parquet engine. Set `ANALYSIS_PYTHON` when these are installed in a
non-default environment.

Submit from the repository root:

```bash
sbatch scripts/v0/shadow_diagnostic/run_minimal_shadow_diagnostic.slurm
```

Paths can be overridden with `REPO_ROOT`, `BUILD_ROOT`, `RUN_ROOT`,
`DATASET_CONFIG`, `INDEX_PATH`, `SIDECAR_PATH`, `ANALYSIS_PYTHON`, and
`ANALYSIS_SCRIPT`. Set `RUN_STAGE_B=0` to stop after the 100-query stage.

The job refuses to overwrite a completed stage. The expected final results are
under `$HOME/IndividualProject/results/v0_shadow_diagnostic/gist1m/20260803-minimal`.
