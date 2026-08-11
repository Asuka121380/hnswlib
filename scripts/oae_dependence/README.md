# OAE edge/query-dependence validation

This directory implements the preregistered necessary-condition test for OAE. It does **not** train a quantizer or modify HNSW search. Static edge geometry is fitted on 500 `gist_learn` queries and evaluated once on 500 disjoint `gist_learn` queries.

## What is tested

For each search event on edge `c -> v`, let `x=q-c`. The primary test asks whether a training-only group based on the unit edge direction and edge length predicts the 32 frozen-probe energies `(P^T x)^2` on unseen queries better than a global mean. The secondary test asks the analogous question for `(x^T r_v0)^2`, where `r_v0` is the residual of the frozen V0 unit-direction PQ codebook.

All event weights sum to one inside each query. Confidence intervals resample queries, never individual events.

## Cluster workflow

Start from the repository root. Set the frozen V0 codebook explicitly; the script refuses an unset or missing path.

```bash
export PYTHON="$HOME/IndividualProject/envs/baseline-trace-py312-v2/bin/python"
export OAE_V0_CODEBOOK=/absolute/path/to/pq_m16_nbits8.v0pq
export RUN_ID="oae-dependence-$(date -u +%Y%m%d-%H%M%S)"
export RUN_ROOT="$PWD/results/oae_dependence/pilot/$RUN_ID"

"$PYTHON" scripts/oae_dependence/freeze_inputs.py \
  --config configs/oae_dependence/pilot_v1.json \
  --output-dir "$RUN_ROOT"
```

The generated trace dataset files contain placeholder ivecs only because the existing V0 trace runner requires a ground-truth file. Recall is ignored; no ground-truth value enters this experiment.

Run the existing trace-enabled runner twice with identical search settings:

```bash
export INDEX="$HOME/IndividualProject/datasets/gist1m/indexes/gist1m_M16_efc200_seed42_gitfd11efdb86c6.bin"
export RUNNER="$HOME/IndividualProject/build/hnsw-trace-on/real_data_trace_runner"

for ROLE in training validation; do
  "$RUNNER" \
    --dataset-config "$RUN_ROOT/${ROLE}_trace_dataset.json" \
    --mode trace --index-path "$INDEX" --output-dir "$RUN_ROOT/trace-$ROLE" \
    --run-id "$RUN_ID-$ROLE" --query-start 0 --query-count 500 \
    --k 10 --ef-search 100 --M 16 --ef-construction 200 --seed 42 \
    --threads 1 --dco-sample-modulus 1 --dco-sample-remainder 0 \
    --edge-samples 0 --collect-geometry true --collect-distance-timing false

  "$PYTHON" scripts/oae_dependence/build_search_event_dataset.py \
    --trace-csv "$RUN_ROOT/trace-$ROLE/dco_trace.csv" \
    --query-manifest "$RUN_ROOT/${ROLE}_query_manifest.json" \
    --output "$RUN_ROOT/${ROLE}_events.npz"
done
```

Run controls, the one frozen evaluation, and fail-closed validation:

```bash
"$PYTHON" scripts/oae_dependence/run_synthetic_tests.py \
  --output "$RUN_ROOT/synthetic_test_report.json"

"$PYTHON" scripts/oae_dependence/evaluate_dependence.py \
  --input-dir "$RUN_ROOT" \
  --training-events "$RUN_ROOT/training_events.npz" \
  --validation-events "$RUN_ROOT/validation_events.npz" \
  --output-dir "$RUN_ROOT"

"$PYTHON" scripts/oae_dependence/validate_run.py --run-dir "$RUN_ROOT"
```

The only scale-up decision is in `run_manifest.json`. A primary failure means stop the current edge-geometry-conditioned OAE direction; a pass only authorizes a small, separately preregistered quantizer study.

## Training-only global A stability diagnostic

After the formal run, the following independent diagnostic checks whether 500
training queries were enough to estimate a global OAE metric. It does not read
validation events, rerun HNSW, or train a quantizer.

```bash
"$PYTHON" scripts/oae_dependence/analyze_global_a_stability.py \
  --run-dir "$RUN_ROOT" \
  --training-events "$RUN_ROOT/training_events_nonzero.npz" \
  --config configs/oae_dependence/global_a_stability_v1.json \
  --output-dir "$RUN_ROOT/global-a-stability"
```

The report compares query-balanced diagonal and frozen-probe covariance
estimates at 50, 100, 250, and 500 queries, repeats 20 independent 250/250
split-half comparisons, and measures stability of the induced importance
ranking on 10,000 frozen V0 residuals.
