# EQ-RCP Stage 2 Runbook

## Question

Stage 2 tests whether a fixed 32-byte direct-edge code preserves enough information for the operational projection `Y=X^T Delta`. It is an offline representation ceiling, not a pruning or recall experiment.

The mandatory comparison matrix is:

- direct-Delta PQ;
- direct-Delta iterative OPQ;
- diagonal OAE-PQ;
- block-diagonal OAE-PQ;
- full-metric OAE-PQ as an offline ceiling;
- historical direction-PQ with exact edge length.

All methods use 32 uint8 subcodes. `ell^2` is outside this code budget under the frozen v1 runtime contract.

## Leakage boundary

- Quantizer codebooks, OPQ rotations, OAE metrics, regularization, and objective weights use only `quantizer_train` queries.
- The diagnostic risk cutoff uses only `decoder_train` queries.
- Projection tails and paired-query comparisons use only `evaluation` queries.
- Encoder assignment sees only `Delta` and frozen static representation parameters. It never sees query, threshold, or label fields.
- `score_oracle_coverage` selects the best scalar reconstruction-margin cutoff on evaluation labels.
- `code_oracle_coverage` first trains a flexible decoder on separate `decoder_train` queries using code bytes, per-block projection contributions, query-block norms, and operational scalars. Its final risk cutoff is selected on evaluation labels, so it remains an optimistic diagnostic ceiling rather than a calibrated guarantee.

## Local development run

```powershell
./scripts/eq_rcp/run_stage2_development.ps1
```

The local configuration uses real 960-dimensional GIST1M edges and a deterministic query-level subsample. It keeps the real 256-centroid, 32-subquantizer code contract. Because the Stage 1 input is development-only and the Stage 2 selection is subsampled, the only allowed final decision is `PENDING_FORMAL_STAGE2`, regardless of diagnostic metrics.

## Formal cluster run

1. Complete formal Stage 1 with a fresh query population, exact frozen native index, and adjacency status `PASS`.
2. Copy `configs/eq_rcp/stage2_cluster_template.json` and fill the formal Stage 1 path/hash.
3. Keep all `max_queries` fields null, all six methods, all registered seeds, and `allow_formal_gate_decision=true`.
4. Freeze thread counts for reproducibility.
5. Run:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/eq_rcp/run_stage2_ceiling.py \
  --config configs/eq_rcp/stage2_cluster.json \
  --output-dir results/eq_rcp/stage2/formal/run-a
python scripts/eq_rcp/validate_stage2_ceiling.py \
  --config configs/eq_rcp/stage2_cluster.json \
  --result-dir results/eq_rcp/stage2/formal/run-a
```

Repeat in a second directory and compare `stage2_results_semantic_sha256`.

## Pre-registered Gate

The best structured OAE candidate (diagonal or block) is paired against the best MSE baseline (PQ or OPQ):

- aggregate Q99 absolute projection error reduction at least 10%;
- paired-query bootstrap improvement CI lower bound greater than zero;
- optimistic code-oracle coverage gain at least 2 percentage points;
- optimistic ideal coverage at least 10%;
- positive Q99 reduction for every registered seed.

Only a full formal run satisfying all conditions may emit `GO_CPP`. A full formal failure emits `NO_GO_CPP`. Every development, pilot, subsampled, or Stage-1-adjacency-pending run emits `PENDING_FORMAL_STAGE2`.

## Interpretation

- Full OAE and structured OAE both fail: the current 32B direct-edge OAE representation is not sufficient.
- Full OAE succeeds but structured OAE fails: the operational metric is informative, but the deployable structure is inadequate.
- Projection improves but decision ceiling does not: representation still lacks threshold-relevant decision information.
- Structured OAE passes all formal criteria: proceed to Stage 3; this still does not prove real pruning safety or speed.
