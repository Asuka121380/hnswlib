# P1 platform x build environment matrix

This is the first follow-up to P0. It isolates the primary matched-recall pair
(`beta=1.45`, no retry, legacy prefetch, `ef=500` versus baseline `ef=435`)
across two platforms and two build variants. Each cell builds on its requested
compute node and then runs five randomized full QPS blocks plus seven P0.3
component blocks under one logical CPU and local-NUMA binding.

The four cells must use the same clean Git commit, frozen data/index/sidecar,
resource profile, and configuration. Start with `exploratory-shared`; repeat a
selected conclusion with `formal-exclusive` only after the four-cell matrix is
complete. A native binary is never reused across nodes.

Example dry run for the existing testing cell:

```bash
bash scripts/v0/performance_ready/submit_environment_cell.sh \
  --platform testing --build-variant portable \
  --resource-profile exploratory-shared \
  --partition testing --qos normal --nodelist gpusrv-2 \
  --time-limit 01:30:00 --memory 24G --dry-run
```

Submit the same command once per cell, changing `--platform`,
`--build-variant`, and the scheduler routing for WEIRDO versus testing. Use a
distinct run root for every cell. Do not infer WEIRDO partition/QoS values from
old reports; inspect `sinfo` and current policy first.

After all four jobs contain `COMPLETE`, summarize them:

```bash
python scripts/v0/performance_ready/summarize_environment_matrix.py \
  --cell weirdo=portable=/absolute/run/weirdo-portable \
  --cell weirdo=native=/absolute/run/weirdo-native \
  --cell testing=portable=/absolute/run/testing-portable \
  --cell testing=native=/absolute/run/testing-native \
  --output /absolute/run/matrix.csv \
  --report /absolute/run/matrix.json
```

The report defines build effect as native minus portable speedup percentage
points within each platform. Platform effect is the second platform supplied
on the command line minus the first, recorded explicitly in
`platform_order_for_effect`.
Interpret these effects together with `exact_l2_ns`, `fast_estimator_ns`, and
`fast_lut_build_ns`; the matrix establishes causal separation of build and
platform, but does not by itself separate cache, memory, and microarchitecture.
