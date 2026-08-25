# Frozen full-range V0 Recall/DCO/QPS trade-off matrix.
# The 0.05 grid covers [1, 2]; four 0.025 points refine the known frontier.

readonly FULL_BETA_VALUES="1.00,1.05,1.10,1.15,1.20,1.25,1.30,1.35,1.40,1.45,1.50,1.525,1.55,1.575,1.60,1.625,1.65,1.675,1.70,1.75,1.80,1.85,1.90,1.95,2.00"
readonly FULL_BETA_MODES="approx-no-retry,approx-retry"
readonly FULL_BETA_PREFETCHES="legacy,gate"
readonly FULL_BETA_QUERY_START=0
readonly FULL_BETA_QUERY_COUNT=1000
readonly FULL_BETA_K=10
readonly FULL_BETA_EF_SEARCH=200
readonly FULL_BETA_WARMUP_QUERIES=100
readonly FULL_BETA_WITHIN_PROCESS_REPEATS=5
readonly FULL_BETA_BLOCKS=5
readonly FULL_BETA_SEED=20260825
