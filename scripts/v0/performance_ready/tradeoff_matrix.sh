# Frozen V0 trade-off experiment matrix.
# Changes to this file change the scientific experiment and require review.

readonly TRADEOFF_BETAS="1.30,1.40,1.45,1.50,1.525,1.55,1.575,1.60,1.625,1.65,1.675,1.70"
readonly TRADEOFF_MODES="approx-no-retry,approx-retry"
readonly TRADEOFF_K=10
readonly TRADEOFF_EF_SEARCH=200

configure_tradeoff_split() {
  case "$1" in
    validation)
      TRADEOFF_QUERY_START=300
      TRADEOFF_QUERY_COUNT=300
      ;;
    heldout)
      TRADEOFF_QUERY_START=600
      TRADEOFF_QUERY_COUNT=400
      ;;
    *)
      echo "Trade-off split must be validation or heldout" >&2
      return 2
      ;;
  esac
}
