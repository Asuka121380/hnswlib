#include <iostream>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "hnswlib/hnswlib.h"

bool metricsAreZero(const hnswlib::V0QueryMetrics& metrics) {
    return metrics.bound_evaluated == 0 &&
        metrics.bound_pruned == 0 &&
        metrics.exact_fallback == 0 &&
        metrics.exact_only_fallback == 0 &&
        metrics.exact_distance_saved == 0 &&
        metrics.lower_bound_violation == 0 &&
        metrics.false_prune == 0 &&
        metrics.exact_distance_computed == 0 &&
        metrics.expanded_nodes == 0 &&
        metrics.edge_scans == 0 &&
        metrics.duplicate_encounters == 0 &&
        metrics.approx_eligible_first_visits == 0 &&
        metrics.approx_first_pruned == 0 &&
        metrics.approx_retry_encountered == 0 &&
        metrics.approx_retry_exact_distance == 0 &&
        metrics.approx_retry_inserted_candidate == 0 &&
        metrics.approx_retry_inserted_result == 0 &&
        metrics.approx_estimator_fallback == 0 &&
        metrics.approx_state_bytes == 0;
}

template<typename Callable>
bool throwsException(Callable callable) {
    try {
        callable();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

class EvenLabelFilter : public hnswlib::BaseFilterFunctor {
 public:
    bool operator()(hnswlib::labeltype id) {
        return id % 2 == 0;
    }
};

int main() {
#ifndef HNSWLIB_ENABLE_EDGE_QUANT_V0
    std::cerr << "HNSWLIB_ENABLE_EDGE_QUANT_V0 is required" << std::endl;
    return 2;
#else
    const size_t n = 256;
    const size_t dimension = 32;
    const size_t k = 10;
    std::mt19937 rng(17);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> base(n * dimension);
    for (size_t i = 0; i < base.size(); ++i) base[i] = normal(rng);

    hnswlib::L2Space space(dimension);
    hnswlib::HierarchicalNSW<float> index(&space, n, 16, 100, 17);
    for (size_t i = 0; i < n; ++i) {
        index.addPoint(base.data() + i * dimension, i);
    }
    index.setEf(50);
    EvenLabelFilter even_label_filter;

    const float* query = base.data();
    const std::priority_queue<std::pair<float, hnswlib::labeltype> > baseline =
        index.searchKnn(query, k);
    const std::priority_queue<std::pair<float, hnswlib::labeltype> >
        filtered_baseline =
            index.searchKnn(query, k, &even_label_filter);
    if (baseline.empty() || filtered_baseline.empty()) {
        throw std::runtime_error("Baseline search unexpectedly returned no results");
    }

    hnswlib::V0QueryMetrics metrics;
    metrics.bound_evaluated = 1;
    if (!throwsException([&index, query, k, &metrics]() {
            (void)index.searchKnnV0(query, k, &metrics);
        })) {
        throw std::runtime_error(
            "V0 search accepted missing query metadata");
    }
    if (!metricsAreZero(metrics)) {
        throw std::runtime_error(
            "Rejected V0 search did not reset query metrics");
    }

    metrics.bound_evaluated = 1;
    if (!throwsException(
            [&index, query, k, &metrics, &even_label_filter]() {
                (void)index.searchKnnV0(
                    query, k, &metrics, &even_label_filter);
            })) {
        throw std::runtime_error(
            "Filtered V0 search accepted missing query metadata");
    }
    if (!metricsAreZero(metrics)) {
        throw std::runtime_error(
            "Rejected filtered V0 search did not reset query metrics");
    }

    std::cout << "v0_feature_isolation_test_ok" << std::endl;
    return 0;
#endif
}
