#include <iostream>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "hnswlib/hnswlib.h"

template<typename Queue>
bool sameQueue(Queue left, Queue right) {
    if (left.size() != right.size()) return false;
    while (!left.empty()) {
        if (left.top() != right.top()) return false;
        left.pop();
        right.pop();
    }
    return true;
}

bool metricsAreZero(const hnswlib::V0QueryMetrics& metrics) {
    return metrics.bound_evaluated == 0 &&
        metrics.bound_pruned == 0 &&
        metrics.exact_fallback == 0 &&
        metrics.exact_only_fallback == 0 &&
        metrics.exact_distance_saved == 0 &&
        metrics.lower_bound_violation == 0 &&
        metrics.false_prune == 0;
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

    for (size_t q = 0; q < 16; ++q) {
        const float* query = base.data() + q * dimension;
        const std::priority_queue<std::pair<float, hnswlib::labeltype> > baseline =
            index.searchKnn(query, k);

        hnswlib::V0QueryMetrics metrics;
        metrics.bound_evaluated = 1;
        const std::priority_queue<std::pair<float, hnswlib::labeltype> > v0 =
            index.searchKnnV0(query, k, &metrics);

        if (!sameQueue(baseline, v0)) {
            throw std::runtime_error("V0 isolation scaffold changed search results");
        }
        if (!metricsAreZero(metrics)) {
            throw std::runtime_error("V0 isolation scaffold produced non-zero metrics");
        }

        const std::priority_queue<std::pair<float, hnswlib::labeltype> > filtered_baseline =
            index.searchKnn(query, k, &even_label_filter);
        const std::priority_queue<std::pair<float, hnswlib::labeltype> > filtered_v0 =
            index.searchKnnV0(query, k, &metrics, &even_label_filter);
        if (!sameQueue(filtered_baseline, filtered_v0)) {
            throw std::runtime_error("Filtered V0 isolation scaffold changed search results");
        }
        if (!metricsAreZero(metrics)) {
            throw std::runtime_error("Filtered V0 isolation scaffold produced non-zero metrics");
        }
    }

    std::cout << "v0_feature_isolation_test_ok" << std::endl;
    return 0;
#endif
}
