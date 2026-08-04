#include <cstddef>
#include <cstdint>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "v0_ratio_test_utils.h"

namespace {

class RecordingCollector : public hnswlib::V0RatioShadowCollector {
 public:
    void append(const hnswlib::V0RatioShadowRecord& record) override {
        records.push_back(record);
    }
    std::vector<hnswlib::V0RatioShadowRecord> records;
};

void testObserveOnlyAndFalsePruneExposure() {
    const uint32_t dimension = 8U;
    const size_t node_count = 24U;
    const size_t k = 3U;
    const std::string sidecar_path =
        "v0_ratio_shadow_validation_test.v0meta";
    v0_test::FileCleanup cleanup(sidecar_path);
    std::mt19937 rng(441U);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> base(node_count * dimension);
    for (size_t i = 0U; i < base.size(); ++i) base[i] = normal(rng);
    hnswlib::L2Space space(dimension);
    hnswlib::HierarchicalNSW<float> index(
        &space, node_count, 4U, 40U, 441U);
    for (size_t i = 0U; i < node_count; ++i) {
        index.addPoint(base.data() + i * dimension, i);
    }
    v0_ratio_test::writeExactDirectionSidecar(
        sidecar_path, index, dimension);
    index.loadEdgeQuantV0Metadata(sidecar_path);

    index.setEf(node_count + 1U);
    hnswlib::V0RatioQueryMetrics pre_threshold;
    const hnswlib::V0RatioCalibrator safe =
        hnswlib::V0RatioCalibrator::forTesting("safe", 1.0e-4, 0.5);
    const std::priority_queue<std::pair<float, hnswlib::labeltype> >
        baseline_pre = index.searchKnn(base.data(), k);
    const std::priority_queue<std::pair<float, hnswlib::labeltype> >
        shadow_pre = index.searchKnnV0RatioShadow(
            base.data(), k, safe, &pre_threshold);
    v0_test::require(
        v0_ratio_test::sameQueue(baseline_pre, shadow_pre) &&
            pre_threshold.ratio_bound_evaluated == 0U &&
            pre_threshold.ratio_exact_fallback > 0U &&
            pre_threshold.exact_distance_saved == 0U,
        "ratio shadow used a bound before the heap was full");

    index.setEf(6U);
    RecordingCollector safe_collector;
    uint64_t evaluated = 0U;
    for (size_t query_id = 0U; query_id < node_count; ++query_id) {
        const float* query = base.data() + query_id * dimension;
        const std::priority_queue<std::pair<float, hnswlib::labeltype> >
            baseline = index.searchKnn(query, k);
        hnswlib::V0RatioQueryMetrics metrics;
        const std::priority_queue<std::pair<float, hnswlib::labeltype> >
            shadow = index.searchKnnV0RatioShadow(
                query, k, safe, &metrics, nullptr,
                &safe_collector, query_id);
        v0_test::require(
            v0_ratio_test::sameQueue(baseline, shadow),
            "ratio shadow changed HNSW results");
        v0_test::require(
            metrics.exact_distance_saved == 0U,
            "ratio shadow reported saved exact distances");
        evaluated += metrics.ratio_bound_evaluated;
    }
    v0_test::require(
        evaluated > 0U && safe_collector.records.size() == evaluated,
        "ratio shadow records and counters do not close");

    const hnswlib::V0RatioCalibrator unsafe =
        hnswlib::V0RatioCalibrator::forTesting(
            "forced_false_prune", -1.0e9, 0.5);
    uint64_t false_prunes = 0U;
    for (size_t query_id = 0U; query_id < node_count; ++query_id) {
        const float* query = base.data() + query_id * dimension;
        const std::priority_queue<std::pair<float, hnswlib::labeltype> >
            baseline = index.searchKnn(query, k);
        hnswlib::V0RatioQueryMetrics metrics;
        const std::priority_queue<std::pair<float, hnswlib::labeltype> >
            shadow = index.searchKnnV0RatioShadow(
                query, k, unsafe, &metrics, nullptr, nullptr, query_id);
        v0_test::require(
            v0_ratio_test::sameQueue(baseline, shadow),
            "unsafe shadow changed control flow");
        false_prunes += metrics.ratio_false_prune;
    }
    v0_test::require(
        false_prunes > 0U,
        "constructed shadow error did not reach the false-prune counter");
}

}  // namespace

int main() {
#if !defined(HNSWLIB_ENABLE_V0_RATIO_ESTIMATOR) || \
    !defined(HNSWLIB_ENABLE_V0_RATIO_SHADOW)
    std::cerr << "V0 ratio shadow is required" << std::endl;
    return 2;
#else
    testObserveOnlyAndFalsePruneExposure();
    std::cout << "v0_ratio_shadow_validation_test_ok" << std::endl;
    return 0;
#endif
}
