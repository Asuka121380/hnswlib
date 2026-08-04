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

void testRealPruningAndMismatchDetection() {
    const uint32_t dimension = 8U;
    const size_t node_count = 24U;
    const size_t k = 3U;
    const std::string sidecar_path =
        "v0_ratio_real_pruning_test.v0meta";
    v0_test::FileCleanup cleanup(sidecar_path);
    std::mt19937 rng(771U);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> base(node_count * dimension);
    for (size_t i = 0U; i < base.size(); ++i) base[i] = normal(rng);
    hnswlib::L2Space space(dimension);
    hnswlib::HierarchicalNSW<float> index(
        &space, node_count, 4U, 40U, 771U);
    for (size_t i = 0U; i < node_count; ++i) {
        index.addPoint(base.data() + i * dimension, i);
    }
    v0_ratio_test::writeExactDirectionSidecar(
        sidecar_path, index, dimension);
    index.loadEdgeQuantV0Metadata(sidecar_path);
    const hnswlib::V0RatioCalibrator safe =
        hnswlib::V0RatioCalibrator::forTesting("safe", 1.0e-4, 0.5);

    index.setEf(node_count + 1U);
    hnswlib::V0RatioQueryMetrics pre_threshold;
    const std::priority_queue<std::pair<float, hnswlib::labeltype> >
        baseline_pre = index.searchKnn(base.data(), k);
    const std::priority_queue<std::pair<float, hnswlib::labeltype> >
        pruned_pre = index.searchKnnV0RatioPruned(
            base.data(), k, safe, &pre_threshold);
    v0_test::require(
        v0_ratio_test::sameQueue(baseline_pre, pruned_pre) &&
            pre_threshold.ratio_bound_evaluated == 0U &&
            pre_threshold.exact_distance_saved == 0U,
        "ratio real pruning used a pre-threshold bound");

    index.setEf(6U);
    uint64_t saved = 0U;
    uint64_t bound_evaluated = 0U;
    uint64_t current_lb_evaluated = 0U;
    for (size_t query_id = 0U; query_id < node_count; ++query_id) {
        const float* query = base.data() + query_id * dimension;
        const std::priority_queue<std::pair<float, hnswlib::labeltype> >
            baseline = index.searchKnn(query, k);
        hnswlib::V0RatioQueryMetrics metrics;
        const std::priority_queue<std::pair<float, hnswlib::labeltype> >
            pruned = index.searchKnnV0RatioPruned(
                query, k, safe, &metrics);
        v0_test::require(
            v0_ratio_test::sameQueue(baseline, pruned),
            "safe exact-direction ratio pruning changed results");
        v0_test::require(
            metrics.ratio_bound_pruned == metrics.exact_distance_saved,
            "ratio pruned/saved counters diverged");
        v0_test::require(
            metrics.ratio_current_lb_evaluated ==
                    metrics.ratio_invalid_fallback &&
                metrics.ratio_current_lb_valid +
                        metrics.ratio_current_lb_invalid ==
                    metrics.ratio_current_lb_evaluated &&
                metrics.ratio_current_lb_skipped_eligible ==
                    metrics.ratio_eligible &&
                metrics.ratio_current_lb_valid ==
                    metrics.ratio_current_lb_fallback,
            "lazy current-LB counters do not close");
#ifndef HNSWLIB_ENABLE_V0_RATIO_FINE_GRAINED_TIMING
        v0_test::require(
            metrics.current_lb_time_ns == 0U &&
                metrics.estimator_time_ns == 0U &&
                metrics.exact_distance_time_ns == 0U,
            "performance build executed fine-grained timing");
#endif
        saved += metrics.exact_distance_saved;
        bound_evaluated += metrics.ratio_bound_evaluated;
        current_lb_evaluated += metrics.ratio_current_lb_evaluated;
    }
    v0_test::require(saved > 0U, "ratio real pruning saved no distances");
    v0_test::require(
        bound_evaluated > 0U && current_lb_evaluated < bound_evaluated,
        "eligible ratio path did not skip eager current-LB evaluation");

    const hnswlib::V0RatioCalibrator unsafe =
        hnswlib::V0RatioCalibrator::forTesting(
            "forced_mismatch", -1.0e9, 0.5);
    uint64_t mismatch_queries = 0U;
    for (size_t query_id = 0U; query_id < node_count; ++query_id) {
        const float* query = base.data() + query_id * dimension;
        const std::priority_queue<std::pair<float, hnswlib::labeltype> >
            baseline = index.searchKnn(query, k);
        hnswlib::V0RatioQueryMetrics metrics;
        const std::priority_queue<std::pair<float, hnswlib::labeltype> >
            pruned = index.searchKnnV0RatioPruned(
                query, k, unsafe, &metrics);
        if (!v0_ratio_test::sameQueue(baseline, pruned)) {
            ++mismatch_queries;
        }
    }
    v0_test::require(
        mismatch_queries > 0U,
        "constructed pruning error did not produce a detectable mismatch");
}

}  // namespace

int main() {
#if !defined(HNSWLIB_ENABLE_V0_RATIO_ESTIMATOR) || \
    !defined(HNSWLIB_ENABLE_V0_RATIO_REAL_PRUNING)
    std::cerr << "V0 ratio real pruning is required" << std::endl;
    return 2;
#else
    testRealPruningAndMismatchDetection();
    std::cout << "v0_ratio_real_pruning_test_ok" << std::endl;
    return 0;
#endif
}
