#include <cmath>
#include <iostream>
#include <queue>
#include <random>
#include <stdexcept>
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

int main() {
#ifndef HNSWLIB_ENABLE_BASELINE_TRACE
    std::cerr << "HNSWLIB_ENABLE_BASELINE_TRACE is required" << std::endl;
    return 2;
#else
    const size_t n = 256;
    const size_t dimension = 1024;
    const size_t k = 10;
    std::mt19937 rng(7);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> base(n * dimension);
    for (size_t i = 0; i < base.size(); ++i) base[i] = normal(rng);

    hnswlib::L2Space space(dimension);
    hnswlib::HierarchicalNSW<float> index(&space, n, 16, 100, 7);
    for (size_t i = 0; i < n; ++i) index.addPoint(base.data() + i * dimension, i);
    index.setEf(50);

    double maximum_relative_geometry_error = 0.0;
    for (size_t q = 0; q < 8; ++q) {
        const float* query = base.data() + q * dimension;
        std::priority_queue<std::pair<float, hnswlib::labeltype> > baseline = index.searchKnn(query, k);

        hnswlib::BaselineTraceConfig config;
        config.query_id = q;
        config.dimension = dimension;
        config.collect_geometry = true;
        config.dco_sample_modulus = 1;
        hnswlib::BaselineTraceCollector trace(config);
        std::priority_queue<std::pair<float, hnswlib::labeltype> > traced =
            index.searchKnnWithTrace(query, k, trace);

        if (!sameQueue(baseline, traced)) throw std::runtime_error("Tracing changed search results");
        if (trace.summary.n_edge_scan != trace.summary.n_duplicate + trace.summary.n_unique_neighbor) {
            throw std::runtime_error("Edge scan accounting mismatch");
        }
        if (trace.summary.n_unique_neighbor != trace.summary.n_dist) {
            throw std::runtime_error("Distance accounting mismatch");
        }
        if (trace.records.size() != trace.summary.n_dist) {
            throw std::runtime_error("Full trace did not record every DCO");
        }
        for (size_t i = 0; i < trace.records.size(); ++i) {
            const hnswlib::BaselineDcoRecord& record = trace.records[i];
            if (record.threshold_valid_before && !std::isfinite(record.relative_margin)) {
                throw std::runtime_error("Valid threshold has invalid margin");
            }
            if (!record.threshold_valid_before && std::isfinite(record.relative_margin)) {
                throw std::runtime_error("Invalid threshold has a margin");
            }
            if (record.geometry_valid) {
                const double reconstructed = record.dist_qc + record.edge_length_cd * record.edge_length_cd -
                    2.0 * record.edge_dot_qcd;
                const double relative_error = std::fabs(reconstructed - record.dist_qd) /
                    std::max(1.0, std::fabs(record.dist_qd));
                maximum_relative_geometry_error = std::max(maximum_relative_geometry_error, relative_error);
            }
        }
    }

    if (maximum_relative_geometry_error > 5e-4) {
        throw std::runtime_error("Geometry identity mismatch");
    }
    std::cout << "baseline_trace_test_ok maximum_relative_geometry_error="
              << maximum_relative_geometry_error << std::endl;
    return 0;
#endif
}
