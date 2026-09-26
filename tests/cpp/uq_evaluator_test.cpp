#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "tools/edge_estimation/evaluator.h"
#include "tools/edge_estimation/backends/pq_packed.h"
#include "hnswlib/edge_estimation/score_bridge.h"

namespace {

uq::EventRecord blank(uq::EventKind kind, uint64_t id) {
    uq::EventRecord event{};
    event.kind = kind; event.event_id = id; event.query_id = 5;
    event.graph_layer = kind == uq::EventKind::Candidate ? 0 : -1;
    event.expansion_id = std::numeric_limits<uint64_t>::max();
    event.edge_id = std::numeric_limits<uint64_t>::max();
    event.source_id = event.target_id = event.neighbor_slot =
        std::numeric_limits<uint32_t>::max();
    return event;
}

class Kernel {
 public:
    Kernel() : query_calls(0), source_calls(0) {}
    void prepareQuery(uint64_t) { ++query_calls; }
    void prepareSource(const uq::EventRecord&) { ++source_calls; }
    double score(const uq::EventRecord& event) {
        if (event.edge_id == 3U) return std::numeric_limits<double>::quiet_NaN();
        return static_cast<double>(event.edge_id + 8U);
    }
    uint64_t query_calls, source_calls;
};

class PackedKernel {
 public:
    PackedKernel() : model_(3U, 4U), query_calls(0), source_calls(0) {
        lut_.resize(3U * 16U, 0.0f);
        for (size_t sub = 0; sub < 3U; ++sub)
            for (size_t code = 0; code < 16U; ++code)
                lut_[sub * 16U + code] = static_cast<float>(code) * 0.01f;
        records_.push_back(uq::packPqCodes(std::vector<uint8_t>{1, 2, 3}, 4U));
        records_.push_back(uq::packPqCodes(std::vector<uint8_t>{2, 3, 4}, 4U));
        records_.push_back(uq::packPqCodes(std::vector<uint8_t>{3, 4, 5}, 4U));
    }
    void prepareQuery(uint64_t) { ++query_calls; }
    void prepareSource(const uq::EventRecord&) { ++source_calls; }
    double score(const uq::EventRecord& event) {
        const size_t index = static_cast<size_t>(event.edge_id - 1U);
        const hnswlib::edge_estimation::DotEstimate dot =
            model_.estimate(lut_, &records_[index][0], records_[index].size());
        return hnswlib::edge_estimation::bridgeDotToSquaredDistance(
            event.d_current, 1.0, dot).squared_distance;
    }
    uint64_t query_calls, source_calls;
 private:
    uq::PackedPqModel model_;
    std::vector<float> lut_;
    std::vector<std::vector<uint8_t> > records_;
};

}  // namespace

int main() {
    std::vector<uq::EventRecord> events;
    events.push_back(blank(uq::EventKind::QueryBegin, 0));
    uq::EventRecord empty_source = blank(uq::EventKind::SourceBegin, 1);
    empty_source.expansion_id = 0; empty_source.source_id = 1;
    events.push_back(empty_source);
    uq::EventRecord source = blank(uq::EventKind::SourceBegin, 2);
    source.expansion_id = 1; source.source_id = 2; source.source_degree = 3;
    events.push_back(source);
    for (uint64_t edge = 1; edge <= 3; ++edge) {
        uq::EventRecord candidate = blank(uq::EventKind::Candidate, events.size());
        candidate.flags = uq::ThresholdValid | uq::FirstVisit | uq::ScoreSlotEligible;
        candidate.expansion_id = 1; candidate.edge_id = edge;
        candidate.source_id = 2; candidate.target_id = static_cast<uint32_t>(edge + 10);
        candidate.neighbor_slot = static_cast<uint32_t>(edge - 1);
        candidate.source_degree = 3;
        candidate.threshold_before = 10.0;
        events.push_back(candidate);
    }
    events.push_back(blank(uq::EventKind::QueryEnd, events.size()));
    uq::validateEvents(events);
    const std::vector<uq::LabelRecord> labels{{3, 12.0}, {4, 10.0}, {5, 15.0}};
    Kernel quality_kernel;
    const uq::QualityCounters quality =
        uq::evaluateQuality(events, labels, 1.0, quality_kernel);
    if (quality.decision_count_s != 3U || quality.tp != 0U || quality.fp != 0U ||
        quality.fn != 2U || quality.tn != 1U || quality.fallback_count != 1U)
        throw std::runtime_error("native quality confusion matrix mismatch");
    if (quality_kernel.query_calls != 1U || quality_kernel.source_calls != 1U)
        throw std::runtime_error("quality lifecycle count mismatch");
    Kernel timing_kernel;
    const uq::ReplayCounters timing = uq::replayOrderedEstimator(events, timing_kernel, 2U);
    if (timing.query_count != 2U || timing.source_count != 2U ||
        timing.eligible_count != 6U || timing.fallback_count != 2U)
        throw std::runtime_error("timing lifecycle count mismatch");
    if (timing_kernel.source_calls != 2U)
        throw std::runtime_error("empty source was prepared");
    PackedKernel packed_kernel;
    const uq::ReplayCounters packed_timing =
        uq::replayOrderedEstimator(events, packed_kernel, 1U);
    if (packed_timing.eligible_count != 3U || packed_timing.fallback_count != 0U ||
        packed_kernel.query_calls != 1U || packed_kernel.source_calls != 1U)
        throw std::runtime_error("packed PQ did not traverse the common evaluator");
    std::cout << "uq_evaluator_test_ok" << std::endl;
    return 0;
}
