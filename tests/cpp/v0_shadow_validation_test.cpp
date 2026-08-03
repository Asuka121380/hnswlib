#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hnswlib/edge_quant_v0_numeric.h"
#include "v0_sidecar_test_utils.h"

namespace {

template<typename Queue>
bool sameQueue(Queue left, Queue right) {
    if (left.size() != right.size()) {
        return false;
    }
    while (!left.empty()) {
        if (left.top() != right.top()) {
            return false;
        }
        left.pop();
        right.pop();
    }
    return true;
}

class RecordingShadowCollector
    : public hnswlib::V0ShadowValidationCollector {
 public:
    void append(const hnswlib::V0ShadowRecord& record) {
        records.push_back(record);
    }

    std::vector<hnswlib::V0ShadowRecord> records;
};

void writeExactDirectionSidecar(
    const std::string& path,
    const hnswlib::HierarchicalNSW<float>& index,
    uint32_t dimension) {
    const hnswlib::V0Layer0GraphView graph =
        index.getV0Layer0GraphView();

    hnswlib::V0SidecarWriteSpec spec;
    spec.dimension = dimension;
    spec.pq_m = 1U;
    spec.pq_nbits = 8U;
    spec.pq_ksub = 256U;
    spec.pq_dsub = dimension;
    spec.node_count = static_cast<uint64_t>(graph.nodeCount());
    spec.training_metadata_json =
        "{\"purpose\":\"milestone9_shadow_validation_test\"}";
    spec.codebook_centroids.assign(
        static_cast<size_t>(spec.pq_ksub) * dimension,
        0.0f);
    spec.node_offsets.push_back(0U);
    for (size_t node = 0U; node < graph.nodeCount(); ++node) {
        spec.directed_edge_count += static_cast<uint64_t>(
            graph.neighbors(
                static_cast<hnswlib::tableint>(node)).size);
        spec.node_offsets.push_back(spec.directed_edge_count);
    }
    v0_test::require(
        spec.directed_edge_count <= spec.pq_ksub,
        "shadow test graph has too many directed edges");
    spec.base_index_sha256 =
        index.getV0SerializedIndexFingerprint();
    spec.adjacency_sha256 = graph.adjacencyFingerprint();

    std::vector<hnswlib::V0EdgeRecord> records;
    records.reserve(
        static_cast<size_t>(spec.directed_edge_count));
    std::vector<float> direction(dimension);
    uint64_t edge_index = 0U;
    for (size_t node = 0U; node < graph.nodeCount(); ++node) {
        const hnswlib::tableint source_id =
            static_cast<hnswlib::tableint>(node);
        const hnswlib::V0Layer0NeighborSpan neighbors =
            graph.neighbors(source_id);
        const float* source = graph.floatVector(source_id);
        for (size_t slot = 0U; slot < neighbors.size; ++slot) {
            const hnswlib::tableint target_id =
                neighbors.ids[slot];
            const float* target = graph.floatVector(target_id);
            const hnswlib::V0EdgeDirectionInfo direction_info =
                hnswlib::fillV0UnitEdgeDirection(
                    source, target, dimension, direction.data());

            hnswlib::V0EdgeRecord record;
            record.code.push_back(
                static_cast<uint8_t>(edge_index));
            if (direction_info.zero_length) {
                record.flags = static_cast<uint8_t>(
                    hnswlib::V0_EDGE_EXACT_ONLY |
                    hnswlib::V0_ZERO_LENGTH_EDGE);
            } else {
                const size_t centroid_offset =
                    static_cast<size_t>(edge_index) * dimension;
                for (uint32_t d = 0U; d < dimension; ++d) {
                    spec.codebook_centroids[
                        centroid_offset + d] = direction[d];
                }
                const hnswlib::V0EdgeNumericMetadata numeric =
                    hnswlib::computeV0EdgeNumericMetadata(
                        source,
                        target,
                        direction.data(),
                        dimension);
                record.edge_length = numeric.edge_length;
                record.direction_error =
                    numeric.direction_error;
                record.anchor_projection =
                    numeric.anchor_projection;
                record.numeric_padding =
                    numeric.numeric_padding;
            }
            records.push_back(record);
            ++edge_index;
        }
    }

    hnswlib::V0SidecarWriter writer(path, spec);
    for (size_t i = 0U; i < records.size(); ++i) {
        writer.writeEdgeRecord(records[i]);
    }
    writer.finalize();
}

void testShadowSearchPreservesBaseline() {
    const uint32_t dimension = 8U;
    const size_t node_count = 24U;
    const size_t k = 3U;
    const std::string sidecar_path =
        "v0_shadow_validation_test.v0meta";
    v0_test::FileCleanup cleanup(sidecar_path);

    std::mt19937 rng(911U);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> base(node_count * dimension);
    for (size_t i = 0U; i < base.size(); ++i) {
        base[i] = normal(rng);
    }

    hnswlib::L2Space space(dimension);
    hnswlib::HierarchicalNSW<float> index(
        &space, node_count, 4U, 40U, 911U);
    for (size_t node = 0U; node < node_count; ++node) {
        index.addPoint(
            base.data() + node * dimension,
            static_cast<hnswlib::labeltype>(node));
    }
    index.setEf(6U);
    writeExactDirectionSidecar(
        sidecar_path, index, dimension);
    index.loadEdgeQuantV0Metadata(sidecar_path);

    index.setEf(node_count + 1U);
    hnswlib::V0QueryMetrics pre_threshold_metrics;
    const std::priority_queue<
        std::pair<float, hnswlib::labeltype> > pre_threshold_baseline =
            index.searchKnn(base.data(), k);
    const std::priority_queue<
        std::pair<float, hnswlib::labeltype> > pre_threshold_shadow =
            index.searchKnnV0(
                base.data(), k, &pre_threshold_metrics);
    v0_test::require(
        sameQueue(
            pre_threshold_baseline,
            pre_threshold_shadow),
        "pre-threshold shadow search changed baseline results");
    v0_test::require(
        pre_threshold_metrics.bound_evaluated == 0U &&
            pre_threshold_metrics.bound_pruned == 0U &&
            pre_threshold_metrics.exact_fallback > 0U &&
            pre_threshold_metrics.exact_distance_saved == 0U,
        "V0 evaluated a bound before the result heap was full");

    index.setEf(6U);
    uint64_t total_bound_evaluated = 0U;
    uint64_t total_would_prune = 0U;
    uint64_t total_raw_prunable = 0U;
    uint64_t total_oracle_prunable = 0U;
    uint64_t total_fallback = 0U;
    RecordingShadowCollector collector;
    for (size_t query_id = 0U;
         query_id < node_count;
         ++query_id) {
        const float* query =
            base.data() + query_id * dimension;
        const std::priority_queue<
            std::pair<float, hnswlib::labeltype> > baseline =
                index.searchKnn(query, k);

        hnswlib::V0QueryMetrics metrics;
        const std::priority_queue<
            std::pair<float, hnswlib::labeltype> > shadow =
                index.searchKnnV0(
                    query,
                    k,
                    &metrics,
                    nullptr,
                    &collector,
                    static_cast<uint64_t>(query_id));

        v0_test::require(
            sameQueue(baseline, shadow),
            "shadow-only V0 search changed baseline results");
        v0_test::require(
            metrics.exact_distance_saved == 0U,
            "Milestone 9 reported saved exact distances");
        if (metrics.lower_bound_violation != 0U) {
            const hnswlib::V0ShadowRecord* violating = nullptr;
            for (size_t i = collector.records.size();
                 i > 0U;
                 --i) {
                if (collector.records[i - 1U].
                        lower_bound_violation) {
                    violating = &collector.records[i - 1U];
                    break;
                }
            }
            v0_test::require(
                violating != nullptr,
                "violation metric has no shadow record");
            const hnswlib::V0ShadowRecord& record = *violating;
            std::ostringstream message;
            message.precision(17);
            message
                << "shadow validation found an invalid lower bound: "
                << "lower=" << record.lower_bound
                << " exact="
                << record.shadow_exact_squared_distance
                << " gap="
                << record.lower_bound -
                    record.shadow_exact_squared_distance
                << " current="
                << record.current_squared_distance
                << " approximate="
                << record.approximate_squared_distance
                << " error_radius="
                << record.error_radius;
            throw std::runtime_error(message.str());
        }
        v0_test::require(
            metrics.false_prune == 0U,
            "shadow validation found a false prune");
        total_bound_evaluated += metrics.bound_evaluated;
        total_would_prune += metrics.bound_pruned;
        total_raw_prunable += metrics.raw_prunable;
        total_oracle_prunable += metrics.oracle_prunable;
        total_fallback +=
            metrics.exact_fallback +
            metrics.exact_only_fallback;
    }

    v0_test::require(
        total_bound_evaluated > 0U,
        "shadow test evaluated no V0 bounds");
    v0_test::require(
        total_would_prune > 0U,
        "shadow test found no would-prune decisions");
    v0_test::require(
        total_fallback > 0U,
        "shadow test did not exercise pre-threshold fallback");
    v0_test::require(
        collector.records.size() ==
            static_cast<size_t>(total_bound_evaluated),
        "shadow record count does not match attempted valid bounds");

    bool observed_would_prune = false;
    uint64_t recorded_raw_prunable = 0U;
    uint64_t recorded_oracle_prunable = 0U;
    for (size_t i = 0U; i < collector.records.size(); ++i) {
        const hnswlib::V0ShadowRecord& record =
            collector.records[i];
        v0_test::require(
            record.lower_bound_valid,
            "exact-direction sidecar produced an invalid bound");
        v0_test::require(
            record.query_id < node_count,
            "shadow record did not preserve its query id");
        v0_test::require(
            record.graph_layer == 0U,
            "V0 shadow record has an unexpected graph layer");
        v0_test::require(
            std::isfinite(record.edge_length) &&
                record.edge_length > 0.0 &&
                std::isfinite(record.direction_error) &&
                record.direction_error >= 0.0,
            "shadow record contains invalid edge metadata");
        v0_test::require(
            record.anchor_projection_lower <=
                record.anchor_projection,
            "anchor projection was not rounded downward");
        v0_test::require(
            record.length_squared_lower <=
                record.edge_length * record.edge_length,
            "edge-length square was not rounded downward");
        const long double component_sum =
            static_cast<long double>(record.direction_error_radius) +
            static_cast<long double>(record.stored_numeric_padding) +
            static_cast<long double>(record.operational_l2_padding);
        const long double closure =
            static_cast<long double>(record.error_radius) -
            component_sum;
        const long double closure_tolerance =
            1.0e-15L *
            (1.0L + std::fabs(
                static_cast<long double>(record.error_radius)));
        v0_test::require(
            closure >= -closure_tolerance,
            "radius components exceed the total radius");
        v0_test::require(
            std::fabs(
                closure - static_cast<long double>(
                    record.rounding_closure_padding)) <=
                closure_tolerance,
            "rounding closure padding does not close the radius");
        const double expected_lower =
            hnswlib::edge_quant_v0_query_detail::addDown(
                record.approximate_squared_distance,
                -record.error_radius);
        v0_test::require(
            record.lower_bound ==
                (expected_lower > 0.0 ? expected_lower : 0.0),
            "shadow record cannot reproduce the current lower bound");
        v0_test::require(
            !record.lower_bound_violation &&
                record.lower_bound <=
                    record.shadow_exact_squared_distance,
            "shadow record contains a lower-bound violation");
        v0_test::require(
            !record.false_prune,
            "shadow record contains a false prune");
        if (record.would_prune) {
            observed_would_prune = true;
            v0_test::require(
                record.shadow_exact_squared_distance >
                    record.threshold,
                "would-prune shadow record is unsafe");
        }
        if (record.approximate_squared_distance >
            record.threshold) {
            ++recorded_raw_prunable;
        }
        if (record.oracle_would_prune) {
            ++recorded_oracle_prunable;
            v0_test::require(
                record.shadow_exact_squared_distance >
                    record.threshold,
                "oracle-would-prune record does not exceed threshold");
        }
    }
    v0_test::require(
        observed_would_prune,
        "collector did not retain a would-prune record");
    v0_test::require(
        total_raw_prunable == recorded_raw_prunable,
        "raw-prunable full counter does not match shadow records");
    v0_test::require(
        total_oracle_prunable == recorded_oracle_prunable,
        "oracle-prunable full counter does not match shadow records");
}

}  // namespace

int main() {
#if !defined(HNSWLIB_ENABLE_EDGE_QUANT_V0) || \
    !defined(HNSWLIB_ENABLE_V0_SHADOW_VALIDATION)
    std::cerr
        << "V0 and shadow validation are required"
        << std::endl;
    return 2;
#else
    testShadowSearchPreservesBaseline();
    std::cout << "v0_shadow_validation_test_ok" << std::endl;
    return 0;
#endif
}
