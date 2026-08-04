#pragma once

#include <cstddef>
#include <cstdint>
#include <queue>
#include <string>
#include <vector>

#include "hnswlib/edge_quant_v0_numeric.h"
#include "v0_sidecar_test_utils.h"

namespace v0_ratio_test {

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

inline void writeExactDirectionSidecar(
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
        "{\"purpose\":\"phase4_ratio_graph_test\"}";
    spec.codebook_centroids.assign(
        static_cast<size_t>(spec.pq_ksub) * dimension, 0.0f);
    spec.node_offsets.push_back(0U);
    for (size_t node = 0U; node < graph.nodeCount(); ++node) {
        spec.directed_edge_count += static_cast<uint64_t>(
            graph.neighbors(
                static_cast<hnswlib::tableint>(node)).size);
        spec.node_offsets.push_back(spec.directed_edge_count);
    }
    v0_test::require(
        spec.directed_edge_count <= spec.pq_ksub,
        "ratio graph test has too many directed edges");
    spec.base_index_sha256 = index.getV0SerializedIndexFingerprint();
    spec.adjacency_sha256 = graph.adjacencyFingerprint();

    std::vector<hnswlib::V0EdgeRecord> records;
    std::vector<float> direction(dimension);
    uint64_t edge_index = 0U;
    for (size_t node = 0U; node < graph.nodeCount(); ++node) {
        const hnswlib::tableint source_id =
            static_cast<hnswlib::tableint>(node);
        const hnswlib::V0Layer0NeighborSpan neighbors =
            graph.neighbors(source_id);
        const float* source = graph.floatVector(source_id);
        for (size_t slot = 0U; slot < neighbors.size; ++slot) {
            const float* target =
                graph.floatVector(neighbors.ids[slot]);
            const hnswlib::V0EdgeDirectionInfo direction_info =
                hnswlib::fillV0UnitEdgeDirection(
                    source, target, dimension, direction.data());
            hnswlib::V0EdgeRecord record;
            record.code.push_back(static_cast<uint8_t>(edge_index));
            if (direction_info.zero_length) {
                record.flags = static_cast<uint8_t>(
                    hnswlib::V0_EDGE_EXACT_ONLY |
                    hnswlib::V0_ZERO_LENGTH_EDGE);
            } else {
                const size_t centroid_offset =
                    static_cast<size_t>(edge_index) * dimension;
                for (uint32_t d = 0U; d < dimension; ++d) {
                    spec.codebook_centroids[centroid_offset + d] =
                        direction[d];
                }
                const hnswlib::V0EdgeNumericMetadata numeric =
                    hnswlib::computeV0EdgeNumericMetadata(
                        source, target, direction.data(), dimension);
                record.edge_length = numeric.edge_length;
                record.direction_error = numeric.direction_error;
                record.anchor_projection = numeric.anchor_projection;
                record.numeric_padding = numeric.numeric_padding;
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

}  // namespace v0_ratio_test
