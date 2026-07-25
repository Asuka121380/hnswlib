#include <cstddef>
#include <cstdint>
#include <iostream>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

bool metricsAreZero(const hnswlib::V0QueryMetrics& metrics) {
    return metrics.bound_evaluated == 0U &&
        metrics.bound_pruned == 0U &&
        metrics.exact_fallback == 0U &&
        metrics.exact_only_fallback == 0U &&
        metrics.exact_distance_saved == 0U &&
        metrics.lower_bound_violation == 0U &&
        metrics.false_prune == 0U;
}

class EvenLabelFilter : public hnswlib::BaseFilterFunctor {
 public:
    bool operator()(hnswlib::labeltype id) {
        return id % 2U == 0U;
    }
};

void writeMatchingSidecar(
    const std::string& sidecar_path,
    const hnswlib::HierarchicalNSW<float>& index,
    uint32_t dimension) {
    const hnswlib::V0Layer0GraphView graph =
        index.getV0Layer0GraphView();

    hnswlib::V0SidecarWriteSpec spec;
    spec.dimension = dimension;
    spec.pq_m = 2U;
    spec.pq_nbits = 1U;
    spec.pq_ksub = 2U;
    spec.pq_dsub = dimension / spec.pq_m;
    spec.node_count = static_cast<uint64_t>(graph.nodeCount());
    spec.training_metadata_json =
        "{\"purpose\":\"milestone6_query_metadata_test\"}";
    spec.codebook_centroids.resize(
        static_cast<size_t>(
            spec.pq_m * spec.pq_ksub * spec.pq_dsub));
    for (size_t i = 0U; i < spec.codebook_centroids.size(); ++i) {
        spec.codebook_centroids[i] =
            static_cast<float>(i) * 0.03125f;
    }
    spec.node_offsets.push_back(0U);
    for (size_t node = 0U; node < graph.nodeCount(); ++node) {
        spec.directed_edge_count += static_cast<uint64_t>(
            graph.neighbors(static_cast<hnswlib::tableint>(node)).size);
        spec.node_offsets.push_back(spec.directed_edge_count);
    }
    spec.base_index_sha256 =
        index.getV0SerializedIndexFingerprint();
    spec.adjacency_sha256 = graph.adjacencyFingerprint();

    hnswlib::V0SidecarWriter writer(sidecar_path, spec);
    uint64_t edge_index = 0U;
    for (size_t node = 0U; node < graph.nodeCount(); ++node) {
        const hnswlib::V0Layer0NeighborSpan neighbors =
            graph.neighbors(static_cast<hnswlib::tableint>(node));
        for (size_t slot = 0U; slot < neighbors.size; ++slot) {
            const uint8_t code[2] = {
                static_cast<uint8_t>(edge_index % 2U),
                static_cast<uint8_t>((edge_index / 2U) % 2U)
            };
            writer.writeEdgeRecord(
                code,
                2U,
                0U,
                1.0 + static_cast<double>(edge_index),
                0.01 * static_cast<double>(edge_index % 7U),
                -0.5 * static_cast<double>(edge_index),
                0.0);
            ++edge_index;
        }
    }
    writer.finalize();
}

void testQueryMetadataLifecycle() {
    const uint32_t dimension = 8U;
    const size_t node_count = 64U;
    const size_t k = 10U;
    const std::string index_path = "v0_query_metadata_test.bin";
    const std::string sidecar_path =
        "v0_query_metadata_test.v0meta";
    v0_test::FileCleanup index_cleanup(index_path);
    v0_test::FileCleanup sidecar_cleanup(sidecar_path);

    std::mt19937 rng(73U);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> base(node_count * dimension);
    for (size_t i = 0U; i < base.size(); ++i) {
        base[i] = normal(rng);
    }

    hnswlib::L2Space space(dimension);
    hnswlib::HierarchicalNSW<float> index(
        &space, node_count + 1U, 12U, 80U, 73U);
    for (size_t node = 0U; node < node_count; ++node) {
        index.addPoint(
            base.data() + node * dimension,
            static_cast<hnswlib::labeltype>(node));
    }
    index.setEf(50U);
    index.saveIndex(index_path);
    v0_test::require(
        index.getV0SerializedIndexFingerprint() ==
            hnswlib::computeV0FileSha256(index_path),
        "in-memory V0 index fingerprint does not match saveIndex");
    writeMatchingSidecar(sidecar_path, index, dimension);

    hnswlib::V0QueryMetrics metrics;
    metrics.bound_evaluated = 1U;
    v0_test::requireThrows(
        [&index, &base, k, &metrics]() {
            (void)index.searchKnnV0(base.data(), k, &metrics);
        },
        "V0 search accepted missing metadata");
    v0_test::require(
        metricsAreZero(metrics),
        "missing-metadata search did not reset metrics");

    index.loadEdgeQuantV0Metadata(sidecar_path);
    v0_test::require(
        index.hasEdgeQuantV0Metadata(),
        "valid V0 metadata was not attached");
    v0_test::require(
        !index.isEdgeQuantV0MetadataStale(),
        "newly attached V0 metadata is stale");
    v0_test::require(
        index.getEdgeQuantV0Metadata().storageBytes() > 0U,
        "attached V0 metadata has no storage");

    const hnswlib::V0Layer0GraphView graph =
        index.getV0Layer0GraphView();
    size_t source = 0U;
    while (source < graph.nodeCount() &&
           graph.neighbors(
               static_cast<hnswlib::tableint>(source)).size == 0U) {
        ++source;
    }
    v0_test::require(
        source < graph.nodeCount(),
        "test graph has no layer-0 edges");
    const uint64_t expected_edge_index =
        index.getEdgeQuantV0Metadata().view().nodeOffset(source);
    v0_test::require(
        index.getEdgeQuantV0Metadata().edgeIndex(
            static_cast<hnswlib::tableint>(source), 0U) ==
            expected_edge_index,
        "source/slot did not map to the expected edge index");
    const hnswlib::V0EdgeRecordView record =
        index.getEdgeQuantV0Record(
            static_cast<hnswlib::tableint>(source), 0U);
    v0_test::require(
        record.code(0U) ==
            static_cast<uint8_t>(expected_edge_index % 2U) &&
        record.code(1U) ==
            static_cast<uint8_t>((expected_edge_index / 2U) % 2U),
        "source/slot returned the wrong PQ code");
    v0_test::requireThrows(
        [&index, &graph, source]() {
            (void)index.getEdgeQuantV0Record(
                static_cast<hnswlib::tableint>(source),
                graph.neighbors(
                    static_cast<hnswlib::tableint>(source)).size);
        },
        "V0 metadata accepted an out-of-range edge slot");

    EvenLabelFilter even_label_filter;
    for (size_t query = 0U; query < 12U; ++query) {
        const float* vector = base.data() + query * dimension;
        const std::priority_queue<
            std::pair<float, hnswlib::labeltype> > baseline =
                index.searchKnn(vector, k);
        const std::priority_queue<
            std::pair<float, hnswlib::labeltype> > v0 =
                index.searchKnnV0(vector, k, &metrics);
        v0_test::require(
            sameQueue(baseline, v0),
            "loaded V0 exact search changed baseline results");
        v0_test::require(
            metricsAreZero(metrics),
            "Milestone 6 V0 search produced pruning metrics");

        const std::priority_queue<
            std::pair<float, hnswlib::labeltype> > filtered_baseline =
                index.searchKnn(vector, k, &even_label_filter);
        const std::priority_queue<
            std::pair<float, hnswlib::labeltype> > filtered_v0 =
                index.searchKnnV0(
                    vector, k, &metrics, &even_label_filter);
        v0_test::require(
            sameQueue(filtered_baseline, filtered_v0),
            "loaded filtered V0 search changed baseline results");
        v0_test::require(
            metricsAreZero(metrics),
            "filtered Milestone 6 search produced pruning metrics");
    }

    const float updated[8] = {
        2.0f, 3.0f, 5.0f, 7.0f,
        11.0f, 13.0f, 17.0f, 19.0f
    };
    index.updatePoint(updated, 0U, 0.0f);
    v0_test::require(
        index.hasEdgeQuantV0Metadata() &&
        index.isEdgeQuantV0MetadataStale(),
        "index mutation did not mark V0 metadata stale");
    v0_test::requireThrows(
        [&index, &base, k, &metrics]() {
            (void)index.searchKnnV0(base.data(), k, &metrics);
        },
        "V0 search accepted stale metadata");
    v0_test::require(
        metricsAreZero(metrics),
        "stale-metadata search did not reset metrics");
    v0_test::require(
        !index.searchKnn(base.data(), k).empty(),
        "baseline search was affected by stale V0 metadata");

    index.clearEdgeQuantV0Metadata();
    v0_test::require(
        !index.hasEdgeQuantV0Metadata() &&
        !index.isEdgeQuantV0MetadataStale(),
        "clearing V0 metadata did not reset lifecycle state");
    v0_test::requireThrows(
        [&index, &sidecar_path]() {
            index.loadEdgeQuantV0Metadata(sidecar_path);
        },
        "V0 metadata accepted a modified in-memory index");
    v0_test::require(
        !index.hasEdgeQuantV0Metadata(),
        "failed stale-index reload changed attachment state");

    hnswlib::HierarchicalNSW<float> loaded_index(
        &space, index_path);
    loaded_index.setEf(50U);
    loaded_index.loadEdgeQuantV0Metadata(sidecar_path);
    v0_test::require(
        loaded_index.hasEdgeQuantV0Metadata() &&
        !loaded_index.isEdgeQuantV0MetadataStale(),
        "saved-index metadata load did not become usable");
    loaded_index.addPoint(updated, node_count);
    v0_test::require(
        loaded_index.isEdgeQuantV0MetadataStale(),
        "addPoint did not mark loaded V0 metadata stale");

    loaded_index.loadIndex(index_path, &space);
    v0_test::require(
        !loaded_index.hasEdgeQuantV0Metadata() &&
        !loaded_index.isEdgeQuantV0MetadataStale(),
        "loadIndex did not clear attached V0 metadata");
    loaded_index.loadEdgeQuantV0Metadata(sidecar_path);
    v0_test::require(
        sameQueue(
            loaded_index.searchKnn(base.data(), k),
            loaded_index.searchKnnV0(base.data(), k, &metrics)),
        "reloaded-index V0 exact search changed baseline results");
}

}  // namespace

int main() {
#ifndef HNSWLIB_ENABLE_EDGE_QUANT_V0
    std::cerr << "HNSWLIB_ENABLE_EDGE_QUANT_V0 is required" << std::endl;
    return 2;
#else
    testQueryMetadataLifecycle();
    std::cout << "v0_query_metadata_test_ok" << std::endl;
    return 0;
#endif
}
