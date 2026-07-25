#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "edge_quant_v0_graph_access.h"
#include "edge_quant_v0_io.h"

namespace hnswlib {

// Immutable query-time binding for one validated V0 sidecar. The owned bytes
// keep every view and edge record alive for the lifetime of this object.
class EdgeQuantV0Metadata {
 public:
    const V0SidecarHeader& header() const {
        return view_.header();
    }

    const V0SidecarView& view() const {
        return view_;
    }

    size_t storageBytes() const {
        return view_.fileSize();
    }

    uint64_t edgeIndex(tableint source_id, size_t layer0_slot) const {
        if (static_cast<uint64_t>(source_id) >= header().node_count) {
            throw std::out_of_range(
                "V0 query metadata source node is out of range");
        }
        const uint64_t first = view_.nodeOffset(source_id);
        const uint64_t last = view_.nodeOffset(
            static_cast<size_t>(source_id) + 1U);
        const uint64_t degree = last - first;
        if (static_cast<uint64_t>(layer0_slot) >= degree) {
            throw std::out_of_range(
                "V0 query metadata layer-0 slot is out of range");
        }
        if (static_cast<uint64_t>(layer0_slot) >
            std::numeric_limits<uint64_t>::max() - first) {
            throw std::overflow_error(
                "V0 query metadata edge index overflow");
        }
        return first + static_cast<uint64_t>(layer0_slot);
    }

    V0EdgeRecordView edgeRecord(
        tableint source_id,
        size_t layer0_slot) const {
        return view_.edgeRecord(static_cast<size_t>(
            edgeIndex(source_id, layer0_slot)));
    }

 private:
    explicit EdgeQuantV0Metadata(
        std::shared_ptr<const V0OwnedSidecar> storage)
        : storage_(std::move(storage)),
          view_(requireStorage(storage_).view()) {}

    static const V0OwnedSidecar& requireStorage(
        const std::shared_ptr<const V0OwnedSidecar>& storage) {
        if (!storage) {
            throw std::invalid_argument(
                "V0 query metadata requires owned sidecar storage");
        }
        return *storage;
    }

    friend std::shared_ptr<const EdgeQuantV0Metadata>
    loadEdgeQuantV0Metadata(
        const std::string& sidecar_path,
        const V0Layer0GraphView& graph,
        uint32_t dimension,
        const V0Sha256Digest& current_index_sha256);

    std::shared_ptr<const V0OwnedSidecar> storage_;
    V0SidecarView view_;
};

inline std::shared_ptr<const EdgeQuantV0Metadata>
loadEdgeQuantV0Metadata(
    const std::string& sidecar_path,
    const V0Layer0GraphView& graph,
    uint32_t dimension,
    const V0Sha256Digest& current_index_sha256) {
    if (sidecar_path.empty()) {
        throw std::invalid_argument(
            "V0 query metadata sidecar path is empty");
    }

    V0OwnedSidecar owned = loadV0Sidecar(sidecar_path);
    const V0SidecarView sidecar = owned.view();

    uint64_t directed_edge_count = 0U;
    for (size_t node = 0U; node < graph.nodeCount(); ++node) {
        const uint64_t degree = static_cast<uint64_t>(
            graph.neighbors(static_cast<tableint>(node)).size);
        if (degree >
            std::numeric_limits<uint64_t>::max() -
                directed_edge_count) {
            throw std::overflow_error(
                "V0 query metadata layer-0 edge count overflow");
        }
        directed_edge_count += degree;
    }

    V0IndexCompatibility expected;
    expected.dimension = dimension;
    expected.node_count = static_cast<uint64_t>(graph.nodeCount());
    expected.directed_edge_count = directed_edge_count;
    expected.base_index_sha256 = current_index_sha256;
    expected.adjacency_sha256 = graph.adjacencyFingerprint();
    validateV0SidecarCompatibility(sidecar.header(), expected);
    validateV0SidecarGraphLayout(sidecar, graph);

    std::shared_ptr<const V0OwnedSidecar> storage(
        new V0OwnedSidecar(std::move(owned)));
    return std::shared_ptr<const EdgeQuantV0Metadata>(
        new EdgeQuantV0Metadata(std::move(storage)));
}

}  // namespace hnswlib
