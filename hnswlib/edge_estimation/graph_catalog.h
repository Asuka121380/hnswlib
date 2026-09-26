#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../edge_quant_v0_graph_access.h"
#include "types.h"

namespace hnswlib {
namespace edge_estimation {

class EdgeCatalog {
 public:
    EdgeCatalog() : node_count_(0) {}

    static EdgeCatalog fromLayer0Graph(const V0Layer0GraphView& graph) {
        EdgeCatalog catalog;
        catalog.node_count_ = graph.nodeCount();
        catalog.source_offsets_.reserve(catalog.node_count_ + 1U);
        catalog.source_offsets_.push_back(0U);
        for (size_t source = 0; source < catalog.node_count_; ++source) {
            const V0Layer0NeighborSpan span =
                graph.neighbors(static_cast<tableint>(source));
            if (span.size > std::numeric_limits<uint64_t>::max() -
                    catalog.targets_.size()) {
                throw std::overflow_error("edge catalog size overflow");
            }
            for (size_t slot = 0; slot < span.size; ++slot) {
                catalog.targets_.push_back(static_cast<uint32_t>(span[slot]));
            }
            catalog.source_offsets_.push_back(
                static_cast<uint64_t>(catalog.targets_.size()));
        }
        catalog.adjacency_digest_ = graph.adjacencyFingerprint();
        EdgeQuantV0Sha256 identity;
        const char magic[] = "UQCAT001";
        identity.update(magic, 8U);
        edgeQuantV0Sha256UpdateUint64LittleEndian(
            identity, static_cast<uint64_t>(catalog.node_count_));
        edgeQuantV0Sha256UpdateUint64LittleEndian(
            identity, static_cast<uint64_t>(catalog.targets_.size()));
        identity.update(catalog.adjacency_digest_.data(),
                        catalog.adjacency_digest_.size());
        for (size_t i = 0; i < catalog.source_offsets_.size(); ++i)
            edgeQuantV0Sha256UpdateUint64LittleEndian(
                identity, catalog.source_offsets_[i]);
        for (size_t i = 0; i < catalog.targets_.size(); ++i)
            edgeQuantV0Sha256UpdateUint32LittleEndian(
                identity, catalog.targets_[i]);
        catalog.identity_digest_ = identity.final();
        catalog.validate();
        return catalog;
    }

    void validate() const {
        if (source_offsets_.size() != node_count_ + 1U ||
            source_offsets_.empty() || source_offsets_[0] != 0U ||
            source_offsets_.back() != targets_.size()) {
            throw std::runtime_error("invalid edge catalog offsets");
        }
        for (size_t i = 1; i < source_offsets_.size(); ++i) {
            if (source_offsets_[i] < source_offsets_[i - 1U]) {
                throw std::runtime_error("non-monotonic edge catalog offsets");
            }
        }
        for (size_t i = 0; i < targets_.size(); ++i) {
            if (targets_[i] >= node_count_) {
                throw std::runtime_error("edge catalog target out of range");
            }
        }
    }

    EdgeId edgeId(uint32_t source_id, uint32_t neighbor_slot) const {
        if (source_id >= node_count_) {
            throw std::out_of_range("edge catalog source out of range");
        }
        const uint64_t first = source_offsets_[source_id];
        const uint64_t last = source_offsets_[source_id + 1U];
        if (neighbor_slot >= last - first) {
            throw std::out_of_range("edge catalog slot out of range");
        }
        return first + neighbor_slot;
    }

    uint32_t target(EdgeId edge_id) const {
        if (edge_id >= targets_.size()) {
            throw std::out_of_range("edge catalog edge id out of range");
        }
        return targets_[static_cast<size_t>(edge_id)];
    }

    size_t nodeCount() const { return node_count_; }
    size_t edgeCount() const { return targets_.size(); }
    const std::vector<uint64_t>& sourceOffsets() const { return source_offsets_; }
    const std::vector<uint32_t>& targets() const { return targets_; }
    const std::array<uint8_t, 32>& adjacencyDigest() const {
        return adjacency_digest_;
    }
    const std::array<uint8_t, 32>& identityDigest() const {
        return identity_digest_;
    }

    void writeBinaryFiles(
        const std::string& source_offsets_path,
        const std::string& targets_path) const {
        validate();
        std::ofstream offsets(source_offsets_path.c_str(),
                              std::ios::binary | std::ios::trunc);
        if (!offsets) throw std::runtime_error("cannot create catalog offsets");
        for (size_t i = 0; i < source_offsets_.size(); ++i)
            writeU64(offsets, source_offsets_[i]);
        if (!offsets) throw std::runtime_error("cannot write catalog offsets");
        std::ofstream targets(targets_path.c_str(),
                              std::ios::binary | std::ios::trunc);
        if (!targets) throw std::runtime_error("cannot create catalog targets");
        for (size_t i = 0; i < targets_.size(); ++i)
            writeU32(targets, targets_[i]);
        if (!targets) throw std::runtime_error("cannot write catalog targets");
    }

 private:
    static void writeU32(std::ofstream& output, uint32_t value) {
        uint8_t bytes[4];
        for (size_t i = 0; i < 4U; ++i)
            bytes[i] = static_cast<uint8_t>((value >> (8U * i)) & 0xffU);
        output.write(reinterpret_cast<const char*>(bytes), 4);
    }
    static void writeU64(std::ofstream& output, uint64_t value) {
        uint8_t bytes[8];
        for (size_t i = 0; i < 8U; ++i)
            bytes[i] = static_cast<uint8_t>((value >> (8U * i)) & 0xffU);
        output.write(reinterpret_cast<const char*>(bytes), 8);
    }

    size_t node_count_;
    std::vector<uint64_t> source_offsets_;
    std::vector<uint32_t> targets_;
    std::array<uint8_t, 32> adjacency_digest_;
    std::array<uint8_t, 32> identity_digest_;
};

}  // namespace edge_estimation
}  // namespace hnswlib
