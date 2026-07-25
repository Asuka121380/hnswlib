#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "edge_quant_v0_checksum.h"
#include "hnswlib.h"

namespace hnswlib {

// These match hnswalg.h. Repeating an identical typedef keeps this header
// independently includable while preserving hnswlib's public node-id type.
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

static_assert(sizeof(tableint) == 4U,
              "V0 adjacency fingerprints require 32-bit tableint");

struct V0Layer0NeighborSpan {
    const tableint* ids;
    size_t size;

    const tableint& operator[](size_t slot) const {
        if (slot >= size) {
            throw std::out_of_range("V0 layer-0 neighbor slot is out of range");
        }
        return ids[slot];
    }
};

// Non-owning, read-only snapshot of the current layer-0 graph layout.
// The owner index must outlive this view and must not be mutated while the
// view is used.
class V0Layer0GraphView {
 public:
    V0Layer0GraphView(
        const char* level0_memory,
        size_t node_count,
        size_t bytes_per_node,
        size_t data_offset,
        size_t data_size,
        size_t label_offset,
        size_t max_layer0_degree)
        : level0_memory_(level0_memory),
          node_count_(node_count),
          bytes_per_node_(bytes_per_node),
          data_offset_(data_offset),
          data_size_(data_size),
          label_offset_(label_offset),
          max_layer0_degree_(max_layer0_degree) {
        if (node_count_ != 0U && level0_memory_ == NULL) {
            throw std::invalid_argument(
                "V0 graph view received null storage for a non-empty index");
        }
        if (node_count_ != 0U &&
            (bytes_per_node_ < sizeof(linklistsizeint) ||
             data_offset_ > bytes_per_node_ ||
             data_size_ > bytes_per_node_ - data_offset_ ||
             label_offset_ > bytes_per_node_ ||
             sizeof(labeltype) > bytes_per_node_ - label_offset_)) {
            throw std::invalid_argument("V0 graph view received invalid layout");
        }
    }

    size_t nodeCount() const {
        return node_count_;
    }

    size_t dataSize() const {
        return data_size_;
    }

    size_t maxLayer0Degree() const {
        return max_layer0_degree_;
    }

    V0Layer0NeighborSpan neighbors(tableint source_id) const {
        const char* node = nodeBase(source_id);
        unsigned short count = 0;
        std::memcpy(&count, node, sizeof(count));
        if (static_cast<size_t>(count) > max_layer0_degree_) {
            throw std::runtime_error(
                "V0 layer-0 neighbor count exceeds index capacity");
        }

        V0Layer0NeighborSpan span;
        span.ids = reinterpret_cast<const tableint*>(
            node + sizeof(linklistsizeint));
        span.size = static_cast<size_t>(count);
        return span;
    }

    const void* vectorData(tableint node_id) const {
        return nodeBase(node_id) + data_offset_;
    }

    const float* floatVector(tableint node_id) const {
        return static_cast<const float*>(vectorData(node_id));
    }

    labeltype externalLabel(tableint node_id) const {
        labeltype label;
        std::memcpy(&label, nodeBase(node_id) + label_offset_, sizeof(label));
        return label;
    }

    template<typename Callback>
    void forEachEdge(Callback callback) const {
        for (size_t source = 0; source < node_count_; ++source) {
            const tableint source_id = static_cast<tableint>(source);
            const V0Layer0NeighborSpan span = neighbors(source_id);
            for (size_t slot = 0; slot < span.size; ++slot) {
                const tableint target_id = span.ids[slot];
                validateTarget(target_id);
                callback(source_id, target_id, slot);
            }
        }
    }

    // Canonical stream:
    // uint64 node_count;
    // repeated { uint32 source_id; uint32 degree; uint32 target_id[degree]; }
    // All integers are hashed in explicit little-endian byte order.
    std::array<uint8_t, 32> adjacencyFingerprint() const {
        EdgeQuantV0Sha256 sha;
        edgeQuantV0Sha256UpdateUint64LittleEndian(
            sha, static_cast<uint64_t>(node_count_));

        for (size_t source = 0; source < node_count_; ++source) {
            const tableint source_id = static_cast<tableint>(source);
            const V0Layer0NeighborSpan span = neighbors(source_id);
            edgeQuantV0Sha256UpdateUint32LittleEndian(
                sha, static_cast<uint32_t>(source_id));
            edgeQuantV0Sha256UpdateUint32LittleEndian(
                sha, static_cast<uint32_t>(span.size));
            for (size_t slot = 0; slot < span.size; ++slot) {
                const tableint target_id = span.ids[slot];
                validateTarget(target_id);
                edgeQuantV0Sha256UpdateUint32LittleEndian(
                    sha, static_cast<uint32_t>(target_id));
            }
        }
        return sha.final();
    }

    std::string adjacencyFingerprintHex() const {
        return edgeQuantV0Sha256Hex(adjacencyFingerprint());
    }

 private:
    const char* nodeBase(tableint node_id) const {
        if (static_cast<size_t>(node_id) >= node_count_) {
            throw std::out_of_range("V0 graph node id is out of range");
        }
        return level0_memory_ +
            static_cast<size_t>(node_id) * bytes_per_node_;
    }

    void validateTarget(tableint target_id) const {
        if (static_cast<size_t>(target_id) >= node_count_) {
            throw std::runtime_error(
                "V0 layer-0 adjacency contains an invalid target id");
        }
    }

    const char* level0_memory_;
    size_t node_count_;
    size_t bytes_per_node_;
    size_t data_offset_;
    size_t data_size_;
    size_t label_offset_;
    size_t max_layer0_degree_;
};

}  // namespace hnswlib
