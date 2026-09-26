#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace hnswlib {
namespace edge_estimation {

using EdgeId = uint64_t;

static const EdgeId kInvalidEdgeId = std::numeric_limits<EdgeId>::max();
static const uint32_t kInvalidNodeId = std::numeric_limits<uint32_t>::max();

enum class EstimateStatus : uint8_t {
    Valid = 0,
    ZeroLength = 1,
    UnsupportedRecord = 2,
    InvalidCode = 3,
    NonFinite = 4,
    NumericOverflow = 5,
    LegacyExactOnly = 6
};

struct DotEstimate {
    double value;
    EstimateStatus status;

    DotEstimate()
        : value(0.0), status(EstimateStatus::UnsupportedRecord) {}
    DotEstimate(double value_in, EstimateStatus status_in)
        : value(value_in), status(status_in) {}
    bool valid() const { return status == EstimateStatus::Valid; }
};

struct EdgeScore {
    double squared_distance;
    EstimateStatus status;

    EdgeScore()
        : squared_distance(0.0),
          status(EstimateStatus::UnsupportedRecord) {}
    EdgeScore(double value, EstimateStatus status_in)
        : squared_distance(value), status(status_in) {}
    bool valid() const { return status == EstimateStatus::Valid; }
};

struct EdgeRef {
    EdgeId edge_id;
    uint32_t source_id;
    uint32_t target_id;
    uint32_t neighbor_slot;
    double edge_length;

    EdgeRef()
        : edge_id(kInvalidEdgeId),
          source_id(kInvalidNodeId),
          target_id(kInvalidNodeId),
          neighbor_slot(kInvalidNodeId),
          edge_length(0.0) {}
};

struct MemoryReport {
    uint64_t shared_bytes;
    uint64_t backend_bytes;
    uint64_t mapping_bytes;
    uint64_t scratch_bytes;

    MemoryReport()
        : shared_bytes(0), backend_bytes(0), mapping_bytes(0),
          scratch_bytes(0) {}
};

}  // namespace edge_estimation
}  // namespace hnswlib
