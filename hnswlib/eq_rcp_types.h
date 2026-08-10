#pragma once

#include <cstddef>
#include <cstdint>

namespace hnswlib {

static const uint32_t kEqRcpFormatVersion = 1U;
static const uint32_t kEqRcpCodeBudgetBytes = 32U;
static const uint32_t kEqRcpDefaultCoarseCodeBytes = 24U;
static const uint32_t kEqRcpDefaultResidualCodeBytes = 8U;
static const uint32_t kEqRcpScalarMetadataBytes = 12U;
static const uint32_t kEqRcpLogicalRecordBytes = 44U;
static const uint32_t kEqRcpAlignedRecordBytes = 48U;

enum class EqRcpRuntimeMode : uint8_t {
    Disabled = 0U,
    Shadow = 1U,
    DeterministicOnly = 2U,
    ProbabilisticPruning = 3U
};

enum class EqRcpStatus : uint8_t {
    Valid = 0U,
    Disabled = 1U,
    MetadataMissing = 2U,
    MetadataStale = 3U,
    InvalidLayout = 4U,
    InvalidCode = 5U,
    ExactFallback = 6U
};

enum class EqRcpMetricType : uint8_t {
    SquaredL2Float32 = 1U
};

struct EqRcpLayoutDescriptor {
    uint32_t format_version;
    uint32_t dimension;
    uint32_t coarse_code_bytes;
    uint32_t residual_code_bytes;
    uint32_t scalar_metadata_bytes;
    uint32_t logical_record_bytes;
    uint32_t edge_record_stride;
    EqRcpMetricType metric_type;

    EqRcpLayoutDescriptor()
        : format_version(kEqRcpFormatVersion),
          dimension(0U),
          coarse_code_bytes(kEqRcpDefaultCoarseCodeBytes),
          residual_code_bytes(kEqRcpDefaultResidualCodeBytes),
          scalar_metadata_bytes(kEqRcpScalarMetadataBytes),
          logical_record_bytes(kEqRcpLogicalRecordBytes),
          edge_record_stride(kEqRcpAlignedRecordBytes),
          metric_type(EqRcpMetricType::SquaredL2Float32) {}

    uint32_t codeBytes() const {
        return coarse_code_bytes + residual_code_bytes;
    }

    bool valid() const {
        if (format_version != kEqRcpFormatVersion || dimension == 0U) {
            return false;
        }
        if (codeBytes() != kEqRcpCodeBudgetBytes ||
            scalar_metadata_bytes != kEqRcpScalarMetadataBytes) {
            return false;
        }
        if (logical_record_bytes != codeBytes() + scalar_metadata_bytes ||
            edge_record_stride < logical_record_bytes) {
            return false;
        }
        return edge_record_stride % 8U == 0U &&
            metric_type == EqRcpMetricType::SquaredL2Float32;
    }
};

}  // namespace hnswlib
