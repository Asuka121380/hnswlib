#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hnswlib {

static const uint32_t EDGE_QUANT_V0_SIDECAR_VERSION = 1U;
static const uint32_t EDGE_QUANT_V0_FIXED_HEADER_SIZE = 256U;
static const uint32_t EDGE_QUANT_V0_CHECKSUM_SECTION_SIZE = 176U;
static const uint32_t EDGE_QUANT_V0_CHECKSUM_SECTION_COUNT = 5U;

enum class V0Endianness : uint8_t {
    Little = 1U
};

enum class V0MetricType : uint8_t {
    SquaredL2 = 1U
};

enum class V0VectorType : uint8_t {
    Float32 = 1U
};

enum class V0MetadataScalarType : uint8_t {
    Float64 = 1U
};

enum V0EdgeFlags {
    V0_EDGE_EXACT_ONLY = 1U << 0,
    V0_ZERO_LENGTH_EDGE = 1U << 1,
    V0_RESERVED_INVALID = 1U << 2
};

typedef std::array<uint8_t, 32> V0Sha256Digest;

struct V0SectionDescriptor {
    uint64_t offset;
    uint64_t size;

    V0SectionDescriptor() : offset(0), size(0) {}
};

struct V0SidecarHeader {
    uint32_t format_version;
    uint32_t header_size;
    V0Endianness endianness;
    V0MetricType metric_type;
    V0VectorType vector_type;
    V0MetadataScalarType metadata_scalar_type;

    uint32_t dimension;
    uint32_t pq_m;
    uint32_t pq_nbits;
    uint32_t pq_ksub;
    uint32_t pq_dsub;
    uint32_t pq_code_size;
    uint32_t edge_record_stride;

    uint64_t node_count;
    uint64_t directed_edge_count;

    V0SectionDescriptor training_metadata;
    V0SectionDescriptor codebook;
    V0SectionDescriptor node_offsets;
    V0SectionDescriptor edge_records;
    V0SectionDescriptor checksums;

    V0Sha256Digest base_index_sha256;
    V0Sha256Digest adjacency_sha256;
    V0Sha256Digest codebook_sha256;

    V0SidecarHeader()
        : format_version(EDGE_QUANT_V0_SIDECAR_VERSION),
          header_size(EDGE_QUANT_V0_FIXED_HEADER_SIZE),
          endianness(V0Endianness::Little),
          metric_type(V0MetricType::SquaredL2),
          vector_type(V0VectorType::Float32),
          metadata_scalar_type(V0MetadataScalarType::Float64),
          dimension(0),
          pq_m(0),
          pq_nbits(0),
          pq_ksub(0),
          pq_dsub(0),
          pq_code_size(0),
          edge_record_stride(0),
          node_count(0),
          directed_edge_count(0) {
        base_index_sha256.fill(0U);
        adjacency_sha256.fill(0U);
        codebook_sha256.fill(0U);
    }
};

struct V0SectionChecksums {
    V0Sha256Digest header;
    V0Sha256Digest training_metadata;
    V0Sha256Digest codebook;
    V0Sha256Digest node_offsets;
    V0Sha256Digest edge_records;

    V0SectionChecksums() {
        header.fill(0U);
        training_metadata.fill(0U);
        codebook.fill(0U);
        node_offsets.fill(0U);
        edge_records.fill(0U);
    }
};

struct V0EdgeRecord {
    std::vector<uint8_t> code;
    uint8_t flags;
    double edge_length;
    double direction_error;
    double anchor_projection;
    double numeric_padding;

    V0EdgeRecord()
        : flags(0),
          edge_length(0.0),
          direction_error(0.0),
          anchor_projection(0.0),
          numeric_padding(0.0) {}
};

// Inputs known before streaming edge records. Node offsets deliberately stay
// in memory (8 * (node_count + 1) bytes); edge records do not.
struct V0SidecarWriteSpec {
    uint32_t dimension;
    uint32_t pq_m;
    uint32_t pq_nbits;
    uint32_t pq_ksub;
    uint32_t pq_dsub;
    uint64_t node_count;
    uint64_t directed_edge_count;

    std::string training_metadata_json;
    std::vector<float> codebook_centroids;
    std::vector<uint64_t> node_offsets;

    V0Sha256Digest base_index_sha256;
    V0Sha256Digest adjacency_sha256;

    V0SidecarWriteSpec()
        : dimension(0),
          pq_m(0),
          pq_nbits(0),
          pq_ksub(0),
          pq_dsub(0),
          node_count(0),
          directed_edge_count(0) {
        base_index_sha256.fill(0U);
        adjacency_sha256.fill(0U);
    }
};

struct V0IndexCompatibility {
    uint32_t dimension;
    uint64_t node_count;
    uint64_t directed_edge_count;
    V0MetricType metric_type;
    V0VectorType vector_type;
    V0Sha256Digest base_index_sha256;
    V0Sha256Digest adjacency_sha256;

    V0IndexCompatibility()
        : dimension(0),
          node_count(0),
          directed_edge_count(0),
          metric_type(V0MetricType::SquaredL2),
          vector_type(V0VectorType::Float32) {
        base_index_sha256.fill(0U);
        adjacency_sha256.fill(0U);
    }
};

// Defined in edge_quant_v0_query_metadata.h. Keeping the declaration here
// lets schema-only users avoid pulling query ownership and graph access into
// offline tools.
class EdgeQuantV0Metadata;

}  // namespace hnswlib
