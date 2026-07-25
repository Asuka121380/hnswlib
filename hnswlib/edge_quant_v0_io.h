#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "edge_quant_v0_checksum.h"
#include "edge_quant_v0_graph_access.h"
#include "edge_quant_v0_metadata.h"

namespace hnswlib {
namespace edge_quant_v0_detail {

static const uint8_t SIDECAR_MAGIC[8] = {
    'H', 'N', 'S', 'W', 'V', '0', 'P', 'Q'
};
static const uint8_t CHECKSUM_MAGIC[8] = {
    'V', '0', 'S', 'U', 'M', 'S', '0', '1'
};

inline uint64_t checkedAdd(uint64_t left, uint64_t right) {
    if (right > std::numeric_limits<uint64_t>::max() - left) {
        throw std::overflow_error("V0 sidecar integer addition overflow");
    }
    return left + right;
}

inline uint64_t checkedMultiply(uint64_t left, uint64_t right) {
    if (left != 0U &&
        right > std::numeric_limits<uint64_t>::max() / left) {
        throw std::overflow_error(
            "V0 sidecar integer multiplication overflow");
    }
    return left * right;
}

inline uint32_t alignUp8(uint32_t value) {
    if (value > std::numeric_limits<uint32_t>::max() - 7U) {
        throw std::overflow_error("V0 edge-record alignment overflow");
    }
    return (value + 7U) & ~static_cast<uint32_t>(7U);
}

inline uint32_t edgeRecordScalarOffset(uint32_t code_size) {
    return alignUp8(code_size + 1U);
}

inline uint32_t edgeRecordStride(uint32_t code_size) {
    const uint32_t scalar_offset = edgeRecordScalarOffset(code_size);
    if (scalar_offset > std::numeric_limits<uint32_t>::max() - 32U) {
        throw std::overflow_error("V0 edge-record stride overflow");
    }
    return scalar_offset + 32U;
}

inline void appendUint32LittleEndian(
    std::vector<uint8_t>& output,
    uint32_t value) {
    for (size_t i = 0; i < 4U; ++i) {
        output.push_back(
            static_cast<uint8_t>((value >> (i * 8U)) & 0xffU));
    }
}

inline void appendUint64LittleEndian(
    std::vector<uint8_t>& output,
    uint64_t value) {
    for (size_t i = 0; i < 8U; ++i) {
        output.push_back(
            static_cast<uint8_t>((value >> (i * 8U)) & 0xffU));
    }
}

inline void appendFloat32LittleEndian(
    std::vector<uint8_t>& output,
    float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    appendUint32LittleEndian(output, bits);
}

inline void appendFloat64LittleEndian(
    std::vector<uint8_t>& output,
    double value) {
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    appendUint64LittleEndian(output, bits);
}

inline uint32_t readUint32LittleEndian(const uint8_t* bytes) {
    uint32_t value = 0;
    for (size_t i = 0; i < 4U; ++i) {
        value |= static_cast<uint32_t>(bytes[i]) << (i * 8U);
    }
    return value;
}

inline uint64_t readUint64LittleEndian(const uint8_t* bytes) {
    uint64_t value = 0;
    for (size_t i = 0; i < 8U; ++i) {
        value |= static_cast<uint64_t>(bytes[i]) << (i * 8U);
    }
    return value;
}

inline float readFloat32LittleEndian(const uint8_t* bytes) {
    const uint32_t bits = readUint32LittleEndian(bytes);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline double readFloat64LittleEndian(const uint8_t* bytes) {
    const uint64_t bits = readUint64LittleEndian(bytes);
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline V0Sha256Digest sha256(const uint8_t* bytes, size_t size) {
    EdgeQuantV0Sha256 sha;
    if (size != 0U) {
        sha.update(bytes, size);
    }
    return sha.final();
}

inline void appendDigest(
    std::vector<uint8_t>& output,
    const V0Sha256Digest& digest) {
    output.insert(output.end(), digest.begin(), digest.end());
}

inline void writeBytes(
    std::ofstream& output,
    const uint8_t* bytes,
    size_t size) {
    if (size == 0U) {
        return;
    }
    if (size >
        static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error(
            "V0 sidecar write exceeds stream-size capacity");
    }
    output.write(
        reinterpret_cast<const char*>(bytes),
        static_cast<std::streamsize>(size));
    if (!output) {
        throw std::runtime_error("Failed to write V0 sidecar");
    }
}

inline void requireRange(
    uint64_t offset,
    uint64_t size,
    uint64_t file_size,
    const char* section_name) {
    if (offset > file_size || size > file_size - offset) {
        throw std::runtime_error(
            std::string("V0 sidecar section is out of bounds: ") +
            section_name);
    }
}

inline void requireEqualDigest(
    const V0Sha256Digest& expected,
    const V0Sha256Digest& actual,
    const char* section_name) {
    if (expected != actual) {
        throw std::runtime_error(
            std::string("V0 sidecar checksum mismatch: ") + section_name);
    }
}

inline std::vector<uint8_t> serializeCodebook(
    const std::vector<float>& centroids) {
    std::vector<uint8_t> bytes;
    bytes.reserve(centroids.size() * sizeof(float));
    for (size_t i = 0; i < centroids.size(); ++i) {
        if (!std::isfinite(static_cast<double>(centroids[i]))) {
            throw std::invalid_argument(
                "V0 codebook contains a non-finite centroid");
        }
        appendFloat32LittleEndian(bytes, centroids[i]);
    }
    return bytes;
}

inline std::vector<uint8_t> serializeNodeOffsets(
    const std::vector<uint64_t>& offsets) {
    std::vector<uint8_t> bytes;
    bytes.reserve(offsets.size() * sizeof(uint64_t));
    for (size_t i = 0; i < offsets.size(); ++i) {
        appendUint64LittleEndian(bytes, offsets[i]);
    }
    return bytes;
}

inline void validateWriteSpec(const V0SidecarWriteSpec& spec) {
    if (spec.dimension == 0U ||
        spec.pq_m == 0U ||
        spec.pq_nbits == 0U ||
        spec.pq_nbits > 8U ||
        spec.pq_dsub == 0U) {
        throw std::invalid_argument(
            "V0 sidecar has invalid PQ dimensions");
    }
    if (spec.pq_m > std::numeric_limits<uint32_t>::max() /
                        spec.pq_dsub ||
        spec.pq_m * spec.pq_dsub != spec.dimension) {
        throw std::invalid_argument(
            "V0 sidecar dimension must equal pq_m * pq_dsub");
    }
    const uint32_t expected_ksub = 1U << spec.pq_nbits;
    if (spec.pq_ksub != expected_ksub) {
        throw std::invalid_argument(
            "V0 sidecar pq_ksub does not match pq_nbits");
    }
    if (spec.node_count >
        static_cast<uint64_t>(std::numeric_limits<size_t>::max() - 1U)) {
        throw std::invalid_argument(
            "V0 sidecar node count exceeds addressable memory");
    }
    if (spec.node_offsets.size() !=
        static_cast<size_t>(spec.node_count) + 1U) {
        throw std::invalid_argument(
            "V0 node-offset table has the wrong length");
    }
    if (spec.node_offsets.empty() || spec.node_offsets[0] != 0U) {
        throw std::invalid_argument(
            "V0 node-offset table must start at zero");
    }
    for (size_t i = 1; i < spec.node_offsets.size(); ++i) {
        if (spec.node_offsets[i] < spec.node_offsets[i - 1U]) {
            throw std::invalid_argument(
                "V0 node-offset table is not monotonic");
        }
    }
    if (spec.node_offsets.back() != spec.directed_edge_count) {
        throw std::invalid_argument(
            "V0 node-offset table does not end at edge count");
    }

    const uint64_t centroid_count = checkedMultiply(
        checkedMultiply(spec.pq_m, spec.pq_ksub), spec.pq_dsub);
    if (centroid_count != spec.codebook_centroids.size()) {
        throw std::invalid_argument(
            "V0 codebook has the wrong centroid count");
    }
}

inline std::vector<uint8_t> serializeHeader(
    const V0SidecarHeader& header) {
    std::vector<uint8_t> output;
    output.reserve(EDGE_QUANT_V0_FIXED_HEADER_SIZE);
    output.insert(output.end(), SIDECAR_MAGIC, SIDECAR_MAGIC + 8U);
    appendUint32LittleEndian(output, header.format_version);
    appendUint32LittleEndian(output, header.header_size);
    output.push_back(static_cast<uint8_t>(header.endianness));
    output.push_back(static_cast<uint8_t>(header.metric_type));
    output.push_back(static_cast<uint8_t>(header.vector_type));
    output.push_back(static_cast<uint8_t>(header.metadata_scalar_type));
    appendUint32LittleEndian(output, header.dimension);
    appendUint32LittleEndian(output, header.pq_m);
    appendUint32LittleEndian(output, header.pq_nbits);
    appendUint32LittleEndian(output, header.pq_ksub);
    appendUint32LittleEndian(output, header.pq_dsub);
    appendUint32LittleEndian(output, header.pq_code_size);
    appendUint32LittleEndian(output, header.edge_record_stride);
    appendUint64LittleEndian(output, header.node_count);
    appendUint64LittleEndian(output, header.directed_edge_count);

    appendUint64LittleEndian(output, header.training_metadata.offset);
    appendUint64LittleEndian(output, header.training_metadata.size);
    appendUint64LittleEndian(output, header.codebook.offset);
    appendUint64LittleEndian(output, header.codebook.size);
    appendUint64LittleEndian(output, header.node_offsets.offset);
    appendUint64LittleEndian(output, header.node_offsets.size);
    appendUint64LittleEndian(output, header.edge_records.offset);
    appendUint64LittleEndian(output, header.edge_records.size);
    appendUint64LittleEndian(output, header.checksums.offset);
    appendUint64LittleEndian(output, header.checksums.size);

    appendDigest(output, header.base_index_sha256);
    appendDigest(output, header.adjacency_sha256);
    appendDigest(output, header.codebook_sha256);
    output.resize(EDGE_QUANT_V0_FIXED_HEADER_SIZE, 0U);
    return output;
}

inline V0SidecarHeader parseHeader(
    const uint8_t* bytes,
    size_t file_size) {
    if (file_size < EDGE_QUANT_V0_FIXED_HEADER_SIZE) {
        throw std::runtime_error("V0 sidecar is shorter than its header");
    }
    if (!std::equal(SIDECAR_MAGIC, SIDECAR_MAGIC + 8U, bytes)) {
        throw std::runtime_error("V0 sidecar magic is invalid");
    }

    V0SidecarHeader header;
    header.format_version = readUint32LittleEndian(bytes + 8U);
    header.header_size = readUint32LittleEndian(bytes + 12U);
    header.endianness = static_cast<V0Endianness>(bytes[16U]);
    header.metric_type = static_cast<V0MetricType>(bytes[17U]);
    header.vector_type = static_cast<V0VectorType>(bytes[18U]);
    header.metadata_scalar_type =
        static_cast<V0MetadataScalarType>(bytes[19U]);
    header.dimension = readUint32LittleEndian(bytes + 20U);
    header.pq_m = readUint32LittleEndian(bytes + 24U);
    header.pq_nbits = readUint32LittleEndian(bytes + 28U);
    header.pq_ksub = readUint32LittleEndian(bytes + 32U);
    header.pq_dsub = readUint32LittleEndian(bytes + 36U);
    header.pq_code_size = readUint32LittleEndian(bytes + 40U);
    header.edge_record_stride = readUint32LittleEndian(bytes + 44U);
    header.node_count = readUint64LittleEndian(bytes + 48U);
    header.directed_edge_count = readUint64LittleEndian(bytes + 56U);

    header.training_metadata.offset =
        readUint64LittleEndian(bytes + 64U);
    header.training_metadata.size =
        readUint64LittleEndian(bytes + 72U);
    header.codebook.offset = readUint64LittleEndian(bytes + 80U);
    header.codebook.size = readUint64LittleEndian(bytes + 88U);
    header.node_offsets.offset = readUint64LittleEndian(bytes + 96U);
    header.node_offsets.size = readUint64LittleEndian(bytes + 104U);
    header.edge_records.offset = readUint64LittleEndian(bytes + 112U);
    header.edge_records.size = readUint64LittleEndian(bytes + 120U);
    header.checksums.offset = readUint64LittleEndian(bytes + 128U);
    header.checksums.size = readUint64LittleEndian(bytes + 136U);

    std::copy(bytes + 144U, bytes + 176U,
              header.base_index_sha256.begin());
    std::copy(bytes + 176U, bytes + 208U,
              header.adjacency_sha256.begin());
    std::copy(bytes + 208U, bytes + 240U,
              header.codebook_sha256.begin());

    for (size_t i = 240U; i < EDGE_QUANT_V0_FIXED_HEADER_SIZE; ++i) {
        if (bytes[i] != 0U) {
            throw std::runtime_error(
                "V0 sidecar reserved header bytes are non-zero");
        }
    }
    return header;
}

inline V0SectionChecksums parseChecksums(
    const uint8_t* bytes,
    size_t size) {
    if (size != EDGE_QUANT_V0_CHECKSUM_SECTION_SIZE) {
        throw std::runtime_error(
            "V0 sidecar checksum section has the wrong size");
    }
    if (!std::equal(CHECKSUM_MAGIC, CHECKSUM_MAGIC + 8U, bytes)) {
        throw std::runtime_error(
            "V0 sidecar checksum-section magic is invalid");
    }
    if (readUint32LittleEndian(bytes + 8U) !=
            EDGE_QUANT_V0_SIDECAR_VERSION ||
        readUint32LittleEndian(bytes + 12U) !=
            EDGE_QUANT_V0_CHECKSUM_SECTION_COUNT) {
        throw std::runtime_error(
            "V0 sidecar checksum-section version is invalid");
    }

    V0SectionChecksums checksums;
    std::copy(bytes + 16U, bytes + 48U, checksums.header.begin());
    std::copy(bytes + 48U, bytes + 80U,
              checksums.training_metadata.begin());
    std::copy(bytes + 80U, bytes + 112U, checksums.codebook.begin());
    std::copy(bytes + 112U, bytes + 144U,
              checksums.node_offsets.begin());
    std::copy(bytes + 144U, bytes + 176U,
              checksums.edge_records.begin());
    return checksums;
}

inline std::vector<uint8_t> serializeChecksums(
    const V0SectionChecksums& checksums) {
    std::vector<uint8_t> output;
    output.reserve(EDGE_QUANT_V0_CHECKSUM_SECTION_SIZE);
    output.insert(output.end(), CHECKSUM_MAGIC, CHECKSUM_MAGIC + 8U);
    appendUint32LittleEndian(output, EDGE_QUANT_V0_SIDECAR_VERSION);
    appendUint32LittleEndian(output, EDGE_QUANT_V0_CHECKSUM_SECTION_COUNT);
    appendDigest(output, checksums.header);
    appendDigest(output, checksums.training_metadata);
    appendDigest(output, checksums.codebook);
    appendDigest(output, checksums.node_offsets);
    appendDigest(output, checksums.edge_records);
    return output;
}

inline void validateHeaderAndLayout(
    const V0SidecarHeader& header,
    size_t file_size) {
    if (header.format_version != EDGE_QUANT_V0_SIDECAR_VERSION ||
        header.header_size != EDGE_QUANT_V0_FIXED_HEADER_SIZE) {
        throw std::runtime_error(
            "V0 sidecar version or header size is unsupported");
    }
    if (header.endianness != V0Endianness::Little ||
        header.metric_type != V0MetricType::SquaredL2 ||
        header.vector_type != V0VectorType::Float32 ||
        header.metadata_scalar_type != V0MetadataScalarType::Float64) {
        throw std::runtime_error(
            "V0 sidecar metric or scalar representation is unsupported");
    }
    if (header.dimension == 0U ||
        header.pq_m == 0U ||
        header.pq_nbits == 0U ||
        header.pq_nbits > 8U ||
        header.pq_dsub == 0U ||
        header.pq_ksub != (1U << header.pq_nbits) ||
        header.pq_code_size != header.pq_m ||
        header.edge_record_stride !=
            edgeRecordStride(header.pq_code_size) ||
        header.pq_m >
            std::numeric_limits<uint32_t>::max() / header.pq_dsub ||
        header.pq_m * header.pq_dsub != header.dimension) {
        throw std::runtime_error(
            "V0 sidecar PQ configuration is inconsistent");
    }

    const uint64_t expected_codebook_size = checkedMultiply(
        checkedMultiply(header.pq_m, header.pq_ksub),
        checkedMultiply(header.pq_dsub, sizeof(float)));
    const uint64_t expected_node_offsets_size = checkedMultiply(
        checkedAdd(header.node_count, 1U), sizeof(uint64_t));
    const uint64_t expected_edge_records_size = checkedMultiply(
        header.directed_edge_count, header.edge_record_stride);

    if (header.codebook.size != expected_codebook_size ||
        header.node_offsets.size != expected_node_offsets_size ||
        header.edge_records.size != expected_edge_records_size ||
        header.checksums.size != EDGE_QUANT_V0_CHECKSUM_SECTION_SIZE) {
        throw std::runtime_error(
            "V0 sidecar section size is inconsistent with its header");
    }

    uint64_t expected_offset = EDGE_QUANT_V0_FIXED_HEADER_SIZE;
    if (header.training_metadata.offset != expected_offset) {
        throw std::runtime_error(
            "V0 sidecar training-metadata offset is not canonical");
    }
    expected_offset = checkedAdd(
        expected_offset, header.training_metadata.size);
    if (header.codebook.offset != expected_offset) {
        throw std::runtime_error(
            "V0 sidecar codebook offset is not canonical");
    }
    expected_offset = checkedAdd(expected_offset, header.codebook.size);
    if (header.node_offsets.offset != expected_offset) {
        throw std::runtime_error(
            "V0 sidecar node-offset offset is not canonical");
    }
    expected_offset = checkedAdd(expected_offset, header.node_offsets.size);
    if (header.edge_records.offset != expected_offset) {
        throw std::runtime_error(
            "V0 sidecar edge-record offset is not canonical");
    }
    expected_offset = checkedAdd(expected_offset, header.edge_records.size);
    if (header.checksums.offset != expected_offset) {
        throw std::runtime_error(
            "V0 sidecar checksum offset is not canonical");
    }
    expected_offset = checkedAdd(expected_offset, header.checksums.size);
    if (expected_offset != static_cast<uint64_t>(file_size)) {
        throw std::runtime_error(
            "V0 sidecar file size does not match its canonical layout");
    }

    requireRange(
        header.training_metadata.offset,
        header.training_metadata.size,
        file_size,
        "training metadata");
    requireRange(
        header.codebook.offset,
        header.codebook.size,
        file_size,
        "codebook");
    requireRange(
        header.node_offsets.offset,
        header.node_offsets.size,
        file_size,
        "node offsets");
    requireRange(
        header.edge_records.offset,
        header.edge_records.size,
        file_size,
        "edge records");
    requireRange(
        header.checksums.offset,
        header.checksums.size,
        file_size,
        "checksums");
}

inline void validateNodeOffsets(
    const uint8_t* bytes,
    const V0SidecarHeader& header) {
    uint64_t previous = readUint64LittleEndian(bytes);
    if (previous != 0U) {
        throw std::runtime_error(
            "V0 sidecar node offsets do not start at zero");
    }
    for (uint64_t i = 1U; i <= header.node_count; ++i) {
        const uint64_t current = readUint64LittleEndian(bytes + i * 8U);
        if (current < previous ||
            current > header.directed_edge_count) {
            throw std::runtime_error(
                "V0 sidecar node offsets are not monotonic");
        }
        previous = current;
    }
    if (previous != header.directed_edge_count) {
        throw std::runtime_error(
            "V0 sidecar node offsets do not end at edge count");
    }
}

inline void validateCodebook(
    const uint8_t* bytes,
    const V0SidecarHeader& header) {
    const uint64_t centroid_count =
        header.codebook.size / sizeof(float);
    for (uint64_t i = 0U; i < centroid_count; ++i) {
        if (!std::isfinite(static_cast<double>(
                readFloat32LittleEndian(bytes + i * sizeof(float))))) {
            throw std::runtime_error(
                "V0 sidecar codebook contains a non-finite centroid");
        }
    }
}

inline void validateEdgeRecords(
    const uint8_t* bytes,
    const V0SidecarHeader& header) {
    const uint32_t scalar_offset =
        edgeRecordScalarOffset(header.pq_code_size);
    const uint8_t allowed_flags =
        V0_EDGE_EXACT_ONLY | V0_ZERO_LENGTH_EDGE | V0_RESERVED_INVALID;
    for (uint64_t edge = 0U; edge < header.directed_edge_count; ++edge) {
        const uint8_t* record =
            bytes + edge * header.edge_record_stride;
        for (uint32_t m = 0U; m < header.pq_code_size; ++m) {
            if (record[m] >= header.pq_ksub) {
                throw std::runtime_error(
                    "V0 sidecar edge record contains an invalid PQ code");
            }
        }
        if ((record[header.pq_code_size] &
             static_cast<uint8_t>(~allowed_flags)) != 0U) {
            throw std::runtime_error(
                "V0 sidecar edge record contains unknown flags");
        }
        for (uint32_t i = header.pq_code_size + 1U;
             i < scalar_offset;
             ++i) {
            if (record[i] != 0U) {
                throw std::runtime_error(
                    "V0 sidecar edge-record padding is non-zero");
            }
        }

        const double edge_length =
            readFloat64LittleEndian(record + scalar_offset);
        const double direction_error =
            readFloat64LittleEndian(record + scalar_offset + 8U);
        const double anchor_projection =
            readFloat64LittleEndian(record + scalar_offset + 16U);
        const double numeric_padding =
            readFloat64LittleEndian(record + scalar_offset + 24U);
        if (!std::isfinite(edge_length) || edge_length < 0.0 ||
            !std::isfinite(direction_error) || direction_error < 0.0 ||
            !std::isfinite(anchor_projection) ||
            !std::isfinite(numeric_padding) || numeric_padding < 0.0) {
            throw std::runtime_error(
                "V0 sidecar edge record contains invalid scalar metadata");
        }
    }
}

inline V0SidecarHeader validateSidecarBytes(
    const uint8_t* bytes,
    size_t file_size) {
    if (bytes == NULL && file_size != 0U) {
        throw std::invalid_argument(
            "V0 sidecar parser received null file bytes");
    }
    const V0SidecarHeader header = parseHeader(bytes, file_size);
    validateHeaderAndLayout(header, file_size);

    const uint8_t* checksum_bytes = bytes + header.checksums.offset;
    const V0SectionChecksums checksums = parseChecksums(
        checksum_bytes, static_cast<size_t>(header.checksums.size));

    requireEqualDigest(
        checksums.header,
        sha256(bytes, EDGE_QUANT_V0_FIXED_HEADER_SIZE),
        "header");
    requireEqualDigest(
        checksums.training_metadata,
        sha256(
            bytes + header.training_metadata.offset,
            static_cast<size_t>(header.training_metadata.size)),
        "training metadata");
    const V0Sha256Digest actual_codebook = sha256(
        bytes + header.codebook.offset,
        static_cast<size_t>(header.codebook.size));
    requireEqualDigest(checksums.codebook, actual_codebook, "codebook");
    requireEqualDigest(
        header.codebook_sha256, actual_codebook, "header codebook");
    requireEqualDigest(
        checksums.node_offsets,
        sha256(
            bytes + header.node_offsets.offset,
            static_cast<size_t>(header.node_offsets.size)),
        "node offsets");
    requireEqualDigest(
        checksums.edge_records,
        sha256(
            bytes + header.edge_records.offset,
            static_cast<size_t>(header.edge_records.size)),
        "edge records");

    validateCodebook(bytes + header.codebook.offset, header);
    validateNodeOffsets(bytes + header.node_offsets.offset, header);
    validateEdgeRecords(bytes + header.edge_records.offset, header);
    return header;
}

}  // namespace edge_quant_v0_detail

class V0EdgeRecordView {
 public:
    V0EdgeRecordView(
        const uint8_t* record,
        const V0SidecarHeader& header)
        : record_(record),
          code_size_(header.pq_code_size),
          scalar_offset_(
              edge_quant_v0_detail::edgeRecordScalarOffset(
                  header.pq_code_size)) {}

    uint8_t code(size_t subquantizer) const {
        if (subquantizer >= code_size_) {
            throw std::out_of_range(
                "V0 PQ subquantizer index is out of range");
        }
        return record_[subquantizer];
    }

    uint32_t codeSize() const {
        return code_size_;
    }

    uint8_t flags() const {
        return record_[code_size_];
    }

    double edgeLength() const {
        return edge_quant_v0_detail::readFloat64LittleEndian(
            record_ + scalar_offset_);
    }

    double directionError() const {
        return edge_quant_v0_detail::readFloat64LittleEndian(
            record_ + scalar_offset_ + 8U);
    }

    double anchorProjection() const {
        return edge_quant_v0_detail::readFloat64LittleEndian(
            record_ + scalar_offset_ + 16U);
    }

    double numericPadding() const {
        return edge_quant_v0_detail::readFloat64LittleEndian(
            record_ + scalar_offset_ + 24U);
    }

 private:
    const uint8_t* record_;
    uint32_t code_size_;
    uint32_t scalar_offset_;
};

class V0SidecarView {
 public:
    const V0SidecarHeader& header() const {
        return header_;
    }

    size_t fileSize() const {
        return size_;
    }

    std::string trainingMetadataJson() const {
        return std::string(
            reinterpret_cast<const char*>(
                bytes_ + header_.training_metadata.offset),
            static_cast<size_t>(header_.training_metadata.size));
    }

    float codebookCentroid(size_t flat_index) const {
        const size_t count =
            static_cast<size_t>(header_.codebook.size / sizeof(float));
        if (flat_index >= count) {
            throw std::out_of_range(
                "V0 codebook centroid index is out of range");
        }
        return edge_quant_v0_detail::readFloat32LittleEndian(
            bytes_ + header_.codebook.offset + flat_index * sizeof(float));
    }

    uint64_t nodeOffset(size_t node_offset_index) const {
        if (node_offset_index > header_.node_count) {
            throw std::out_of_range(
                "V0 node-offset index is out of range");
        }
        return edge_quant_v0_detail::readUint64LittleEndian(
            bytes_ + header_.node_offsets.offset +
            node_offset_index * sizeof(uint64_t));
    }

    V0EdgeRecordView edgeRecord(size_t edge_index) const {
        if (edge_index >= header_.directed_edge_count) {
            throw std::out_of_range(
                "V0 edge-record index is out of range");
        }
        return V0EdgeRecordView(
            bytes_ + header_.edge_records.offset +
                edge_index * header_.edge_record_stride,
            header_);
    }

 private:
    V0SidecarView(
        const uint8_t* bytes,
        size_t size,
        const V0SidecarHeader& header)
        : bytes_(bytes), size_(size), header_(header) {}

    friend class V0OwnedSidecar;
    friend V0SidecarView parseV0Sidecar(
        const uint8_t* bytes,
        size_t size);

    const uint8_t* bytes_;
    size_t size_;
    V0SidecarHeader header_;
};

class V0OwnedSidecar {
 public:
    explicit V0OwnedSidecar(std::vector<uint8_t> bytes)
        : bytes_(std::move(bytes)),
          header_(edge_quant_v0_detail::validateSidecarBytes(
              bytes_.empty() ? NULL : bytes_.data(),
              bytes_.size())) {}

    V0OwnedSidecar(V0OwnedSidecar&&) = default;
    V0OwnedSidecar& operator=(V0OwnedSidecar&&) = default;
    V0OwnedSidecar(const V0OwnedSidecar&) = delete;
    V0OwnedSidecar& operator=(const V0OwnedSidecar&) = delete;

    V0SidecarView view() const {
        return V0SidecarView(bytes_.data(), bytes_.size(), header_);
    }

    const V0SidecarHeader& header() const {
        return header_;
    }

 private:
    std::vector<uint8_t> bytes_;
    V0SidecarHeader header_;
};

class V0SidecarWriter {
 public:
    V0SidecarWriter(
        const std::string& path,
        const V0SidecarWriteSpec& spec)
        : written_edge_count_(0U),
          finalized_(false) {
        edge_quant_v0_detail::validateWriteSpec(spec);
        output_.open(path.c_str(), std::ios::binary | std::ios::trunc);
        if (!output_) {
            throw std::runtime_error("Failed to open V0 sidecar for writing");
        }

        training_bytes_.assign(
            spec.training_metadata_json.begin(),
            spec.training_metadata_json.end());
        codebook_bytes_ =
            edge_quant_v0_detail::serializeCodebook(
                spec.codebook_centroids);
        node_offset_bytes_ =
            edge_quant_v0_detail::serializeNodeOffsets(spec.node_offsets);

        header_.dimension = spec.dimension;
        header_.pq_m = spec.pq_m;
        header_.pq_nbits = spec.pq_nbits;
        header_.pq_ksub = spec.pq_ksub;
        header_.pq_dsub = spec.pq_dsub;
        header_.pq_code_size = spec.pq_m;
        header_.edge_record_stride =
            edge_quant_v0_detail::edgeRecordStride(header_.pq_code_size);
        header_.node_count = spec.node_count;
        header_.directed_edge_count = spec.directed_edge_count;
        header_.base_index_sha256 = spec.base_index_sha256;
        header_.adjacency_sha256 = spec.adjacency_sha256;
        header_.codebook_sha256 = edge_quant_v0_detail::sha256(
            codebook_bytes_.data(), codebook_bytes_.size());

        uint64_t offset = EDGE_QUANT_V0_FIXED_HEADER_SIZE;
        header_.training_metadata.offset = offset;
        header_.training_metadata.size = training_bytes_.size();
        offset = edge_quant_v0_detail::checkedAdd(
            offset, header_.training_metadata.size);
        header_.codebook.offset = offset;
        header_.codebook.size = codebook_bytes_.size();
        offset = edge_quant_v0_detail::checkedAdd(
            offset, header_.codebook.size);
        header_.node_offsets.offset = offset;
        header_.node_offsets.size = node_offset_bytes_.size();
        offset = edge_quant_v0_detail::checkedAdd(
            offset, header_.node_offsets.size);
        header_.edge_records.offset = offset;
        header_.edge_records.size =
            edge_quant_v0_detail::checkedMultiply(
                header_.directed_edge_count,
                header_.edge_record_stride);
        offset = edge_quant_v0_detail::checkedAdd(
            offset, header_.edge_records.size);
        header_.checksums.offset = offset;
        header_.checksums.size =
            EDGE_QUANT_V0_CHECKSUM_SECTION_SIZE;

        header_bytes_ =
            edge_quant_v0_detail::serializeHeader(header_);
        edge_quant_v0_detail::writeBytes(
            output_, header_bytes_.data(), header_bytes_.size());
        edge_quant_v0_detail::writeBytes(
            output_, training_bytes_.data(), training_bytes_.size());
        edge_quant_v0_detail::writeBytes(
            output_, codebook_bytes_.data(), codebook_bytes_.size());
        edge_quant_v0_detail::writeBytes(
            output_, node_offset_bytes_.data(), node_offset_bytes_.size());
    }

    ~V0SidecarWriter() {
        output_.close();
    }

    V0SidecarWriter(const V0SidecarWriter&) = delete;
    V0SidecarWriter& operator=(const V0SidecarWriter&) = delete;

    const V0SidecarHeader& header() const {
        return header_;
    }

    void writeEdgeRecord(const V0EdgeRecord& record) {
        writeEdgeRecord(
            record.code.empty() ? NULL : record.code.data(),
            record.code.size(),
            record.flags,
            record.edge_length,
            record.direction_error,
            record.anchor_projection,
            record.numeric_padding);
    }

    void writeEdgeRecord(
        const uint8_t* code,
        size_t code_size,
        uint8_t flags,
        double edge_length,
        double direction_error,
        double anchor_projection,
        double numeric_padding) {
        if (finalized_) {
            throw std::logic_error(
                "Cannot append to a finalized V0 sidecar");
        }
        if (written_edge_count_ >= header_.directed_edge_count) {
            throw std::runtime_error(
                "Too many edge records written to V0 sidecar");
        }
        if (code_size != header_.pq_code_size ||
            (code_size != 0U && code == NULL)) {
            throw std::invalid_argument(
                "V0 edge record has the wrong PQ code size");
        }
        for (size_t i = 0; i < code_size; ++i) {
            if (code[i] >= header_.pq_ksub) {
                throw std::invalid_argument(
                    "V0 edge record contains an invalid PQ code");
            }
        }
        const uint8_t allowed_flags =
            V0_EDGE_EXACT_ONLY | V0_ZERO_LENGTH_EDGE |
            V0_RESERVED_INVALID;
        if ((flags & static_cast<uint8_t>(~allowed_flags)) != 0U) {
            throw std::invalid_argument(
                "V0 edge record contains unknown flags");
        }
        if (!std::isfinite(edge_length) || edge_length < 0.0 ||
            !std::isfinite(direction_error) || direction_error < 0.0 ||
            !std::isfinite(anchor_projection) ||
            !std::isfinite(numeric_padding) || numeric_padding < 0.0) {
            throw std::invalid_argument(
                "V0 edge record contains invalid scalar metadata");
        }

        std::vector<uint8_t> bytes;
        bytes.reserve(header_.edge_record_stride);
        bytes.insert(bytes.end(), code, code + code_size);
        bytes.push_back(flags);
        bytes.resize(
            edge_quant_v0_detail::edgeRecordScalarOffset(
                header_.pq_code_size),
            0U);
        edge_quant_v0_detail::appendFloat64LittleEndian(
            bytes, edge_length);
        edge_quant_v0_detail::appendFloat64LittleEndian(
            bytes, direction_error);
        edge_quant_v0_detail::appendFloat64LittleEndian(
            bytes, anchor_projection);
        edge_quant_v0_detail::appendFloat64LittleEndian(
            bytes, numeric_padding);
        if (bytes.size() != header_.edge_record_stride) {
            throw std::logic_error(
                "V0 edge-record serializer produced the wrong stride");
        }
        edge_quant_v0_detail::writeBytes(
            output_, bytes.data(), bytes.size());
        edge_record_sha_.update(bytes.data(), bytes.size());
        ++written_edge_count_;
    }

    void finalize() {
        if (finalized_) {
            throw std::logic_error("V0 sidecar is already finalized");
        }
        if (written_edge_count_ != header_.directed_edge_count) {
            throw std::runtime_error(
                "V0 sidecar edge-record count is incomplete");
        }

        V0SectionChecksums checksums;
        checksums.header = edge_quant_v0_detail::sha256(
            header_bytes_.data(), header_bytes_.size());
        checksums.training_metadata = edge_quant_v0_detail::sha256(
            training_bytes_.data(), training_bytes_.size());
        checksums.codebook = edge_quant_v0_detail::sha256(
            codebook_bytes_.data(), codebook_bytes_.size());
        checksums.node_offsets = edge_quant_v0_detail::sha256(
            node_offset_bytes_.data(), node_offset_bytes_.size());
        checksums.edge_records = edge_record_sha_.final();
        const std::vector<uint8_t> checksum_bytes =
            edge_quant_v0_detail::serializeChecksums(checksums);
        edge_quant_v0_detail::writeBytes(
            output_, checksum_bytes.data(), checksum_bytes.size());
        output_.flush();
        if (!output_) {
            throw std::runtime_error("Failed to finalize V0 sidecar");
        }
        output_.close();
        finalized_ = true;
    }

 private:
    std::ofstream output_;
    V0SidecarHeader header_;
    std::vector<uint8_t> header_bytes_;
    std::vector<uint8_t> training_bytes_;
    std::vector<uint8_t> codebook_bytes_;
    std::vector<uint8_t> node_offset_bytes_;
    EdgeQuantV0Sha256 edge_record_sha_;
    uint64_t written_edge_count_;
    bool finalized_;
};

inline V0SidecarView parseV0Sidecar(
    const uint8_t* bytes,
    size_t size) {
    const V0SidecarHeader header =
        edge_quant_v0_detail::validateSidecarBytes(bytes, size);
    return V0SidecarView(bytes, size, header);
}

inline V0OwnedSidecar loadV0Sidecar(const std::string& path) {
    std::ifstream input(path.c_str(), std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Failed to open V0 sidecar for reading");
    }
    const std::streamoff end = input.tellg();
    if (end < 0) {
        throw std::runtime_error("Failed to determine V0 sidecar size");
    }
    if (static_cast<uint64_t>(end) >
        static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error(
            "V0 sidecar is too large for this address space");
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(end));
    if (bytes.size() >
        static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error(
            "V0 sidecar read exceeds stream-size capacity");
    }
    input.seekg(0, std::ios::beg);
    if (!bytes.empty()) {
        input.read(
            reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    }
    if (!input && !bytes.empty()) {
        throw std::runtime_error("Failed to read complete V0 sidecar");
    }
    return V0OwnedSidecar(std::move(bytes));
}

inline V0Sha256Digest computeV0FileSha256(const std::string& path) {
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open file for V0 SHA-256");
    }
    EdgeQuantV0Sha256 sha;
    std::vector<char> buffer(1024U * 1024U);
    while (input) {
        input.read(
            buffer.data(),
            static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) {
            sha.update(buffer.data(), static_cast<size_t>(count));
        }
    }
    if (!input.eof()) {
        throw std::runtime_error("Failed while hashing file for V0");
    }
    return sha.final();
}

inline void validateV0SidecarCompatibility(
    const V0SidecarHeader& header,
    const V0IndexCompatibility& expected) {
    if (header.dimension != expected.dimension) {
        throw std::runtime_error(
            "V0 sidecar dimension does not match the index");
    }
    if (header.node_count != expected.node_count) {
        throw std::runtime_error(
            "V0 sidecar node count does not match the index");
    }
    if (header.directed_edge_count != expected.directed_edge_count) {
        throw std::runtime_error(
            "V0 sidecar edge count does not match the index");
    }
    if (header.metric_type != expected.metric_type ||
        header.vector_type != expected.vector_type) {
        throw std::runtime_error(
            "V0 sidecar metric or vector type does not match the index");
    }
    if (header.base_index_sha256 != expected.base_index_sha256) {
        throw std::runtime_error(
            "V0 sidecar base-index fingerprint does not match");
    }
    if (header.adjacency_sha256 != expected.adjacency_sha256) {
        throw std::runtime_error(
            "V0 sidecar adjacency fingerprint does not match");
    }
}

inline void validateV0SidecarGraphLayout(
    const V0SidecarView& sidecar,
    const V0Layer0GraphView& graph) {
    const V0SidecarHeader& header = sidecar.header();
    if (header.node_count != graph.nodeCount()) {
        throw std::runtime_error(
            "V0 sidecar node count does not match layer-0 graph view");
    }
    for (size_t node = 0; node < graph.nodeCount(); ++node) {
        const uint64_t first = sidecar.nodeOffset(node);
        const uint64_t last = sidecar.nodeOffset(node + 1U);
        if (last - first !=
            graph.neighbors(static_cast<tableint>(node)).size) {
            throw std::runtime_error(
                "V0 sidecar edge slots do not match layer-0 adjacency");
        }
    }
}

// Compatibility name retained from the isolation scaffold.
struct EdgeQuantV0Io {};

}  // namespace hnswlib
