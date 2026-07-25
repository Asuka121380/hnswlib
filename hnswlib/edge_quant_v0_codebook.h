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
#include <vector>

#include "edge_quant_v0_checksum.h"

namespace hnswlib {

static const uint32_t EDGE_QUANT_V0_PQ_CODEBOOK_VERSION = 1U;
static const uint32_t EDGE_QUANT_V0_PQ_HEADER_SIZE = 256U;
static const uint32_t EDGE_QUANT_V0_PQ_CHECKSUM_SIZE = 112U;

struct V0PQCodebook {
    uint32_t dimension;
    uint32_t pq_m;
    uint32_t pq_nbits;
    uint32_t pq_ksub;
    uint32_t pq_dsub;
    uint32_t code_size;
    uint64_t centroid_count;
    std::string training_metadata_json;
    std::vector<float> centroids;
    std::array<uint8_t, 32> source_manifest_sha256;
    std::array<uint8_t, 32> source_directions_sha256;
    std::array<uint8_t, 32> centroids_sha256;
    std::array<uint8_t, 32> file_sha256;

    V0PQCodebook()
        : dimension(0U),
          pq_m(0U),
          pq_nbits(0U),
          pq_ksub(0U),
          pq_dsub(0U),
          code_size(0U),
          centroid_count(0U) {
        source_manifest_sha256.fill(0U);
        source_directions_sha256.fill(0U);
        centroids_sha256.fill(0U);
        file_sha256.fill(0U);
    }
};

namespace edge_quant_v0_codebook_detail {

static const uint8_t MAGIC[8] = {
    'V', '0', 'P', 'Q', 'B', 'O', 'O', 'K'
};
static const uint8_t CHECKSUM_MAGIC[8] = {
    'V', '0', 'P', 'Q', 'S', 'U', 'M', '1'
};

inline uint32_t readUint32(const uint8_t* bytes) {
    uint32_t value = 0U;
    for (size_t i = 0U; i < 4U; ++i) {
        value |= static_cast<uint32_t>(bytes[i]) << (8U * i);
    }
    return value;
}

inline uint64_t readUint64(const uint8_t* bytes) {
    uint64_t value = 0U;
    for (size_t i = 0U; i < 8U; ++i) {
        value |= static_cast<uint64_t>(bytes[i]) << (8U * i);
    }
    return value;
}

inline float readFloat32(const uint8_t* bytes) {
    const uint32_t bits = readUint32(bytes);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline uint64_t checkedAdd(uint64_t left, uint64_t right) {
    if (right > std::numeric_limits<uint64_t>::max() - left) {
        throw std::overflow_error("V0PQ section offset overflow");
    }
    return left + right;
}

inline uint64_t checkedMultiply(uint64_t left, uint64_t right) {
    if (left != 0U &&
        right > std::numeric_limits<uint64_t>::max() / left) {
        throw std::overflow_error("V0PQ section size overflow");
    }
    return left * right;
}

inline std::array<uint8_t, 32> sha256(
    const uint8_t* bytes,
    size_t size) {
    EdgeQuantV0Sha256 sha;
    if (size != 0U) {
        sha.update(bytes, size);
    }
    return sha.final();
}

inline std::vector<uint8_t> readFile(const std::string& path) {
    std::ifstream input(path.c_str(), std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Failed to open V0PQ codebook");
    }
    const std::streamoff end = input.tellg();
    if (end < 0 ||
        static_cast<uint64_t>(end) >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("Invalid V0PQ codebook size");
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(end));
    input.seekg(0, std::ios::beg);
    if (!bytes.empty()) {
        input.read(
            reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    }
    if (!input && !bytes.empty()) {
        throw std::runtime_error("Failed to read complete V0PQ codebook");
    }
    return bytes;
}

inline void requireDigest(
    const uint8_t* expected,
    const std::array<uint8_t, 32>& actual,
    const char* section) {
    if (!std::equal(actual.begin(), actual.end(), expected)) {
        throw std::runtime_error(
            std::string("V0PQ checksum mismatch: ") + section);
    }
}

}  // namespace edge_quant_v0_codebook_detail

inline V0PQCodebook loadV0PQCodebook(const std::string& path) {
    using namespace edge_quant_v0_codebook_detail;
    const std::vector<uint8_t> bytes = readFile(path);
    if (bytes.size() < EDGE_QUANT_V0_PQ_HEADER_SIZE) {
        throw std::runtime_error("V0PQ file is shorter than its header");
    }
    if (!std::equal(MAGIC, MAGIC + 8U, bytes.begin())) {
        throw std::runtime_error("V0PQ magic is invalid");
    }
    if (readUint32(bytes.data() + 8U) !=
            EDGE_QUANT_V0_PQ_CODEBOOK_VERSION ||
        readUint32(bytes.data() + 12U) !=
            EDGE_QUANT_V0_PQ_HEADER_SIZE) {
        throw std::runtime_error("Unsupported V0PQ version or header size");
    }
    const uint8_t representation[4] = {1U, 1U, 1U, 0U};
    if (!std::equal(
            representation,
            representation + 4U,
            bytes.begin() + 16U)) {
        throw std::runtime_error("Unsupported V0PQ representation");
    }

    V0PQCodebook result;
    result.dimension = readUint32(bytes.data() + 20U);
    result.pq_m = readUint32(bytes.data() + 24U);
    result.pq_nbits = readUint32(bytes.data() + 28U);
    result.pq_ksub = readUint32(bytes.data() + 32U);
    result.pq_dsub = readUint32(bytes.data() + 36U);
    result.code_size = readUint32(bytes.data() + 40U);
    result.centroid_count = readUint64(bytes.data() + 44U);
    const uint64_t metadata_offset = readUint64(bytes.data() + 52U);
    const uint64_t metadata_size = readUint64(bytes.data() + 60U);
    const uint64_t centroids_offset = readUint64(bytes.data() + 68U);
    const uint64_t centroids_size = readUint64(bytes.data() + 76U);
    const uint64_t checksums_offset = readUint64(bytes.data() + 84U);
    const uint64_t checksums_size = readUint64(bytes.data() + 92U);
    std::copy(
        bytes.begin() + 100U,
        bytes.begin() + 132U,
        result.source_manifest_sha256.begin());
    std::copy(
        bytes.begin() + 132U,
        bytes.begin() + 164U,
        result.source_directions_sha256.begin());
    std::copy(
        bytes.begin() + 164U,
        bytes.begin() + 196U,
        result.centroids_sha256.begin());

    for (size_t i = 196U; i < EDGE_QUANT_V0_PQ_HEADER_SIZE; ++i) {
        if (bytes[i] != 0U) {
            throw std::runtime_error(
                "V0PQ reserved header bytes are non-zero");
        }
    }
    if (result.dimension == 0U || result.pq_m == 0U ||
        result.pq_nbits == 0U || result.pq_nbits > 8U ||
        result.pq_ksub != (1U << result.pq_nbits) ||
        result.pq_dsub == 0U ||
        result.dimension != result.pq_m * result.pq_dsub ||
        result.code_size != result.pq_m) {
        throw std::runtime_error("V0PQ configuration is inconsistent");
    }
    const uint64_t expected_centroids =
        checkedMultiply(
            checkedMultiply(result.pq_m, result.pq_ksub),
            result.pq_dsub);
    if (result.centroid_count != expected_centroids ||
        centroids_size != checkedMultiply(expected_centroids, 4U)) {
        throw std::runtime_error("V0PQ centroid section size is invalid");
    }
    uint64_t expected_offset = EDGE_QUANT_V0_PQ_HEADER_SIZE;
    if (metadata_offset != expected_offset) {
        throw std::runtime_error("V0PQ metadata offset is not canonical");
    }
    expected_offset = checkedAdd(expected_offset, metadata_size);
    if (centroids_offset != expected_offset) {
        throw std::runtime_error("V0PQ centroid offset is not canonical");
    }
    expected_offset = checkedAdd(expected_offset, centroids_size);
    if (checksums_offset != expected_offset ||
        checksums_size != EDGE_QUANT_V0_PQ_CHECKSUM_SIZE) {
        throw std::runtime_error("V0PQ checksum section is not canonical");
    }
    expected_offset = checkedAdd(expected_offset, checksums_size);
    if (expected_offset != bytes.size()) {
        throw std::runtime_error(
            "V0PQ file size does not match its canonical layout");
    }

    const uint8_t* checksum = bytes.data() + checksums_offset;
    if (!std::equal(
            CHECKSUM_MAGIC,
            CHECKSUM_MAGIC + 8U,
            checksum) ||
        readUint32(checksum + 8U) !=
            EDGE_QUANT_V0_PQ_CODEBOOK_VERSION ||
        readUint32(checksum + 12U) != 3U) {
        throw std::runtime_error("V0PQ checksum section is invalid");
    }
    const std::array<uint8_t, 32> header_sha =
        sha256(bytes.data(), EDGE_QUANT_V0_PQ_HEADER_SIZE);
    const std::array<uint8_t, 32> metadata_sha =
        sha256(
            bytes.data() + metadata_offset,
            static_cast<size_t>(metadata_size));
    const std::array<uint8_t, 32> centroids_sha =
        sha256(
            bytes.data() + centroids_offset,
            static_cast<size_t>(centroids_size));
    requireDigest(checksum + 16U, header_sha, "header");
    requireDigest(checksum + 48U, metadata_sha, "training metadata");
    requireDigest(checksum + 80U, centroids_sha, "centroids");
    if (centroids_sha != result.centroids_sha256) {
        throw std::runtime_error(
            "V0PQ stored centroid fingerprint does not match");
    }

    result.training_metadata_json.assign(
        reinterpret_cast<const char*>(bytes.data() + metadata_offset),
        static_cast<size_t>(metadata_size));
    result.centroids.resize(
        static_cast<size_t>(result.centroid_count));
    for (size_t i = 0U; i < result.centroids.size(); ++i) {
        result.centroids[i] =
            readFloat32(bytes.data() + centroids_offset + i * 4U);
        if (!std::isfinite(static_cast<double>(result.centroids[i]))) {
            throw std::runtime_error(
                "V0PQ contains a non-finite centroid");
        }
    }
    result.file_sha256 = sha256(bytes.data(), bytes.size());
    return result;
}

}  // namespace hnswlib
