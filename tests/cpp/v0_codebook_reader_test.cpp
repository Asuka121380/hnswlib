#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hnswlib/edge_quant_v0_codebook.h"

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template<typename Callable>
void requireThrows(Callable callable, const std::string& message) {
    try {
        callable();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

void appendUint32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (size_t i = 0U; i < 4U; ++i) {
        bytes.push_back(
            static_cast<uint8_t>((value >> (8U * i)) & 0xffU));
    }
}

void appendUint64(std::vector<uint8_t>& bytes, uint64_t value) {
    for (size_t i = 0U; i < 8U; ++i) {
        bytes.push_back(
            static_cast<uint8_t>((value >> (8U * i)) & 0xffU));
    }
}

void appendFloat32(std::vector<uint8_t>& bytes, float value) {
    uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    appendUint32(bytes, bits);
}

std::array<uint8_t, 32> sha(
    const uint8_t* bytes,
    size_t size) {
    hnswlib::EdgeQuantV0Sha256 digest;
    if (size != 0U) {
        digest.update(bytes, size);
    }
    return digest.final();
}

void appendDigest(
    std::vector<uint8_t>& bytes,
    const std::array<uint8_t, 32>& digest) {
    bytes.insert(bytes.end(), digest.begin(), digest.end());
}

std::vector<uint8_t> makeCodebook() {
    const uint32_t dimension = 4U;
    const uint32_t pq_m = 2U;
    const uint32_t nbits = 8U;
    const uint32_t ksub = 256U;
    const uint32_t dsub = 2U;
    const uint64_t centroid_count =
        static_cast<uint64_t>(pq_m) * ksub * dsub;
    const std::string metadata =
        "{\"quantizer_name\":\"synthetic\"}";
    std::vector<uint8_t> centroids;
    for (uint64_t i = 0U; i < centroid_count; ++i) {
        appendFloat32(
            centroids,
            static_cast<float>(i) / 1024.0f);
    }
    const std::array<uint8_t, 32> metadata_source =
        sha(reinterpret_cast<const uint8_t*>("manifest"), 8U);
    const std::array<uint8_t, 32> direction_source =
        sha(reinterpret_cast<const uint8_t*>("directions"), 10U);
    const std::array<uint8_t, 32> centroid_sha =
        sha(centroids.data(), centroids.size());

    const uint64_t metadata_offset = 256U;
    const uint64_t centroids_offset =
        metadata_offset + metadata.size();
    const uint64_t checksums_offset =
        centroids_offset + centroids.size();

    std::vector<uint8_t> header;
    const uint8_t magic[] = {
        'V', '0', 'P', 'Q', 'B', 'O', 'O', 'K'
    };
    header.insert(header.end(), magic, magic + 8U);
    appendUint32(header, 1U);
    appendUint32(header, 256U);
    const uint8_t representation[] = {1U, 1U, 1U, 0U};
    header.insert(
        header.end(), representation, representation + 4U);
    appendUint32(header, dimension);
    appendUint32(header, pq_m);
    appendUint32(header, nbits);
    appendUint32(header, ksub);
    appendUint32(header, dsub);
    appendUint32(header, pq_m);
    appendUint64(header, centroid_count);
    appendUint64(header, metadata_offset);
    appendUint64(header, metadata.size());
    appendUint64(header, centroids_offset);
    appendUint64(header, centroids.size());
    appendUint64(header, checksums_offset);
    appendUint64(header, 112U);
    appendDigest(header, metadata_source);
    appendDigest(header, direction_source);
    appendDigest(header, centroid_sha);
    header.resize(256U, 0U);

    const std::array<uint8_t, 32> header_sha =
        sha(header.data(), header.size());
    const std::array<uint8_t, 32> metadata_sha =
        sha(
            reinterpret_cast<const uint8_t*>(metadata.data()),
            metadata.size());
    std::vector<uint8_t> checksums;
    const uint8_t checksum_magic[] = {
        'V', '0', 'P', 'Q', 'S', 'U', 'M', '1'
    };
    checksums.insert(
        checksums.end(), checksum_magic, checksum_magic + 8U);
    appendUint32(checksums, 1U);
    appendUint32(checksums, 3U);
    appendDigest(checksums, header_sha);
    appendDigest(checksums, metadata_sha);
    appendDigest(checksums, centroid_sha);

    std::vector<uint8_t> result = header;
    result.insert(result.end(), metadata.begin(), metadata.end());
    result.insert(result.end(), centroids.begin(), centroids.end());
    result.insert(
        result.end(), checksums.begin(), checksums.end());
    return result;
}

void writeFile(
    const std::string& path,
    const std::vector<uint8_t>& bytes) {
    std::ofstream output(
        path.c_str(), std::ios::binary | std::ios::trunc);
    output.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    require(static_cast<bool>(output), "failed to write V0PQ fixture");
}

}  // namespace

int main() {
    const std::string valid_path =
        "v0_codebook_reader_test.v0pq";
    const std::string corrupt_path =
        "v0_codebook_reader_test_corrupt.v0pq";
    std::remove(valid_path.c_str());
    std::remove(corrupt_path.c_str());
    try {
        const std::vector<uint8_t> bytes = makeCodebook();
        writeFile(valid_path, bytes);
        const hnswlib::V0PQCodebook codebook =
            hnswlib::loadV0PQCodebook(valid_path);
        require(
            codebook.dimension == 4U &&
            codebook.pq_m == 2U &&
            codebook.pq_nbits == 8U &&
            codebook.pq_ksub == 256U &&
            codebook.pq_dsub == 2U,
            "V0PQ configuration round-trip mismatch");
        require(
            codebook.centroids.size() == 1024U,
            "V0PQ centroid count mismatch");
        require(
            std::fabs(codebook.centroids[17] - 17.0f / 1024.0f) <
                1e-8f,
            "V0PQ centroid value mismatch");
        require(
            codebook.training_metadata_json ==
                "{\"quantizer_name\":\"synthetic\"}",
            "V0PQ metadata mismatch");

        std::vector<uint8_t> corrupt = bytes;
        corrupt[300U] ^= 1U;
        writeFile(corrupt_path, corrupt);
        requireThrows(
            [&corrupt_path]() {
                (void)hnswlib::loadV0PQCodebook(corrupt_path);
            },
            "corrupt V0PQ codebook was accepted");
        std::remove(valid_path.c_str());
        std::remove(corrupt_path.c_str());
        std::cout << "v0_codebook_reader_test_ok" << std::endl;
        return 0;
    } catch (...) {
        std::remove(valid_path.c_str());
        std::remove(corrupt_path.c_str());
        throw;
    }
}
