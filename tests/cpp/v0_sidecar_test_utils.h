#pragma once

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hnswlib/hnswlib.h"

namespace v0_test {

inline void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template<typename Callable>
inline void requireThrows(Callable callable, const std::string& message) {
    try {
        callable();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

inline hnswlib::V0Sha256Digest digestOf(const std::string& value) {
    hnswlib::EdgeQuantV0Sha256 sha;
    if (!value.empty()) {
        sha.update(value.data(), value.size());
    }
    return sha.final();
}

inline hnswlib::V0SidecarWriteSpec tinySpec() {
    hnswlib::V0SidecarWriteSpec spec;
    spec.dimension = 4;
    spec.pq_m = 2;
    spec.pq_nbits = 1;
    spec.pq_ksub = 2;
    spec.pq_dsub = 2;
    spec.node_count = 3;
    spec.directed_edge_count = 3;
    spec.training_metadata_json =
        "{\"quantizer\":\"synthetic\",\"seed\":17}";
    const float centroid_values[] = {
        0.0f, 0.5f,
        1.0f, 1.5f,
        -1.0f, -0.5f,
        2.0f, 2.5f
    };
    spec.codebook_centroids.assign(
        centroid_values,
        centroid_values + sizeof(centroid_values) / sizeof(float));
    const uint64_t offsets[] = {0U, 2U, 2U, 3U};
    spec.node_offsets.assign(offsets, offsets + 4U);
    spec.base_index_sha256 = digestOf("synthetic-base-index");
    spec.adjacency_sha256 = digestOf("synthetic-adjacency");
    return spec;
}

inline std::vector<hnswlib::V0EdgeRecord> tinyRecords() {
    std::vector<hnswlib::V0EdgeRecord> records(3);
    records[0].code.push_back(0U);
    records[0].code.push_back(1U);
    records[0].edge_length = 1.25;
    records[0].direction_error = 0.125;
    records[0].anchor_projection = -2.5;

    records[1].code.push_back(1U);
    records[1].code.push_back(0U);
    records[1].flags = hnswlib::V0_EDGE_EXACT_ONLY;
    records[1].edge_length = 2.5;
    records[1].direction_error = 0.25;
    records[1].anchor_projection = 4.0;
    records[1].numeric_padding = 0.001;

    records[2].code.push_back(0U);
    records[2].code.push_back(0U);
    records[2].flags =
        hnswlib::V0_EDGE_EXACT_ONLY |
        hnswlib::V0_ZERO_LENGTH_EDGE;
    records[2].edge_length = 0.0;
    records[2].direction_error = 0.0;
    records[2].anchor_projection = 0.0;
    return records;
}

inline void writeTinySidecar(const std::string& path) {
    const hnswlib::V0SidecarWriteSpec spec = tinySpec();
    const std::vector<hnswlib::V0EdgeRecord> records = tinyRecords();
    hnswlib::V0SidecarWriter writer(path, spec);
    for (size_t i = 0; i < records.size(); ++i) {
        writer.writeEdgeRecord(records[i]);
    }
    writer.finalize();
}

inline std::vector<uint8_t> readFile(const std::string& path) {
    std::ifstream input(path.c_str(), std::ios::binary | std::ios::ate);
    require(static_cast<bool>(input), "failed to open test file");
    const std::streamoff size = input.tellg();
    require(size >= 0, "failed to determine test file size");
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    input.seekg(0, std::ios::beg);
    if (!bytes.empty()) {
        input.read(
            reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    }
    require(
        static_cast<bool>(input) || bytes.empty(),
        "failed to read test file");
    return bytes;
}

inline void writeFile(
    const std::string& path,
    const std::vector<uint8_t>& bytes) {
    std::ofstream output(
        path.c_str(), std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "failed to open test output");
    if (!bytes.empty()) {
        output.write(
            reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    }
    require(static_cast<bool>(output), "failed to write test output");
}

class FileCleanup {
 public:
    explicit FileCleanup(const std::string& path) : path_(path) {
        std::remove(path_.c_str());
    }

    ~FileCleanup() {
        std::remove(path_.c_str());
    }

 private:
    std::string path_;
};

}  // namespace v0_test
