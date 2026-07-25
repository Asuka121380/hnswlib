#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hnswlib/edge_quant_v0_encoder.h"

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void writeUint32(char* destination, uint32_t value) {
    std::memcpy(destination, &value, sizeof(value));
}

void writeFloat(char* destination, float value) {
    std::memcpy(destination, &value, sizeof(value));
}

hnswlib::V0Layer0GraphView makeGraph(
    std::vector<char>& storage) {
    const size_t nodes = 3U;
    const size_t max_degree = 2U;
    const size_t data_offset =
        sizeof(hnswlib::linklistsizeint) +
        max_degree * sizeof(hnswlib::tableint);
    const size_t data_size = 4U * sizeof(float);
    const size_t label_offset = data_offset + data_size;
    const size_t stride =
        label_offset + sizeof(hnswlib::labeltype);
    storage.assign(nodes * stride, 0);

    char* node0 = storage.data();
    writeUint32(node0, 2U);
    writeUint32(node0 + 4U, 1U);
    writeUint32(node0 + 8U, 0U);

    char* node1 = storage.data() + stride;
    writeUint32(node1, 1U);
    writeUint32(node1 + 4U, 0U);
    writeFloat(node1 + data_offset, 1.0f);

    char* node2 = storage.data() + 2U * stride;
    writeUint32(node2, 0U);
    writeFloat(node2 + data_offset + 4U, 2.0f);

    return hnswlib::V0Layer0GraphView(
        storage.data(),
        nodes,
        stride,
        data_offset,
        data_size,
        label_offset,
        max_degree);
}

class SyntheticCodec {
 public:
    uint32_t dimension() const {
        return 4U;
    }

    uint32_t codeSize() const {
        return 2U;
    }

    void encode(
        const float* vectors,
        size_t count,
        uint8_t* codes) const {
        for (size_t row = 0U; row < count; ++row) {
            const float first = vectors[row * 4U];
            codes[row * 2U] =
                first > 0.0f ? 1U : (first < 0.0f ? 2U : 0U);
            codes[row * 2U + 1U] = 0U;
        }
    }

    void decode(
        const uint8_t* codes,
        size_t count,
        float* vectors) const {
        for (size_t row = 0U; row < count; ++row) {
            vectors[row * 4U] =
                codes[row * 2U] == 1U
                    ? 1.0f
                    : (codes[row * 2U] == 2U ? -1.0f : 0.0f);
            vectors[row * 4U + 1U] = 0.0f;
            vectors[row * 4U + 2U] = 0.0f;
            vectors[row * 4U + 3U] = 0.0f;
        }
    }
};

hnswlib::V0Sha256Digest digest(const std::string& text) {
    hnswlib::EdgeQuantV0Sha256 sha;
    sha.update(text.data(), text.size());
    return sha.final();
}

void testNumericContract() {
    const float source[] = {1.0f, 2.0f, 0.0f, 0.0f};
    const float target[] = {4.0f, 6.0f, 0.0f, 0.0f};
    float direction[4] = {};
    const hnswlib::V0EdgeDirectionInfo prepared =
        hnswlib::fillV0UnitEdgeDirection(
            source, target, 4U, direction);
    require(!prepared.zero_length, "non-zero edge classified as zero");
    require(
        std::fabs(prepared.edge_length - 5.0) < 1e-15,
        "edge length mismatch");
    require(
        std::fabs(direction[0] - 0.6f) < 1e-7f &&
        std::fabs(direction[1] - 0.8f) < 1e-7f,
        "float32 unit direction mismatch");
    const hnswlib::V0EdgeNumericMetadata metadata =
        hnswlib::computeV0EdgeNumericMetadata(
            source, target, direction, 4U);
    double squared_error = 0.0;
    const double inverse_length = 1.0 / prepared.edge_length;
    for (size_t d = 0U; d < 4U; ++d) {
        const double edge_component =
            static_cast<double>(target[d]) -
            static_cast<double>(source[d]);
        const double exact_direction =
            edge_component * inverse_length;
        const double difference =
            exact_direction - static_cast<double>(direction[d]);
        squared_error += difference * difference;
    }
    const double exact_error = std::sqrt(squared_error);
    require(
        metadata.direction_error >= exact_error,
        "stored direction error is not conservative");
    require(
        metadata.numeric_padding == 0.0,
        "Milestone 5 numeric padding must be zero");
}

}  // namespace

int main() {
#ifndef HNSWLIB_ENABLE_EDGE_QUANT_V0
    std::cerr << "HNSWLIB_ENABLE_EDGE_QUANT_V0 is required" << std::endl;
    return 2;
#else
    testNumericContract();
    std::vector<char> storage;
    const hnswlib::V0Layer0GraphView graph = makeGraph(storage);
    const std::string path = "v0_all_edge_encoder_test.v0meta";
    std::remove(path.c_str());
    std::remove((path + ".partial").c_str());
    try {
        hnswlib::V0AllEdgeEncodingSpec spec;
        spec.dimension = 4U;
        spec.pq_m = 2U;
        spec.pq_nbits = 8U;
        spec.pq_ksub = 256U;
        spec.pq_dsub = 2U;
        spec.block_size = 2U;
        spec.output_sidecar = path;
        spec.training_metadata_json =
            "{\"quantizer\":\"synthetic\"}";
        spec.codebook_centroids.assign(1024U, 0.0f);
        spec.codebook_centroids[(0U * 256U + 1U) * 2U] = 1.0f;
        spec.codebook_centroids[(0U * 256U + 2U) * 2U] = -1.0f;
        spec.base_index_sha256 = digest("index");
        spec.adjacency_sha256 = graph.adjacencyFingerprint();

        SyntheticCodec codec;
        const hnswlib::V0AllEdgeEncodingMetrics metrics =
            hnswlib::encodeV0AllLayer0Edges(graph, spec, codec);
        require(metrics.node_count == 3U, "node count mismatch");
        require(
            metrics.directed_edge_count == 3U &&
            metrics.encoded_edge_count == 2U &&
            metrics.zero_length_edge_count == 1U,
            "edge accounting mismatch");
        require(metrics.block_count == 2U, "block count mismatch");

        const hnswlib::V0OwnedSidecar sidecar =
            hnswlib::loadV0Sidecar(path);
        require(
            sidecar.view().nodeOffset(0U) == 0U &&
            sidecar.view().nodeOffset(1U) == 2U &&
            sidecar.view().nodeOffset(2U) == 3U &&
            sidecar.view().nodeOffset(3U) == 3U,
            "node offsets do not preserve adjacency slots");
        const hnswlib::V0EdgeRecordView forward =
            sidecar.view().edgeRecord(0U);
        const hnswlib::V0EdgeRecordView zero =
            sidecar.view().edgeRecord(1U);
        const hnswlib::V0EdgeRecordView reverse =
            sidecar.view().edgeRecord(2U);
        require(
            forward.code(0U) == 1U &&
            reverse.code(0U) == 2U,
            "encoded edge code mismatch");
        require(
            zero.code(0U) == 0U &&
            zero.code(1U) == 0U &&
            zero.flags() ==
                (hnswlib::V0_EDGE_EXACT_ONLY |
                 hnswlib::V0_ZERO_LENGTH_EDGE) &&
            zero.edgeLength() == 0.0 &&
            zero.directionError() == 0.0 &&
            zero.anchorProjection() == 0.0,
            "zero-length record contract mismatch");
        require(
            forward.directionError() > 0.0 &&
            reverse.directionError() > 0.0,
            "perfect reconstruction error was not rounded upward");
        std::remove(path.c_str());
        std::cout << "v0_all_edge_encoder_test_ok" << std::endl;
        return 0;
    } catch (...) {
        std::remove(path.c_str());
        std::remove((path + ".partial").c_str());
        throw;
    }
#endif
}
