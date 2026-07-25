#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "hnswlib/hnswlib.h"

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

void writeUint32(char* destination, uint32_t value) {
    std::memcpy(destination, &value, sizeof(value));
}

void writeFloat(char* destination, float value) {
    std::memcpy(destination, &value, sizeof(value));
}

hnswlib::V0Layer0GraphView makeSyntheticGraph(
    std::vector<char>& storage) {
    const size_t node_count = 3U;
    const size_t max_degree = 2U;
    const size_t data_offset =
        sizeof(hnswlib::linklistsizeint) +
        max_degree * sizeof(hnswlib::tableint);
    const size_t data_size = 2U * sizeof(float);
    const size_t label_offset = data_offset + data_size;
    const size_t stride = label_offset + sizeof(hnswlib::labeltype);
    storage.assign(node_count * stride, 0);

    char* node0 = storage.data();
    writeUint32(node0, 2U);
    writeUint32(node0 + 4U, 1U);
    writeUint32(node0 + 8U, 2U);
    writeFloat(node0 + data_offset, 0.0f);
    writeFloat(node0 + data_offset + 4U, 0.0f);

    char* node1 = storage.data() + stride;
    writeUint32(node1, 1U);
    writeUint32(node1 + 4U, 0U);
    writeFloat(node1 + data_offset, 0.0f);
    writeFloat(node1 + data_offset + 4U, 0.0f);

    char* node2 = storage.data() + 2U * stride;
    writeUint32(node2, 1U);
    writeUint32(node2 + 4U, 0U);
    writeFloat(node2 + data_offset, 3.0f);
    writeFloat(node2 + data_offset + 4U, 4.0f);

    return hnswlib::V0Layer0GraphView(
        storage.data(),
        node_count,
        stride,
        data_offset,
        data_size,
        label_offset,
        max_degree);
}

void testSplitMix64GoldenVector() {
    hnswlib::V0SplitMix64 random(0U);
    require(
        random.next() == 0xe220a8397b1dcdafULL,
        "SplitMix64 golden output 0 mismatch");
    require(
        random.next() == 0x6e789e6aa1b965f4ULL,
        "SplitMix64 golden output 1 mismatch");
    require(
        random.next() == 0x06c45d188009454fULL,
        "SplitMix64 golden output 2 mismatch");
}

void testSamplingAndZeroLengthEdges() {
    std::vector<char> storage;
    hnswlib::V0Layer0GraphView graph = makeSyntheticGraph(storage);
    const hnswlib::V0EdgeDirectionSample sample =
        hnswlib::sampleV0Layer0EdgeDirections(graph, 2U, 10U, 17U);
    require(sample.directed_edge_count == 4U, "directed edge count mismatch");
    require(sample.valid_edge_count == 2U, "valid edge count mismatch");
    require(
        sample.zero_length_edge_count == 2U,
        "zero-length edge count mismatch");
    require(sample.sampleCount() == 2U, "produced sample count mismatch");
    require(sample.directions.size() == 4U, "direction matrix size mismatch");

    require(
        std::fabs(sample.directions[0] - 0.6f) < 1e-6f &&
        std::fabs(sample.directions[1] - 0.8f) < 1e-6f,
        "forward unit direction mismatch");
    require(
        std::fabs(sample.directions[2] + 0.6f) < 1e-6f &&
        std::fabs(sample.directions[3] + 0.8f) < 1e-6f,
        "reverse unit direction mismatch");
    for (size_t row = 0; row < sample.sampleCount(); ++row) {
        double squared_norm = 0.0;
        for (size_t d = 0; d < sample.dimension; ++d) {
            const double value =
                sample.directions[row * sample.dimension + d];
            require(std::isfinite(value), "sample direction is non-finite");
            squared_norm += value * value;
        }
        require(
            std::fabs(std::sqrt(squared_norm) - 1.0) < 1e-6,
            "sample direction is not unit norm");
    }

    const hnswlib::V0EdgeDirectionSample first =
        hnswlib::sampleV0Layer0EdgeDirections(graph, 2U, 1U, 99U);
    const hnswlib::V0EdgeDirectionSample second =
        hnswlib::sampleV0Layer0EdgeDirections(graph, 2U, 1U, 99U);
    require(
        first.directions == second.directions,
        "same seed did not reproduce the same reservoir");
    const hnswlib::V0EdgeDirectionSample seed_zero =
        hnswlib::sampleV0Layer0EdgeDirections(graph, 2U, 1U, 0U);
    const hnswlib::V0EdgeDirectionSample seed_two =
        hnswlib::sampleV0Layer0EdgeDirections(graph, 2U, 1U, 2U);
    require(
        seed_zero.directions != seed_two.directions,
        "different golden seeds did not change the reservoir");

    const hnswlib::V0EdgeDirectionSample count_only =
        hnswlib::sampleV0Layer0EdgeDirections(graph, 2U, 0U, 99U);
    require(
        count_only.sampleCount() == 0U &&
        count_only.valid_edge_count == 2U &&
        count_only.zero_length_edge_count == 2U,
        "zero-capacity reservoir accounting mismatch");

    requireThrows(
        [&graph]() {
            (void)hnswlib::sampleV0Layer0EdgeDirections(
                graph, 3U, 1U, 0U);
        },
        "dimension mismatch was accepted");

    writeFloat(storage.data() + 12U, std::numeric_limits<float>::quiet_NaN());
    requireThrows(
        [&graph]() {
            (void)hnswlib::sampleV0Layer0EdgeDirections(
                graph, 2U, 1U, 0U);
        },
        "non-finite vector was accepted");
}

void testDirectionMatrixFile() {
    std::vector<char> storage;
    const hnswlib::V0Layer0GraphView graph = makeSyntheticGraph(storage);
    const hnswlib::V0EdgeDirectionSample sample =
        hnswlib::sampleV0Layer0EdgeDirections(graph, 2U, 10U, 17U);
    const std::string path = "v0_edge_sampler_test.f32";
    std::remove(path.c_str());
    std::remove((path + ".partial").c_str());

    const hnswlib::V0DirectionMatrixWriteResult written =
        hnswlib::writeV0DirectionMatrix(path, sample);
    require(written.byte_count == 16U, "direction byte count mismatch");
    require(
        written.sha256 == hnswlib::computeV0FileSha256(path),
        "direction file SHA-256 mismatch");

    std::ifstream input(path.c_str(), std::ios::binary);
    std::vector<uint8_t> bytes(16U);
    input.read(
        reinterpret_cast<char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    require(static_cast<bool>(input), "failed to read direction matrix");
    input.close();
    require(
        std::fabs(
            hnswlib::edge_quant_v0_detail::readFloat32LittleEndian(
                bytes.data()) -
            0.6f) < 1e-6f,
        "direction matrix is not little-endian float32");
    requireThrows(
        [&path, &sample]() {
            (void)hnswlib::writeV0DirectionMatrix(path, sample);
        },
        "direction writer overwrote an existing file");
    require(
        std::remove(path.c_str()) == 0,
        "failed to remove direction matrix test output");
}

void writeTestIndex(const std::string& path) {
    const size_t dimension = 8U;
    const size_t node_count = 64U;
    std::vector<float> base(node_count * dimension);
    for (size_t node = 0; node < node_count; ++node) {
        for (size_t d = 0; d < dimension; ++d) {
            base[node * dimension + d] =
                static_cast<float>((node * 17U + d * 11U) % 101U) +
                static_cast<float>(node) * 0.0001f;
        }
    }
    hnswlib::L2Space space(dimension);
    hnswlib::HierarchicalNSW<float> index(
        &space, node_count, 12, 80, 41);
    for (size_t node = 0; node < node_count; ++node) {
        index.addPoint(base.data() + node * dimension, node);
    }
    index.saveIndex(path);
}

}  // namespace

int main(int argc, char** argv) {
#ifndef HNSWLIB_ENABLE_EDGE_QUANT_V0
    std::cerr << "HNSWLIB_ENABLE_EDGE_QUANT_V0 is required" << std::endl;
    return 2;
#else
    if (argc == 3 && std::string(argv[1]) == "--write-index") {
        writeTestIndex(argv[2]);
        std::cout << "v0_edge_sampler_test_index_ok" << std::endl;
        return 0;
    }
    testSplitMix64GoldenVector();
    testSamplingAndZeroLengthEdges();
    testDirectionMatrixFile();
    std::cout << "v0_edge_sampler_test_ok" << std::endl;
    return 0;
#endif
}
