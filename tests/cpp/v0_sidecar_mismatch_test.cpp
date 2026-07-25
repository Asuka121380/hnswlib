#include <algorithm>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "v0_sidecar_test_utils.h"

namespace {

void writeUint64LittleEndian(
    std::vector<uint8_t>& bytes,
    size_t offset,
    uint64_t value) {
    for (size_t i = 0; i < 8U; ++i) {
        bytes[offset + i] =
            static_cast<uint8_t>((value >> (i * 8U)) & 0xffU);
    }
}

void replaceDigest(
    std::vector<uint8_t>& bytes,
    size_t offset,
    const hnswlib::V0Sha256Digest& digest) {
    std::copy(digest.begin(), digest.end(), bytes.begin() + offset);
}

void testMalformedFiles() {
    const std::string valid_path = "v0_sidecar_mismatch_valid.v0meta";
    const std::string corrupt_path = "v0_sidecar_mismatch_corrupt.v0meta";
    v0_test::FileCleanup valid_cleanup(valid_path);
    v0_test::FileCleanup corrupt_cleanup(corrupt_path);
    v0_test::writeTinySidecar(valid_path);
    const std::vector<uint8_t> valid = v0_test::readFile(valid_path);

    std::vector<uint8_t> corrupt = valid;
    corrupt[0] ^= 0xffU;
    v0_test::requireThrows(
        [&corrupt]() {
            (void)hnswlib::parseV0Sidecar(
                corrupt.data(), corrupt.size());
        },
        "invalid sidecar magic was accepted");

    corrupt = valid;
    corrupt.resize(corrupt.size() - 1U);
    v0_test::requireThrows(
        [&corrupt]() {
            (void)hnswlib::parseV0Sidecar(
                corrupt.data(), corrupt.size());
        },
        "truncated sidecar was accepted");

    corrupt = valid;
    corrupt[80U] ^= 1U;
    v0_test::requireThrows(
        [&corrupt]() {
            (void)hnswlib::parseV0Sidecar(
                corrupt.data(), corrupt.size());
        },
        "non-canonical codebook offset was accepted");

    corrupt = valid;
    const uint64_t training_offset =
        hnswlib::edge_quant_v0_detail::readUint64LittleEndian(
            corrupt.data() + 64U);
    corrupt[static_cast<size_t>(training_offset)] ^= 1U;
    v0_test::requireThrows(
        [&corrupt]() {
            (void)hnswlib::parseV0Sidecar(
                corrupt.data(), corrupt.size());
        },
        "training-metadata checksum mismatch was accepted");

    corrupt = valid;
    corrupt[240U] = 1U;
    v0_test::requireThrows(
        [&corrupt]() {
            (void)hnswlib::parseV0Sidecar(
                corrupt.data(), corrupt.size());
        },
        "non-zero reserved header byte was accepted");

    corrupt = valid;
    const uint64_t node_offsets_offset =
        hnswlib::edge_quant_v0_detail::readUint64LittleEndian(
            corrupt.data() + 96U);
    const uint64_t node_offsets_size =
        hnswlib::edge_quant_v0_detail::readUint64LittleEndian(
            corrupt.data() + 104U);
    const uint64_t checksums_offset =
        hnswlib::edge_quant_v0_detail::readUint64LittleEndian(
            corrupt.data() + 128U);
    writeUint64LittleEndian(
        corrupt,
        static_cast<size_t>(node_offsets_offset + 16U),
        1U);
    replaceDigest(
        corrupt,
        static_cast<size_t>(checksums_offset + 112U),
        hnswlib::edge_quant_v0_detail::sha256(
            corrupt.data() + node_offsets_offset,
            static_cast<size_t>(node_offsets_size)));
    v0_test::requireThrows(
        [&corrupt]() {
            (void)hnswlib::parseV0Sidecar(
                corrupt.data(), corrupt.size());
        },
        "non-monotonic node offsets with valid checksum were accepted");

    corrupt = valid;
    const uint64_t edge_records_offset =
        hnswlib::edge_quant_v0_detail::readUint64LittleEndian(
            corrupt.data() + 112U);
    const uint64_t edge_records_size =
        hnswlib::edge_quant_v0_detail::readUint64LittleEndian(
            corrupt.data() + 120U);
    const uint32_t code_size =
        hnswlib::edge_quant_v0_detail::readUint32LittleEndian(
            corrupt.data() + 40U);
    const uint32_t scalar_offset =
        hnswlib::edge_quant_v0_detail::edgeRecordScalarOffset(code_size);
    writeUint64LittleEndian(
        corrupt,
        static_cast<size_t>(edge_records_offset + scalar_offset),
        0x7ff8000000000000ULL);
    replaceDigest(
        corrupt,
        static_cast<size_t>(checksums_offset + 144U),
        hnswlib::edge_quant_v0_detail::sha256(
            corrupt.data() + edge_records_offset,
            static_cast<size_t>(edge_records_size)));
    v0_test::requireThrows(
        [&corrupt]() {
            (void)hnswlib::parseV0Sidecar(
                corrupt.data(), corrupt.size());
        },
        "NaN edge metadata with valid checksum was accepted");

    v0_test::writeFile(corrupt_path, corrupt);
    v0_test::requireThrows(
        [&corrupt_path]() {
            (void)hnswlib::loadV0Sidecar(corrupt_path);
        },
        "file loader accepted a malformed sidecar");
}

void testWriterValidation() {
    const std::string path = "v0_sidecar_invalid_writer.v0meta";
    v0_test::FileCleanup cleanup(path);

    hnswlib::V0SidecarWriteSpec invalid = v0_test::tinySpec();
    invalid.node_offsets[2] = 1U;
    v0_test::requireThrows(
        [&path, &invalid]() {
            hnswlib::V0SidecarWriter writer(path, invalid);
        },
        "writer accepted non-monotonic node offsets");

    invalid = v0_test::tinySpec();
    invalid.codebook_centroids.pop_back();
    v0_test::requireThrows(
        [&path, &invalid]() {
            hnswlib::V0SidecarWriter writer(path, invalid);
        },
        "writer accepted wrong codebook size");

    invalid = v0_test::tinySpec();
    {
        hnswlib::V0SidecarWriter writer(path, invalid);
        const std::vector<hnswlib::V0EdgeRecord> records =
            v0_test::tinyRecords();
        writer.writeEdgeRecord(records[0]);
        v0_test::requireThrows(
            [&writer]() { writer.finalize(); },
            "writer finalized an incomplete edge stream");
    }

    const std::vector<hnswlib::V0EdgeRecord> records =
        v0_test::tinyRecords();
    {
        hnswlib::V0SidecarWriter writer(path, v0_test::tinySpec());
        hnswlib::V0EdgeRecord bad = records[0];
        bad.edge_length = std::numeric_limits<double>::quiet_NaN();
        v0_test::requireThrows(
            [&writer, &bad]() { writer.writeEdgeRecord(bad); },
            "writer accepted NaN edge metadata");
    }
}

void testCompatibilityMismatch() {
    const std::string path = "v0_sidecar_compatibility.v0meta";
    v0_test::FileCleanup cleanup(path);
    const hnswlib::V0SidecarWriteSpec spec = v0_test::tinySpec();
    v0_test::writeTinySidecar(path);
    const hnswlib::V0OwnedSidecar sidecar =
        hnswlib::loadV0Sidecar(path);
    const std::vector<uint8_t> sidecar_bytes = v0_test::readFile(path);
    v0_test::require(
        hnswlib::computeV0FileSha256(path) ==
            hnswlib::edge_quant_v0_detail::sha256(
                sidecar_bytes.data(),
                sidecar_bytes.size()),
        "streaming file SHA-256 mismatch");

    hnswlib::V0IndexCompatibility expected;
    expected.dimension = spec.dimension;
    expected.node_count = spec.node_count;
    expected.directed_edge_count = spec.directed_edge_count;
    expected.base_index_sha256 = spec.base_index_sha256;
    expected.adjacency_sha256 = spec.adjacency_sha256;
    hnswlib::validateV0SidecarCompatibility(
        sidecar.header(), expected);

    hnswlib::V0IndexCompatibility mismatch = expected;
    mismatch.dimension += 1U;
    v0_test::requireThrows(
        [&sidecar, &mismatch]() {
            hnswlib::validateV0SidecarCompatibility(
                sidecar.header(), mismatch);
        },
        "dimension mismatch was accepted");

    mismatch = expected;
    mismatch.adjacency_sha256[0] ^= 1U;
    v0_test::requireThrows(
        [&sidecar, &mismatch]() {
            hnswlib::validateV0SidecarCompatibility(
                sidecar.header(), mismatch);
        },
        "adjacency fingerprint mismatch was accepted");

    mismatch = expected;
    mismatch.base_index_sha256[0] ^= 1U;
    v0_test::requireThrows(
        [&sidecar, &mismatch]() {
            hnswlib::validateV0SidecarCompatibility(
                sidecar.header(), mismatch);
        },
        "base-index fingerprint mismatch was accepted");
}

void testGraphSlotCompatibility() {
    const size_t dimension = 4;
    const size_t node_count = 24;
    std::vector<float> base(node_count * dimension);
    for (size_t i = 0; i < base.size(); ++i) {
        base[i] = static_cast<float>((i * 13U) % 29U);
    }

    hnswlib::L2Space space(dimension);
    hnswlib::HierarchicalNSW<float> index(
        &space, node_count + 1U, 8, 50, 31);
    for (size_t node = 0; node < node_count; ++node) {
        index.addPoint(base.data() + node * dimension, node);
    }
    const hnswlib::V0Layer0GraphView graph =
        index.getV0Layer0GraphView();

    hnswlib::V0SidecarWriteSpec spec;
    spec.dimension = dimension;
    spec.pq_m = 2;
    spec.pq_nbits = 1;
    spec.pq_ksub = 2;
    spec.pq_dsub = 2;
    spec.node_count = node_count;
    spec.training_metadata_json = "{}";
    spec.codebook_centroids.assign(8U, 0.0f);
    spec.node_offsets.push_back(0U);
    for (size_t node = 0; node < node_count; ++node) {
        spec.directed_edge_count +=
            graph.neighbors(static_cast<hnswlib::tableint>(node)).size;
        spec.node_offsets.push_back(spec.directed_edge_count);
    }
    spec.base_index_sha256 = v0_test::digestOf("graph-layout-base");
    spec.adjacency_sha256 = graph.adjacencyFingerprint();

    const std::string path = "v0_sidecar_graph_layout.v0meta";
    v0_test::FileCleanup cleanup(path);
    {
        hnswlib::V0SidecarWriter writer(path, spec);
        const uint8_t code[2] = {0U, 0U};
        for (uint64_t edge = 0; edge < spec.directed_edge_count; ++edge) {
            writer.writeEdgeRecord(
                code, 2U, 0U, 1.0, 0.0, 0.0, 0.0);
        }
        writer.finalize();
    }
    const hnswlib::V0OwnedSidecar sidecar =
        hnswlib::loadV0Sidecar(path);
    hnswlib::validateV0SidecarGraphLayout(sidecar.view(), graph);

    const float added[4] = {100.0f, 101.0f, 102.0f, 103.0f};
    index.addPoint(added, node_count);
    const hnswlib::V0Layer0GraphView changed_graph =
        index.getV0Layer0GraphView();
    v0_test::requireThrows(
        [&sidecar, &changed_graph]() {
            hnswlib::validateV0SidecarGraphLayout(
                sidecar.view(), changed_graph);
        },
        "stale graph layout was accepted");
}

}  // namespace

int main() {
#ifndef HNSWLIB_ENABLE_EDGE_QUANT_V0
    std::cerr << "HNSWLIB_ENABLE_EDGE_QUANT_V0 is required" << std::endl;
    return 2;
#else
    testMalformedFiles();
    testWriterValidation();
    testCompatibilityMismatch();
    testGraphSlotCompatibility();
    std::cout << "v0_sidecar_mismatch_test_ok" << std::endl;
    return 0;
#endif
}
