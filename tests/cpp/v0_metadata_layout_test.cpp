#include <iostream>
#include <string>
#include <vector>

#include "v0_sidecar_test_utils.h"

int main() {
#ifndef HNSWLIB_ENABLE_EDGE_QUANT_V0
    std::cerr << "HNSWLIB_ENABLE_EDGE_QUANT_V0 is required" << std::endl;
    return 2;
#else
    const std::string path = "v0_metadata_layout_test.v0meta";
    v0_test::FileCleanup cleanup(path);
    v0_test::writeTinySidecar(path);
    const std::vector<uint8_t> bytes = v0_test::readFile(path);

    const uint8_t expected_magic[8] = {
        'H', 'N', 'S', 'W', 'V', '0', 'P', 'Q'
    };
    for (size_t i = 0; i < 8U; ++i) {
        v0_test::require(bytes[i] == expected_magic[i], "magic mismatch");
    }
    v0_test::require(
        bytes[8] == 1U && bytes[9] == 0U &&
        bytes[10] == 0U && bytes[11] == 0U,
        "version is not explicit little-endian");
    v0_test::require(
        bytes[12] == 0U && bytes[13] == 1U &&
        bytes[14] == 0U && bytes[15] == 0U,
        "header size is not explicit little-endian");
    v0_test::require(
        bytes[16] == 1U && bytes[17] == 1U &&
        bytes[18] == 1U && bytes[19] == 1U,
        "representation enum layout mismatch");
    v0_test::require(
        hnswlib::edge_quant_v0_detail::readUint32LittleEndian(
            bytes.data() + 20U) == 4U,
        "dimension offset mismatch");
    v0_test::require(
        hnswlib::edge_quant_v0_detail::readUint32LittleEndian(
            bytes.data() + 44U) == 40U,
        "edge-record stride offset mismatch");

    const hnswlib::V0SidecarView view =
        hnswlib::parseV0Sidecar(bytes.data(), bytes.size());
    const hnswlib::V0SidecarHeader& header = view.header();
    v0_test::require(
        header.training_metadata.offset ==
            hnswlib::EDGE_QUANT_V0_FIXED_HEADER_SIZE,
        "training metadata is not immediately after header");
    v0_test::require(
        header.codebook.offset ==
            header.training_metadata.offset +
            header.training_metadata.size,
        "codebook is not canonical");
    v0_test::require(
        header.node_offsets.offset ==
            header.codebook.offset + header.codebook.size,
        "node offsets are not canonical");
    v0_test::require(
        header.edge_records.offset ==
            header.node_offsets.offset + header.node_offsets.size,
        "edge records are not canonical");
    v0_test::require(
        header.checksums.offset ==
            header.edge_records.offset + header.edge_records.size,
        "checksums are not canonical");
    v0_test::require(
        bytes.size() ==
            header.checksums.offset + header.checksums.size,
        "canonical file size mismatch");

    const uint32_t scalar_offset =
        hnswlib::edge_quant_v0_detail::edgeRecordScalarOffset(
            header.pq_code_size);
    const uint8_t* first_record =
        bytes.data() + header.edge_records.offset;
    for (uint32_t i = header.pq_code_size + 1U;
         i < scalar_offset;
         ++i) {
        v0_test::require(
            first_record[i] == 0U,
            "edge-record padding is not explicit zero");
    }

    const std::string empty_path =
        "v0_metadata_layout_empty_test.v0meta";
    v0_test::FileCleanup empty_cleanup(empty_path);
    hnswlib::V0SidecarWriteSpec empty_spec = v0_test::tinySpec();
    empty_spec.node_count = 0U;
    empty_spec.directed_edge_count = 0U;
    empty_spec.node_offsets.assign(1U, 0U);
    {
        hnswlib::V0SidecarWriter writer(empty_path, empty_spec);
        writer.finalize();
    }
    const hnswlib::V0OwnedSidecar empty_sidecar =
        hnswlib::loadV0Sidecar(empty_path);
    v0_test::require(
        empty_sidecar.header().node_count == 0U &&
        empty_sidecar.header().directed_edge_count == 0U,
        "empty sidecar round-trip mismatch");

    std::cout << "v0_metadata_layout_test_ok" << std::endl;
    return 0;
#endif
}
