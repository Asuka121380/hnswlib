#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "v0_sidecar_test_utils.h"

namespace {

void requireNear(double actual, double expected, const char* message) {
    if (std::fabs(actual - expected) > 1e-12) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main(int argc, char** argv) {
#ifndef HNSWLIB_ENABLE_EDGE_QUANT_V0
    std::cerr << "HNSWLIB_ENABLE_EDGE_QUANT_V0 is required" << std::endl;
    return 2;
#else
    const bool validate_existing =
        argc == 3 && std::string(argv[1]) == "--validate-existing";
    const bool keep_output = argc == 2 || validate_existing;
    const std::string path = validate_existing ?
        argv[2] :
        (argc == 2 ? argv[1] : "v0_sidecar_roundtrip_test.v0meta");
    if (!validate_existing) {
        std::remove(path.c_str());
    }

    const hnswlib::V0SidecarWriteSpec spec = v0_test::tinySpec();
    const std::vector<hnswlib::V0EdgeRecord> records =
        v0_test::tinyRecords();
    if (!validate_existing) {
        hnswlib::V0SidecarWriter writer(path, spec);
        for (size_t i = 0; i < records.size(); ++i) {
            writer.writeEdgeRecord(records[i]);
        }
        writer.finalize();
    }

    const hnswlib::V0OwnedSidecar owned =
        hnswlib::loadV0Sidecar(path);
    const hnswlib::V0SidecarView view = owned.view();
    const hnswlib::V0SidecarHeader& header = view.header();

    v0_test::require(header.dimension == spec.dimension, "dimension mismatch");
    v0_test::require(header.pq_m == spec.pq_m, "pq_m mismatch");
    v0_test::require(header.pq_nbits == spec.pq_nbits, "nbits mismatch");
    v0_test::require(header.pq_ksub == spec.pq_ksub, "ksub mismatch");
    v0_test::require(header.pq_dsub == spec.pq_dsub, "dsub mismatch");
    v0_test::require(
        header.pq_code_size == spec.pq_m,
        "code size mismatch");
    v0_test::require(
        header.node_count == spec.node_count,
        "node count mismatch");
    v0_test::require(
        header.directed_edge_count == spec.directed_edge_count,
        "edge count mismatch");
    v0_test::require(
        header.base_index_sha256 == spec.base_index_sha256,
        "base-index fingerprint mismatch");
    v0_test::require(
        header.adjacency_sha256 == spec.adjacency_sha256,
        "adjacency fingerprint mismatch");
    v0_test::require(
        view.trainingMetadataJson() == spec.training_metadata_json,
        "training metadata mismatch");

    for (size_t i = 0; i < spec.codebook_centroids.size(); ++i) {
        requireNear(
            view.codebookCentroid(i),
            spec.codebook_centroids[i],
            "codebook centroid mismatch");
    }
    for (size_t i = 0; i < spec.node_offsets.size(); ++i) {
        v0_test::require(
            view.nodeOffset(i) == spec.node_offsets[i],
            "node offset mismatch");
    }
    for (size_t i = 0; i < records.size(); ++i) {
        const hnswlib::V0EdgeRecordView actual = view.edgeRecord(i);
        v0_test::require(
            actual.codeSize() == records[i].code.size(),
            "edge code size mismatch");
        for (size_t m = 0; m < records[i].code.size(); ++m) {
            v0_test::require(
                actual.code(m) == records[i].code[m],
                "edge code mismatch");
        }
        v0_test::require(
            actual.flags() == records[i].flags,
            "edge flags mismatch");
        requireNear(
            actual.edgeLength(),
            records[i].edge_length,
            "edge length mismatch");
        requireNear(
            actual.directionError(),
            records[i].direction_error,
            "direction error mismatch");
        requireNear(
            actual.anchorProjection(),
            records[i].anchor_projection,
            "anchor projection mismatch");
        requireNear(
            actual.numericPadding(),
            records[i].numeric_padding,
            "numeric padding mismatch");
    }

    hnswlib::V0IndexCompatibility compatibility;
    compatibility.dimension = spec.dimension;
    compatibility.node_count = spec.node_count;
    compatibility.directed_edge_count = spec.directed_edge_count;
    compatibility.base_index_sha256 = spec.base_index_sha256;
    compatibility.adjacency_sha256 = spec.adjacency_sha256;
    hnswlib::validateV0SidecarCompatibility(header, compatibility);

    v0_test::requireThrows(
        [&view]() { (void)view.nodeOffset(view.header().node_count + 1U); },
        "out-of-range node offset was accepted");
    v0_test::requireThrows(
        [&view]() {
            (void)view.edgeRecord(view.header().directed_edge_count);
        },
        "out-of-range edge record was accepted");

    if (!keep_output) {
        v0_test::require(
            std::remove(path.c_str()) == 0,
            "failed to remove round-trip sidecar");
    }
    std::cout << "v0_sidecar_roundtrip_test_ok" << std::endl;
    return 0;
#endif
}
