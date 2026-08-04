#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "v0_sidecar_test_utils.h"
#include "hnswlib/edge_quant_v0_cap_diagnostic.h"

namespace {

void requireNear(double actual, double expected, const char* message) {
    v0_test::require(std::fabs(actual - expected) < 1e-12, message);
}

void testRawGeometryExport() {
    const std::string path = "v0_cap_diagnostic_export_test.v0meta";
    v0_test::FileCleanup cleanup(path);

    hnswlib::V0SidecarWriteSpec spec;
    spec.dimension = 4U;
    spec.pq_m = 2U;
    spec.pq_nbits = 8U;
    spec.pq_ksub = 256U;
    spec.pq_dsub = 2U;
    spec.node_count = 1U;
    spec.directed_edge_count = 1U;
    spec.training_metadata_json = "{\"purpose\":\"cap_phase1_test\"}";
    spec.codebook_centroids.resize(2U * 256U * 2U, 0.0f);
    spec.codebook_centroids[(0U * 256U + 3U) * 2U] = 1.0f;
    spec.codebook_centroids[(1U * 256U + 7U) * 2U + 1U] = 1.0f;
    spec.node_offsets.push_back(0U);
    spec.node_offsets.push_back(1U);
    spec.base_index_sha256 = v0_test::digestOf("cap-input-index");
    spec.adjacency_sha256 = v0_test::digestOf("cap-input-adjacency");

    hnswlib::V0EdgeRecord record;
    record.code.push_back(3U);
    record.code.push_back(7U);
    record.edge_length = 1.0;
    record.direction_error = 1.0;
    hnswlib::V0SidecarWriter writer(path, spec);
    writer.writeEdgeRecord(record);
    writer.finalize();

    const hnswlib::V0OwnedSidecar owned = hnswlib::loadV0Sidecar(path);
    const hnswlib::V0SidecarView sidecar = owned.view();
    const hnswlib::V0EdgeRecordView edge = sidecar.edgeRecord(0U);
    const float query[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float current[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float candidate[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    const hnswlib::V0SphericalCapDiagnosticInput input =
        hnswlib::computeV0SphericalCapDiagnosticInput(
            query, current, candidate, sidecar, edge);
    v0_test::require(input.valid, "raw cap diagnostic input is invalid");
    requireNear(input.reconstruction_norm, std::sqrt(2.0),
                "reconstruction norm mismatch");
    requireNear(input.x_norm, 2.0, "x norm mismatch");
    requireNear(input.x_dot_reconstruction, 2.0,
                "x dot reconstruction mismatch");
    requireNear(input.true_edge_norm, 1.0, "true edge norm mismatch");
    requireNear(input.x_dot_true_direction, 1.0,
                "true direction support mismatch");
    requireNear(input.geometric_squared_distance, 3.0,
                "geometric squared distance mismatch");
    requireNear(input.actual_direction_error, 1.0,
                "actual direction error mismatch");

    const hnswlib::V0SphericalCapDiagnosticInput zero_edge =
        hnswlib::computeV0SphericalCapDiagnosticInput(
            query, current, current, sidecar, edge);
    v0_test::require(!zero_edge.valid, "zero true edge was accepted");
}

}  // namespace

int main() {
#ifndef HNSWLIB_ENABLE_V0_SPHERICAL_CAP_DIAGNOSTIC
    return 2;
#else
    testRawGeometryExport();
    return 0;
#endif
}
