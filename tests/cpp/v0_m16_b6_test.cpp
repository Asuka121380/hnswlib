#include <cmath>
#include <vector>
#include "v0_sidecar_test_utils.h"

int main() {
    const std::string path = "v0_m16_b6_test.v0meta";
    v0_test::FileCleanup cleanup(path);
    hnswlib::V0SidecarWriteSpec spec;
    spec.dimension = 960; spec.pq_m = 16; spec.pq_nbits = 6;
    spec.pq_ksub = 64; spec.pq_dsub = 60;
    spec.node_count = 1; spec.directed_edge_count = 1;
    spec.node_offsets = {0, 1}; spec.training_metadata_json = "{}";
    spec.codebook_centroids.resize(960 * 64);
    for (size_t i = 0; i < spec.codebook_centroids.size(); ++i)
        spec.codebook_centroids[i] = float(int(i % 19) - 9) / 32;
    std::vector<uint8_t> code(16);
    std::vector<float> query(960);
    for (size_t i = 0; i < query.size(); ++i) query[i] = float(int(i % 13) - 6) / 16;
    for (size_t i = 0; i < code.size(); ++i) code[i] = (i * 7) % 64;
    {
        hnswlib::V0SidecarWriter writer(path, spec);
        writer.writeEdgeRecord(code.data(), code.size(), 0, 2.0, 0.1, 0.25, 0.0);
        writer.finalize();
    }
    auto owned = hnswlib::loadV0Sidecar(path);
    auto sidecar = owned.view();
    hnswlib::V0QueryLut lut(query.data(), sidecar);
    v0_test::require(lut.tableBytes() == 4096, "M16 b6 LUT is not 4 KiB");
    v0_test::require(hnswlib::edge_quant_v0_detail::edgeRecordStride(16) == 56,
                     "unexpected byte-code record stride");
    double direct = 0;
    for (size_t m = 0; m < 16; ++m)
        for (size_t d = 0; d < 60; ++d)
            direct += double(query[m * 60 + d]) * spec.codebook_centroids[(m * 64 + code[m]) * 60 + d];
    hnswlib::EdgeQuantV0ApproxQueryContext fast(query.data(), sidecar.header(), spec.codebook_centroids.data());
    auto estimate = fast.evaluateRawFast(sidecar.edgeRecord(0), 10.0);
    v0_test::require(estimate.valid(), "M16 b6 estimate invalid");
    v0_test::require(std::fabs(estimate.approximate_squared_distance - (10 + 4 - 4 * (direct - .25))) < 1e-5,
                     "M16 b6 raw estimate differs from direct reconstruction");
}
