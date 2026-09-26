#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

#include "hnswlib/hnswlib.h"
#include "tools/edge_estimation/backends/pq_legacy.h"
#include "tools/edge_estimation/backends/pq_qjl_legacy.h"

int main() {
    if (uq::legacyStatus(hnswlib::V0BoundStatus::Valid) !=
        hnswlib::edge_estimation::EstimateStatus::Valid) {
        return 1;
    }
    if (uq::legacyStatus(hnswlib::V0BoundStatus::ExactOnly) !=
        hnswlib::edge_estimation::EstimateStatus::LegacyExactOnly) {
        return 1;
    }
    const std::filesystem::path root("uq_legacy_artifact_fixture");
    const std::filesystem::path query_path("uq_legacy_artifact_queries.fvecs");
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "legacy");
    hnswlib::V0SidecarWriteSpec spec;
    spec.dimension = 2U; spec.pq_m = 1U; spec.pq_nbits = 1U;
    spec.pq_ksub = 2U; spec.pq_dsub = 2U;
    spec.node_count = 1U; spec.directed_edge_count = 1U;
    spec.training_metadata_json = "{}";
    spec.codebook_centroids = {1.0f, 0.0f, 0.0f, 1.0f};
    spec.node_offsets = {0U, 1U};
    {
        hnswlib::V0SidecarWriter writer((root / "legacy/sidecar.bin").string(), spec);
        const uint8_t code[] = {0U};
        writer.writeEdgeRecord(code, 1U, 0U, 2.0, 0.0, 0.0, 0.0);
        writer.finalize();
    }
    {
        std::ofstream query(query_path, std::ios::binary | std::ios::trunc);
        const uint32_t dimension = 2U; const float values[] = {3.0f, 4.0f};
        query.write(reinterpret_cast<const char*>(&dimension), 4U);
        query.write(reinterpret_cast<const char*>(values), 8U);
    }
    {
        std::ofstream config(root / "native.cfg", std::ios::trunc);
        config << "format=uq-pq-legacy/1\nbackend=pq_legacy\ncoverage=full_graph\n"
               << "edge_count=1\ncatalog_identity=" << std::string(64U, '0')
               << "\nsidecar=legacy/sidecar.bin\nsidecar_sha256="
               << uq::artifactSha256(root / "legacy/sidecar.bin") << "\n";
    }
    uq::Header header{}; header.dimension = 2U;
    std::shared_ptr<const uq::QueryStore> queries =
        std::make_shared<uq::QueryStore>(query_path, 2U);
    uq::PqLegacyArtifactKernel kernel(root, queries, header);
    kernel.prepareQuery(0U);
    uq::EventRecord event{}; event.query_id = 0U; event.edge_id = 0U;
    event.source_id = 0U; event.neighbor_slot = 0U; event.d_current = 25.0;
    if (std::fabs(kernel.score(event) - 17.0) > 1e-6)
        throw std::runtime_error("legacy artifact lifecycle produced the wrong score");
    std::filesystem::remove(query_path);
    std::filesystem::remove_all(root);
    std::cout << "uq_legacy_adapter_compile_test_ok" << std::endl;
    return 0;
}
