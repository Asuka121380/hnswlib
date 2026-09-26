#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>

#include "tools/edge_estimation/backends/rotated_pq.h"
#include "tools/edge_estimation/event_format.h"
#include "tools/edge_estimation/query_store.h"

namespace {
void writeFloats(const std::filesystem::path& path, const float* values, size_t count) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(values), static_cast<std::streamsize>(count * 4U));
    if (!output) throw std::runtime_error("failed to write float fixture");
}
}

int main() {
    const std::filesystem::path root("uq_rotated_pq_fixture");
    const std::filesystem::path queries_path("uq_rotated_pq_queries.fvecs");
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "model");
    std::filesystem::create_directories(root / "records");
    const float codebook[] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float rotation[] = {0.0f, 1.0f, 1.0f, 0.0f};
    writeFloats(root / "model/codebook.f32le", codebook, 4U);
    writeFloats(root / "model/rotation.f32le", rotation, 4U);
    std::vector<uint8_t> record;
    uq::detail::appendF64(record, 1.0);
    uq::detail::appendF64(record, 0.0);
    record.push_back(0U);
    uq::detail::writeFile((root / "records/edges.bin").string(), record);
    {
        std::ofstream queries(queries_path, std::ios::binary | std::ios::trunc);
        const uint32_t dimension = 2U;
        const float query[] = {2.0f, 3.0f};
        queries.write(reinterpret_cast<const char*>(&dimension), 4);
        queries.write(reinterpret_cast<const char*>(query), 8);
    }
    std::ofstream config(root / "native.cfg", std::ios::trunc);
    config << "format=uq-rotated-pq/1\nbackend=opq\nimplementation=test\nprovider=test\n"
           << "dimension=2\nm=1\nnbits=1\ndsub=2\nedge_count=1\nrecord_size=17\n"
           << "catalog_identity=" << std::string(64U, '0') << "\ncoverage=full_graph\n"
           << "codebook=model/codebook.f32le\ncodebook_sha256="
           << uq::artifactSha256(root / "model/codebook.f32le") << "\n"
           << "rotation=model/rotation.f32le\nrotation_sha256="
           << uq::artifactSha256(root / "model/rotation.f32le") << "\n"
           << "rotation_layout=row_major_r_times_column\nrotation_bias=none\n"
           << "records=records/edges.bin\nrecords_sha256="
           << uq::artifactSha256(root / "records/edges.bin") << "\n";
    config.close();
    uq::Header header{};
    header.dimension = 2U;
    std::shared_ptr<const uq::QueryStore> queries =
        std::make_shared<uq::QueryStore>(queries_path, 2U);
    {
        uq::RotatedPqArtifactKernel kernel(root, queries, header);
        kernel.prepareQuery(0U);
        uq::EventRecord event{};
        event.query_id = 0U;
        event.edge_id = 0U;
        event.d_current = 10.0;
        const double score = kernel.score(event);
        if (std::fabs(score - 5.0) > 1e-6)
            throw std::runtime_error("rotated-PQ score used the wrong matrix orientation");
    }
    std::filesystem::remove(queries_path);
    std::filesystem::remove_all(root);
    return 0;
}
