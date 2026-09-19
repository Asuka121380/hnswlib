#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <queue>
#include <random>
#include <vector>

#include "hnswlib/edge_quant_v0_numeric.h"
#include "v0_sidecar_test_utils.h"
#include "hnswlib/edge_quant_v0_residual_io.h"

namespace {
template<class Q> bool same(Q a, Q b) {
    if (a.size() != b.size()) return false;
    while (!a.empty()) {
        if (a.top() != b.top()) return false;
        a.pop(); b.pop();
    }
    return true;
}
}

int main() {
    const uint32_t dim = 8U;
    const size_t n = 24U;
    const char* sidecar = "v0_residual_search_test.v0meta";
    const char* companion_path = "v0_residual_search_test.v0res";
    const char* keep_path = "v0_residual_search_keep.v0res";
    const char* prune_path = "v0_residual_search_prune.v0res";
    const char* encoded_path = "v0_residual_search_encoded.v0res";
    const char* index_path = "v0_residual_search_index.bin";
    const char* matrix_path = "v0_residual_search_matrix.f32";
    const char* query_path = "v0_residual_search_queries.fvecs";
    const char* trace_path = "v0_residual_search_trace.csv";
    const char* events_path = "v0_residual_search_events.csv";
    const char* edges_path = "v0_residual_search_edges.bin";
    v0_test::FileCleanup sidecar_cleanup(sidecar);
    v0_test::FileCleanup companion_cleanup(companion_path);
    v0_test::FileCleanup keep_cleanup(keep_path);
    v0_test::FileCleanup prune_cleanup(prune_path);
    v0_test::FileCleanup encoded_cleanup(encoded_path);
    v0_test::FileCleanup index_cleanup(index_path);
    v0_test::FileCleanup matrix_cleanup(matrix_path);
    v0_test::FileCleanup query_cleanup(query_path);
    v0_test::FileCleanup trace_cleanup(trace_path);
    v0_test::FileCleanup events_cleanup(events_path);
    v0_test::FileCleanup edges_cleanup(edges_path);
    std::mt19937 rng(911U);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> base(n * dim);
    for (size_t i = 0; i < base.size(); ++i) base[i] = normal(rng);
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, n, 4U, 40U, 911U);
    for (size_t i = 0; i < n; ++i)
        index.addPoint(base.data() + i * dim, i);
    const hnswlib::V0Layer0GraphView graph = index.getV0Layer0GraphView();
    hnswlib::V0SidecarWriteSpec spec;
    spec.dimension = dim; spec.pq_m = 1U; spec.pq_nbits = 8U;
    spec.pq_ksub = 256U; spec.pq_dsub = dim;
    spec.node_count = n;
    spec.training_metadata_json = "{\"purpose\":\"residual-search-test\"}";
    spec.codebook_centroids.assign(256U * dim, 0.0f);
    spec.node_offsets.push_back(0U);
    for (size_t node = 0; node < n; ++node) {
        spec.directed_edge_count += graph.neighbors(node).size;
        spec.node_offsets.push_back(spec.directed_edge_count);
    }
    v0_test::require(spec.directed_edge_count <= 256U,
                     "test graph exceeds codebook capacity");
    spec.base_index_sha256 = index.getV0SerializedIndexFingerprint();
    spec.adjacency_sha256 = graph.adjacencyFingerprint();
    std::vector<hnswlib::V0EdgeRecord> records;
    std::vector<float> direction(dim);
    graph.forEachEdge([&](hnswlib::tableint source,
                          hnswlib::tableint target, size_t) {
        const hnswlib::V0EdgeDirectionInfo info =
            hnswlib::fillV0UnitEdgeDirection(
                graph.floatVector(source), graph.floatVector(target),
                dim, direction.data());
        hnswlib::V0EdgeRecord record;
        record.code.push_back(static_cast<uint8_t>(records.size()));
        if (info.zero_length) {
            record.flags = hnswlib::V0_EDGE_EXACT_ONLY |
                           hnswlib::V0_ZERO_LENGTH_EDGE;
        } else {
            for (uint32_t j = 0; j < dim; ++j)
                spec.codebook_centroids[records.size() * dim + j] = direction[j];
            const hnswlib::V0EdgeNumericMetadata numeric =
                hnswlib::computeV0EdgeNumericMetadata(
                    graph.floatVector(source), graph.floatVector(target),
                    direction.data(), dim);
            record.edge_length = numeric.edge_length;
            record.direction_error = numeric.direction_error;
            record.anchor_projection = numeric.anchor_projection;
            record.numeric_padding = numeric.numeric_padding;
        }
        records.push_back(record);
    });
    {
        hnswlib::V0SidecarWriter writer(sidecar, spec);
        for (size_t i = 0; i < records.size(); ++i)
            writer.writeEdgeRecord(records[i]);
        writer.finalize();
    }
    index.loadEdgeQuantV0Metadata(sidecar);
    hnswlib::V0ResidualIdentity identity;
    identity.index_sha = index.getV0SerializedIndexFingerprint();
    identity.sidecar_sha = hnswlib::v0ResidualFileSha(sidecar);
    identity.adjacency_sha = graph.adjacencyFingerprint();
    std::vector<float> matrix(8U * dim, 0.0f);
    for (uint32_t j = 0; j < dim; ++j) matrix[j * dim + j] = 1.0f;
    index.saveIndex(index_path);
    {
        std::ofstream output(query_path, std::ios::binary);
        for (size_t i = 0; i < n; ++i) {
            const int32_t stored_dim = static_cast<int32_t>(dim);
            output.write(reinterpret_cast<const char*>(&stored_dim), 4U);
            output.write(reinterpret_cast<const char*>(base.data() + i * dim),
                         static_cast<std::streamsize>(dim * sizeof(float)));
        }
    }
    {
        const hnswlib::V0Layer0NeighborSpan first = graph.neighbors(0U);
        v0_test::require(first.size > 0U, "test graph has no first edge");
        const hnswlib::tableint target = first.ids[0];
        const hnswlib::DISTFUNC<float> distance = space.get_dist_func();
        const void* param = space.get_dist_func_param();
        const float* query = base.data();
        std::ofstream trace(trace_path);
        trace << "query_id,current_node_id,candidate_id,graph_layer,threshold,"
                 "exact_squared_distance,current_squared_distance\n";
        trace << "0,0," << target << ",0,1," <<
            distance(query, graph.floatVector(target), param) << ',' <<
            distance(query, graph.floatVector(0U), param) << '\n';
    }
#ifdef _WIN32
    const char* exporter = "v0_residual_exporter.exe";
#else
    const char* exporter = "./v0_residual_exporter";
#endif
    const std::string export_command = std::string(exporter) +
        " --index " + index_path + " --sidecar " + sidecar +
        " --queries " + query_path + " --input " + trace_path +
        " --events " + events_path + " --edges " + edges_path +
        " --dimension 8";
    v0_test::require(std::system(export_command.c_str()) == 0,
                     "residual exporter failed on test index");
    {
        std::ifstream edge_file(edges_path, std::ios::binary);
        char magic[8];
        edge_file.read(magic, 8U);
        v0_test::require(std::string(magic, 8U) == "V0RGE002",
                         "exported edge format mismatch");
    }
    {
        std::ofstream output(matrix_path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(matrix.data()),
                     static_cast<std::streamsize>(matrix.size() * sizeof(float)));
    }
#ifdef _WIN32
    const char* encoder = "v0_residual_encoder.exe";
#else
    const char* encoder = "./v0_residual_encoder";
#endif
    const std::string command = std::string(encoder) +
        " --index " + index_path + " --sidecar " + sidecar +
        " --matrix " + matrix_path + " --output " + encoded_path +
        " --dimension 8 --bits 8 --seed 42 --chunk-edges 7";
    v0_test::require(std::system(command.c_str()) == 0,
                     "full graph residual encoder failed on test index");
    const hnswlib::V0ResidualCompanion encoded(
        encoded_path, identity, dim, records.size());
    v0_test::require(encoded.bits() == 8U && encoded.count() == records.size(),
                     "encoded companion layout mismatch");
    {
        hnswlib::V0ResidualCompanionWriter writer(
            companion_path, identity, matrix, dim, 8U, 42U, records.size());
        for (size_t i = 0; i < records.size(); ++i)
            writer.append(std::vector<uint8_t>(1U, 0U), 0.0f, 0.0f, false);
        writer.finish();
    }
    const hnswlib::V0ResidualCompanion companion(
        companion_path, identity, dim, records.size());
    {
        hnswlib::V0ResidualCompanionWriter writer(
            keep_path, identity, matrix, dim, 8U, 42U, records.size());
        for (size_t i = 0; i < records.size(); ++i)
            writer.append(std::vector<uint8_t>(1U, 0U), 0.0f, -1e6f, true);
        writer.finish();
    }
    {
        hnswlib::V0ResidualCompanionWriter writer(
            prune_path, identity, matrix, dim, 8U, 42U, records.size());
        for (size_t i = 0; i < records.size(); ++i)
            writer.append(std::vector<uint8_t>(1U, 0U), 0.0f, 1e6f, true);
        writer.finish();
    }
    const hnswlib::V0ResidualCompanion keep(
        keep_path, identity, dim, records.size());
    const hnswlib::V0ResidualCompanion prune(
        prune_path, identity, dim, records.size());
    index.setEf(6U);
    hnswlib::V0ResidualPruningConfig config;
    uint64_t fallback = 0U, evaluated = 0U, pruned = 0U;
    for (size_t i = 0; i < n; ++i) {
        const float* query = base.data() + i * dim;
        hnswlib::V0QueryMetrics metrics;
        const auto baseline = index.searchKnn(query, 3U);
        const auto active = index.searchKnnV0Residual(
            query, 3U, config, companion, &metrics);
        const auto encoded_result = index.searchKnnV0ResidualFast(
            query, 3U, config, encoded);
        v0_test::require(same(baseline, encoded_result),
                         "actual encoder changed exact result for zero residual");
        const auto fast = index.searchKnnV0ResidualFast(
            query, 3U, config, companion);
        v0_test::require(same(baseline, active) && same(active, fast),
                         "invalid companion fallback changed HNSW results");
        hnswlib::V0QueryMetrics keep_metrics, prune_metrics;
        const auto kept = index.searchKnnV0Residual(
            query, 3U, config, keep, &keep_metrics);
        const auto pruned_result = index.searchKnnV0Residual(
            query, 3U, config, prune, &prune_metrics);
        v0_test::require(same(baseline, kept) &&
                         same(kept, index.searchKnnV0ResidualFast(
                             query, 3U, config, keep)),
                         "residual keep path changed exact results");
        v0_test::require(same(pruned_result, index.searchKnnV0ResidualFast(
                             query, 3U, config, prune)),
                         "residual prune fast path disagrees with metrics path");
        fallback += metrics.residual_fallback;
        evaluated += keep_metrics.residual_evaluated;
        pruned += prune_metrics.residual_pruned;
    }
    v0_test::require(fallback > 0U, "residual fallback path was not exercised");
    v0_test::require(evaluated > 0U && pruned > 0U,
                     "residual correction and prune paths were not exercised");
    return 0;
}
