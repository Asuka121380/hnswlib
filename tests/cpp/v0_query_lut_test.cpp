#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "v0_sidecar_test_utils.h"

namespace {

void writeM32Sidecar(
    const std::string& path,
    std::vector<float>* centroids,
    std::vector<uint8_t>* code) {
    const uint32_t dimension = 64U;
    const uint32_t pq_m = 32U;
    const uint32_t pq_ksub = 256U;
    const uint32_t pq_dsub = 2U;

    centroids->resize(
        static_cast<size_t>(pq_m) * pq_ksub * pq_dsub);
    for (uint32_t subquantizer = 0U;
         subquantizer < pq_m;
         ++subquantizer) {
        for (uint32_t centroid = 0U;
             centroid < pq_ksub;
             ++centroid) {
            for (uint32_t coordinate = 0U;
                 coordinate < pq_dsub;
                 ++coordinate) {
                const size_t index =
                    (static_cast<size_t>(subquantizer) * pq_ksub +
                     centroid) *
                        pq_dsub +
                    coordinate;
                (*centroids)[index] =
                    static_cast<float>(
                        static_cast<int>((index * 37U) % 101U) - 50) /
                    128.0f;
            }
        }
    }

    code->resize(pq_m);
    for (uint32_t subquantizer = 0U;
         subquantizer < pq_m;
         ++subquantizer) {
        (*code)[subquantizer] =
            static_cast<uint8_t>(
                (subquantizer * 29U + 7U) % pq_ksub);
    }

    hnswlib::V0SidecarWriteSpec spec;
    spec.dimension = dimension;
    spec.pq_m = pq_m;
    spec.pq_nbits = 8U;
    spec.pq_ksub = pq_ksub;
    spec.pq_dsub = pq_dsub;
    spec.node_count = 1U;
    spec.directed_edge_count = 1U;
    spec.training_metadata_json =
        "{\"purpose\":\"milestone8_query_lut_test\"}";
    spec.codebook_centroids = *centroids;
    spec.node_offsets.push_back(0U);
    spec.node_offsets.push_back(1U);
    spec.base_index_sha256 = v0_test::digestOf("lut-index");
    spec.adjacency_sha256 = v0_test::digestOf("lut-adjacency");

    hnswlib::V0EdgeRecord record;
    record.code = *code;
    record.edge_length = 1.0;
    record.direction_error = 0.25;
    record.anchor_projection = 0.0;

    hnswlib::V0SidecarWriter writer(path, spec);
    writer.writeEdgeRecord(record);
    writer.finalize();
}

void testM32QueryLut() {
    const std::string path = "v0_query_lut_test.v0meta";
    v0_test::FileCleanup cleanup(path);

    std::vector<float> centroids;
    std::vector<uint8_t> code;
    writeM32Sidecar(path, &centroids, &code);
    hnswlib::V0OwnedSidecar owned =
        hnswlib::loadV0Sidecar(path);
    const hnswlib::V0SidecarView sidecar = owned.view();

    std::vector<float> query(64U);
    for (size_t coordinate = 0U;
         coordinate < query.size();
         ++coordinate) {
        query[coordinate] =
            static_cast<float>(
                static_cast<int>((coordinate * 17U) % 31U) - 15) /
            16.0f;
    }

    const hnswlib::V0QueryLut lut(query.data(), sidecar);
    v0_test::require(
        lut.dimension() == 64U &&
        lut.subquantizerCount() == 32U &&
        lut.centroidCountPerSubquantizer() == 256U &&
        lut.subvectorDimension() == 2U,
        "M32 query LUT layout is incorrect");
    v0_test::require(
        lut.tableBytes() == 32U * 256U * sizeof(float),
        "M32 query LUT is not 32 KiB");

    const hnswlib::V0EdgeRecordView edge =
        sidecar.edgeRecord(0U);
    long double direct = 0.0L;
    for (uint32_t subquantizer = 0U;
         subquantizer < 32U;
         ++subquantizer) {
        for (uint32_t coordinate = 0U;
             coordinate < 2U;
             ++coordinate) {
            const size_t centroid_index =
                (static_cast<size_t>(subquantizer) * 256U +
                 code[subquantizer]) *
                    2U +
                coordinate;
            direct +=
                static_cast<long double>(
                    query[subquantizer * 2U + coordinate]) *
                static_cast<long double>(
                    centroids[centroid_index]);
        }
    }
    const double lookup = lut.innerProductUpper(edge);
    v0_test::require(
        lookup >= static_cast<double>(direct) - 1e-12,
        "query LUT lookup is not an upper bound");
    v0_test::require(
        std::fabs(lookup - static_cast<double>(direct)) < 1e-4,
        "query LUT lookup differs excessively from direct reconstruction");

    v0_test::requireThrows(
        [&lut]() {
            (void)lut.value(32U, 0U);
        },
        "query LUT accepted an invalid subquantizer");
    v0_test::requireThrows(
        [&lut]() {
            (void)lut.value(0U, 256U);
        },
        "query LUT accepted an invalid centroid");

    query[3] = std::numeric_limits<float>::quiet_NaN();
    v0_test::requireThrows(
        [&query, &sidecar]() {
            hnswlib::V0QueryLut invalid(
                query.data(), sidecar);
            (void)invalid;
        },
        "query LUT accepted a non-finite query");
}

}  // namespace

int main() {
#ifndef HNSWLIB_ENABLE_EDGE_QUANT_V0
    return 2;
#else
    testM32QueryLut();
    return 0;
#endif
}
