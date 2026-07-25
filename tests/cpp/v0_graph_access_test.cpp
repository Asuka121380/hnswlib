#include <cstdio>
#include <cstring>
#include <iostream>
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
void requireOutOfRange(Callable callable, const std::string& message) {
    try {
        callable();
    } catch (const std::out_of_range&) {
        return;
    }
    throw std::runtime_error(message);
}

std::string rawAdjacencyFingerprint(
    const hnswlib::HierarchicalNSW<float>& index) {
    hnswlib::EdgeQuantV0Sha256 sha;
    const size_t node_count = index.cur_element_count.load();
    hnswlib::edgeQuantV0Sha256UpdateUint64LittleEndian(
        sha, static_cast<uint64_t>(node_count));

    for (size_t source = 0; source < node_count; ++source) {
        hnswlib::linklistsizeint* raw =
            index.get_linklist0(static_cast<hnswlib::tableint>(source));
        const size_t degree = index.getListCount(raw);
        const hnswlib::tableint* ids =
            reinterpret_cast<const hnswlib::tableint*>(raw + 1);

        hnswlib::edgeQuantV0Sha256UpdateUint32LittleEndian(
            sha, static_cast<uint32_t>(source));
        hnswlib::edgeQuantV0Sha256UpdateUint32LittleEndian(
            sha, static_cast<uint32_t>(degree));
        for (size_t slot = 0; slot < degree; ++slot) {
            hnswlib::edgeQuantV0Sha256UpdateUint32LittleEndian(
                sha, static_cast<uint32_t>(ids[slot]));
        }
    }
    return hnswlib::edgeQuantV0Sha256Hex(sha.final());
}

void testSha256GoldenVector() {
    hnswlib::EdgeQuantV0Sha256 sha;
    const char* input = "abc";
    sha.update(input, 3);
    require(
        hnswlib::edgeQuantV0Sha256Hex(sha.final()) ==
            "ba7816bf8f01cfea414140de5dae2223"
            "b00361a396177a9cb410ff61f20015ad",
        "SHA-256 golden vector mismatch");
}

void testEmptyGraph() {
    const size_t dimension = 8;
    hnswlib::L2Space space(dimension);
    hnswlib::HierarchicalNSW<float> index(&space, 32, 8, 40, 11);
    const hnswlib::V0Layer0GraphView view =
        index.getV0Layer0GraphView();

    require(view.nodeCount() == 0, "empty graph reports non-zero nodes");
    size_t edge_count = 0;
    view.forEachEdge(
        [&edge_count](hnswlib::tableint, hnswlib::tableint, size_t) {
            ++edge_count;
        });
    require(edge_count == 0, "empty graph callback reported an edge");
    requireOutOfRange(
        [&view]() { view.neighbors(0); },
        "empty graph accepted node zero");
}

void testGraphAccessAndReload() {
    const size_t dimension = 8;
    const size_t node_count = 96;
    std::vector<float> base(node_count * dimension);
    for (size_t node = 0; node < node_count; ++node) {
        for (size_t d = 0; d < dimension; ++d) {
            base[node * dimension + d] =
                static_cast<float>((node + 1U) * (d + 3U) % 37U) +
                static_cast<float>(node) * 0.001f;
        }
    }

    hnswlib::L2Space space(dimension);
    hnswlib::HierarchicalNSW<float> index(
        &space, node_count + 1U, 12, 80, 23);
    for (size_t node = 0; node < node_count; ++node) {
        index.addPoint(
            base.data() + node * dimension,
            static_cast<hnswlib::labeltype>(1000U + node));
    }

    const hnswlib::V0Layer0GraphView view =
        index.getV0Layer0GraphView();
    require(view.nodeCount() == node_count, "graph view node count mismatch");
    require(
        view.dataSize() == dimension * sizeof(float),
        "graph view data size mismatch");
    require(
        view.maxLayer0Degree() == index.maxM0_,
        "graph view layer-0 capacity mismatch");

    size_t raw_edge_count = 0;
    size_t callback_edge_count = 0;
    for (size_t source = 0; source < node_count; ++source) {
        hnswlib::linklistsizeint* raw =
            index.get_linklist0(static_cast<hnswlib::tableint>(source));
        const size_t raw_degree = index.getListCount(raw);
        const hnswlib::tableint* raw_ids =
            reinterpret_cast<const hnswlib::tableint*>(raw + 1);
        const hnswlib::V0Layer0NeighborSpan span =
            view.neighbors(static_cast<hnswlib::tableint>(source));

        require(span.size == raw_degree, "neighbor count mismatch");
        for (size_t slot = 0; slot < raw_degree; ++slot) {
            require(span[slot] == raw_ids[slot], "neighbor slot mismatch");
        }
        require(
            view.vectorData(static_cast<hnswlib::tableint>(source)) ==
                index.getDataByInternalId(
                    static_cast<hnswlib::tableint>(source)),
            "vector pointer mismatch");
        require(
            view.externalLabel(static_cast<hnswlib::tableint>(source)) ==
                index.getExternalLabel(
                    static_cast<hnswlib::tableint>(source)),
            "external label mismatch");
        raw_edge_count += raw_degree;
    }

    index.forEachV0Layer0Edge(
        [&view, &callback_edge_count](
            hnswlib::tableint source,
            hnswlib::tableint target,
            size_t slot) {
            require(
                view.neighbors(source)[slot] == target,
                "callback slot order mismatch");
            ++callback_edge_count;
        });
    require(
        callback_edge_count == raw_edge_count,
        "callback edge count mismatch");

    requireOutOfRange(
        [&view, node_count]() {
            view.neighbors(static_cast<hnswlib::tableint>(node_count));
        },
        "graph view accepted an invalid source id");
    if (view.neighbors(0).size != 0U) {
        requireOutOfRange(
            [&view]() { (void)view.neighbors(0)[view.neighbors(0).size]; },
            "neighbor span accepted an invalid slot");
    }

    const std::string fingerprint = view.adjacencyFingerprintHex();
    require(fingerprint.size() == 64U, "fingerprint has wrong length");
    require(
        fingerprint == rawAdjacencyFingerprint(index),
        "wrapper and raw adjacency fingerprints differ");
    require(
        fingerprint == index.getV0Layer0AdjacencyFingerprintHex(),
        "index fingerprint facade differs from graph view");

    const std::string saved_index_path =
        "v0_graph_access_test_index.bin";
    std::remove(saved_index_path.c_str());
    index.saveIndex(saved_index_path);
    {
        hnswlib::HierarchicalNSW<float> loaded(&space, saved_index_path);
        require(
            loaded.getV0Layer0AdjacencyFingerprintHex() == fingerprint,
            "adjacency fingerprint changed after save/load");
        require(
            loaded.getV0Layer0GraphView().nodeCount() == node_count,
            "reloaded graph view node count mismatch");
    }
    require(
        std::remove(saved_index_path.c_str()) == 0,
        "failed to remove graph access test index");
}

}  // namespace

int main() {
#ifndef HNSWLIB_ENABLE_EDGE_QUANT_V0
    std::cerr << "HNSWLIB_ENABLE_EDGE_QUANT_V0 is required" << std::endl;
    return 2;
#else
    testSha256GoldenVector();
    testEmptyGraph();
    testGraphAccessAndReload();
    std::cout << "v0_graph_access_test_ok" << std::endl;
    return 0;
#endif
}
