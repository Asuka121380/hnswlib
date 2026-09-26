#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hnswlib/hnswlib.h"
#include "hnswlib/edge_estimation/observer.h"
#include "capture_observer.h"

namespace {

std::string sha256File(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("failed to hash file: " + path.string());
    hnswlib::EdgeQuantV0Sha256 sha;
    char buffer[1U << 16U];
    while (input) {
        input.read(buffer, sizeof(buffer));
        const std::streamsize count = input.gcount();
        if (count > 0) sha.update(buffer, static_cast<size_t>(count));
    }
    if (!input.eof()) throw std::runtime_error("failed while hashing file");
    return hnswlib::edgeQuantV0Sha256Hex(sha.final());
}

std::vector<uint64_t> readQueryIds(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open query-id file");
    std::vector<uint64_t> ids;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        std::istringstream parser(line);
        uint64_t id = 0U;
        std::string extra;
        if (!(parser >> id) || (parser >> extra))
            throw std::runtime_error("invalid query-id file");
        ids.push_back(id);
    }
    if (ids.empty()) throw std::runtime_error("query-id file is empty");
    return ids;
}

std::vector<float> readFvec(
    const std::filesystem::path& path, uint64_t row, uint32_t dimension) {
    const uint64_t stride = sizeof(uint32_t) +
        static_cast<uint64_t>(dimension) * sizeof(float);
    const uint64_t size = std::filesystem::file_size(path);
    if (stride == 0U || size % stride != 0U || row >= size / stride)
        throw std::runtime_error("query fvecs size or row is invalid");
    std::ifstream input(path, std::ios::binary);
    input.seekg(static_cast<std::streamoff>(row * stride));
    uint32_t stored_dimension = 0U;
    input.read(reinterpret_cast<char*>(&stored_dimension), sizeof(stored_dimension));
    if (!input || stored_dimension != dimension)
        throw std::runtime_error("query fvecs dimension mismatch");
    std::vector<float> vector(dimension);
    input.read(reinterpret_cast<char*>(vector.data()),
               static_cast<std::streamsize>(dimension * sizeof(float)));
    if (!input) throw std::runtime_error("truncated query fvecs");
    return vector;
}

void writeCatalog(
    const std::filesystem::path& root,
    const hnswlib::edge_estimation::EdgeCatalog& catalog) {
    const std::filesystem::path directory = root / "edge_catalog";
    std::filesystem::create_directories(directory);
    const std::filesystem::path offsets = directory / "source_offsets.u64le";
    const std::filesystem::path targets = directory / "targets.u32le";
    catalog.writeBinaryFiles(offsets.string(), targets.string());
    std::ofstream manifest((directory / "manifest.json").string());
    manifest << "{\"schema_version\":1,\"layer\":0,\"node_count\":"
             << catalog.nodeCount() << ",\"edge_count\":" << catalog.edgeCount()
             << ",\"identity_kind\":\"uqcat_v1_sha256\",\"identity_sha256\":\""
             << hnswlib::edgeQuantV0Sha256Hex(catalog.identityDigest())
             << "\",\"adjacency_sha256\":\""
             << hnswlib::edgeQuantV0Sha256Hex(catalog.adjacencyDigest())
             << "\",\"files\":{\"source_offsets.u64le\":{\"size\":"
             << std::filesystem::file_size(offsets) << ",\"sha256\":\""
             << sha256File(offsets) << "\"},\"targets.u32le\":{\"size\":"
             << std::filesystem::file_size(targets) << ",\"sha256\":\""
             << sha256File(targets) << "\"}}}\n";
    if (!manifest) throw std::runtime_error("failed to write catalog manifest");
}

int catalogReal(const std::filesystem::path& index_path, uint32_t dimension,
                const std::filesystem::path& output) {
    if (dimension == 0U || std::filesystem::exists(output))
        throw std::invalid_argument("catalog output exists or dimension is zero");
    const std::filesystem::path partial(output.string() + ".partial");
    if (std::filesystem::exists(partial)) throw std::runtime_error("catalog partial output exists");
    hnswlib::L2Space space(dimension);
    hnswlib::HierarchicalNSW<float> index(&space, index_path.string());
    const hnswlib::edge_estimation::EdgeCatalog catalog =
        hnswlib::edge_estimation::EdgeCatalog::fromLayer0Graph(index.getV0Layer0GraphView());
    std::filesystem::create_directories(partial);
    // writeCatalog creates an edge_catalog child for dataset use; promote its files here.
    writeCatalog(partial, catalog);
    const std::filesystem::path nested = partial / "edge_catalog";
    for (const auto& entry : std::filesystem::directory_iterator(nested))
        std::filesystem::rename(entry.path(), partial / entry.path().filename());
    std::filesystem::remove(nested);
    std::ofstream complete((partial / "complete.json").string());
    complete << "{\"schema_version\":1,\"stage\":\"catalog\",\"index_sha256\":\""
             << sha256File(index_path) << "\"}\n"; complete.close();
    std::filesystem::rename(partial, output);
    std::cout << "{\"node_count\":" << catalog.nodeCount()
              << ",\"edge_count\":" << catalog.edgeCount() << "}\n";
    return 0;
}

int captureReal(
    const std::filesystem::path& index_path,
    const std::filesystem::path& queries_path,
    const std::filesystem::path& query_ids_path,
    uint32_t dimension,
    size_t k,
    size_t ef,
    const std::filesystem::path& output) {
    if (dimension == 0U || k == 0U || ef < k)
        throw std::invalid_argument("capture dimension/k/ef are invalid");
    if (std::filesystem::exists(output))
        throw std::runtime_error("capture output already exists");
    const std::filesystem::path partial(output.string() + ".partial");
    if (std::filesystem::exists(partial))
        throw std::runtime_error("capture partial output already exists");
    const std::vector<uint64_t> query_ids = readQueryIds(query_ids_path);
    hnswlib::L2Space space(dimension);
    hnswlib::HierarchicalNSW<float> index(&space, index_path.string());
    index.setEf(ef);
    const hnswlib::edge_estimation::EdgeCatalog catalog =
        hnswlib::edge_estimation::EdgeCatalog::fromLayer0Graph(
            index.getV0Layer0GraphView());
    uq::DatasetCaptureObserver observer(catalog, query_ids);
    for (size_t i = 0; i < query_ids.size(); ++i) {
        const std::vector<float> query = readFvec(queries_path, query_ids[i], dimension);
        hnswlib::edge_estimation::ScopedCaptureObserver scope(&observer);
        (void)index.searchKnn(query.data(), k);
    }
    std::filesystem::create_directories(partial);
    writeCatalog(partial, catalog);
    const std::filesystem::path events = partial / "events.bin";
    const std::filesystem::path labels = partial / "labels.bin";
    const std::filesystem::path ranges = partial / "query_ranges.bin";
    observer.write(events.string(), labels.string(), ranges.string(), dimension);
    const std::filesystem::path manifest_path = partial / "manifest.json";
    std::ofstream manifest(manifest_path.string());
    manifest << "{\"schema_version\":1,\"capture_mode\":\"exact_baseline\","
             << "\"source_search_method\":\"hnsw_layer0\",\"dimension\":" << dimension
             << ",\"k\":" << k << ",\"ef\":" << ef
             << ",\"query_count\":" << query_ids.size()
             << ",\"event_count\":" << observer.events().size()
             << ",\"label_count\":" << observer.labels().size()
             << ",\"index_sha256\":\"" << sha256File(index_path)
             << "\",\"queries_sha256\":\"" << sha256File(queries_path)
             << "\",\"query_ids_sha256\":\"" << sha256File(query_ids_path)
             << "\",\"capabilities\":{\"ordered_timing\":true},"
             << "\"coverage\":\"subset\",\"files\":{"
             << "\"events.bin\":{\"size\":" << std::filesystem::file_size(events)
             << ",\"sha256\":\"" << sha256File(events) << "\"},"
             << "\"labels.bin\":{\"size\":" << std::filesystem::file_size(labels)
             << ",\"sha256\":\"" << sha256File(labels) << "\"},"
             << "\"query_ranges.bin\":{\"size\":" << std::filesystem::file_size(ranges)
             << ",\"sha256\":\"" << sha256File(ranges) << "\"}}}\n";
    if (!manifest) throw std::runtime_error("failed to write capture manifest");
    manifest.close();
    std::ofstream complete((partial / "complete.json").string());
    complete << "{\"schema_version\":1,\"stage\":\"capture\",\"manifest_sha256\":\""
             << sha256File(manifest_path) << "\"}\n";
    complete.close();
    std::filesystem::rename(partial, output);
    std::cout << "{\"event_count\":" << observer.events().size()
              << ",\"label_count\":" << observer.labels().size()
              << ",\"query_count\":" << query_ids.size() << "}\n";
    return 0;
}

int synthetic(const std::filesystem::path& output) {
    if (std::filesystem::exists(output))
        throw std::runtime_error("capture output already exists");
    std::filesystem::path partial_output(output.string() + ".partial");
    if (std::filesystem::exists(partial_output))
        throw std::runtime_error("capture partial output already exists");
    const size_t dimension = 8U;
    const size_t point_count = 128U;
    const size_t query_count = 4U;
    std::mt19937 rng(20260924U);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> points(point_count * dimension);
    for (size_t i = 0; i < points.size(); ++i) points[i] = normal(rng);
    hnswlib::L2Space space(dimension);
    hnswlib::HierarchicalNSW<float> index(
        &space, point_count, 12U, 80U, 20260924U);
    for (size_t i = 0; i < point_count; ++i)
        index.addPoint(&points[i * dimension], static_cast<hnswlib::labeltype>(5000U + i));
    index.setEf(40U);
    const hnswlib::edge_estimation::EdgeCatalog catalog =
        hnswlib::edge_estimation::EdgeCatalog::fromLayer0Graph(
            index.getV0Layer0GraphView());
    std::vector<uint64_t> query_ids;
    for (size_t i = 0; i < query_count; ++i) query_ids.push_back(100U + i);
    uq::DatasetCaptureObserver observer(catalog, query_ids);
    for (size_t i = 0; i < query_count; ++i) {
        hnswlib::edge_estimation::ScopedCaptureObserver scope(&observer);
        (void)index.searchKnn(&points[(20U + i) * dimension], 10U);
    }
    std::filesystem::create_directories(partial_output);
    const std::filesystem::path catalog_directory = partial_output / "edge_catalog";
    std::filesystem::create_directories(catalog_directory);
    catalog.writeBinaryFiles(
        (catalog_directory / "source_offsets.u64le").string(),
        (catalog_directory / "targets.u32le").string());
    std::ofstream catalog_manifest((catalog_directory / "manifest.json").string());
    catalog_manifest << "{\n"
                     << "  \"schema_version\": 1,\n"
                     << "  \"layer\": 0,\n"
                     << "  \"node_count\": " << catalog.nodeCount() << ",\n"
                     << "  \"edge_count\": " << catalog.edgeCount() << ",\n"
                     << "  \"identity_kind\": \"uqcat_v1_sha256\",\n"
                     << "  \"identity_sha256\": \""
                     << hnswlib::edgeQuantV0Sha256Hex(catalog.identityDigest())
                     << "\",\n"
                     << "  \"adjacency_sha256\": \""
                     << hnswlib::edgeQuantV0Sha256Hex(catalog.adjacencyDigest())
                     << "\"\n"
                     << "}\n";
    if (!catalog_manifest) throw std::runtime_error("failed to write catalog manifest");
    catalog_manifest.close();
    const std::filesystem::path events_path = partial_output / "events.bin";
    const std::filesystem::path labels_path = partial_output / "labels.bin";
    const std::filesystem::path ranges_path = partial_output / "query_ranges.bin";
    observer.write(events_path.string(), labels_path.string(), ranges_path.string(),
                   static_cast<uint32_t>(dimension));
    const std::filesystem::path manifest_path = partial_output / "manifest.json";
    std::ofstream manifest(manifest_path.string());
    manifest << "{\n"
             << "  \"schema_version\": 1,\n"
             << "  \"capture_mode\": \"exact_baseline_synthetic\",\n"
             << "  \"source_search_method\": \"hnsw_layer0\",\n"
             << "  \"dimension\": " << dimension << ",\n"
             << "  \"event_count\": " << observer.events().size() << ",\n"
             << "  \"label_count\": " << observer.labels().size() << ",\n"
             << "  \"query_count\": " << observer.ranges().size() << ",\n"
             << "  \"capabilities\": {\"ordered_timing\": true},\n"
             << "  \"coverage\": \"subset\",\n"
             << "  \"formal_result_eligible\": false,\n"
             << "  \"files\": {\n"
             << "    \"events.bin\": {\"size\": " << std::filesystem::file_size(events_path)
             << ", \"sha256\": \"" << sha256File(events_path) << "\"},\n"
             << "    \"labels.bin\": {\"size\": " << std::filesystem::file_size(labels_path)
             << ", \"sha256\": \"" << sha256File(labels_path) << "\"},\n"
             << "    \"query_ranges.bin\": {\"size\": " << std::filesystem::file_size(ranges_path)
             << ", \"sha256\": \"" << sha256File(ranges_path) << "\"}\n"
             << "  }\n"
             << "}\n";
    if (!manifest) throw std::runtime_error("failed to write capture manifest");
    manifest.close();
    const std::filesystem::path complete_path = partial_output / "complete.json";
    std::ofstream complete(complete_path.string());
    complete << "{\"schema_version\":1,\"stage\":\"capture\","
             << "\"manifest\":{\"path\":\"manifest.json\",\"size\":"
             << std::filesystem::file_size(manifest_path)
             << ",\"sha256\":\"" << sha256File(manifest_path) << "\"}}\n";
    if (!complete) throw std::runtime_error("failed to write capture completion marker");
    complete.close();
    std::filesystem::rename(partial_output, output);
    std::cout << "{\"event_count\":" << observer.events().size()
              << ",\"label_count\":" << observer.labels().size()
              << ",\"query_count\":" << observer.ranges().size() << "}\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--synthetic")
            return synthetic(std::filesystem::path(argv[2]));
        if (argc == 7 && std::string(argv[1]) == "--catalog" &&
            std::string(argv[2]) == "--index" && std::string(argv[4]) == "--dimension")
            return catalogReal(argv[3], static_cast<uint32_t>(std::stoul(argv[5])), argv[6]);
        if (argc == 15 && std::string(argv[1]) == "--index" &&
            std::string(argv[3]) == "--queries" &&
            std::string(argv[5]) == "--query-ids" &&
            std::string(argv[7]) == "--dimension" &&
            std::string(argv[9]) == "--k" && std::string(argv[11]) == "--ef" &&
            std::string(argv[13]) == "--out") {
            return captureReal(argv[2], argv[4], argv[6],
                static_cast<uint32_t>(std::stoul(argv[8])),
                static_cast<size_t>(std::stoull(argv[10])),
                static_cast<size_t>(std::stoull(argv[12])), argv[14]);
        }
        std::cerr << "usage:\n  uq_capture --synthetic OUT\n"
                  << "  uq_capture --catalog --index INDEX --dimension D OUT\n"
                  << "  uq_capture --index INDEX --queries FVEC --query-ids IDS "
                  << "--dimension D --k K --ef EF --out OUT\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "uq_capture: " << error.what() << std::endl;
        return 1;
    }
}
