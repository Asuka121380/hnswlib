#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

#include "hnswlib/hnswlib.h"
#include "hnswlib/edge_estimation/active_policy.h"
#include "hnswlib/edge_estimation/graph_catalog.h"
#include "../../tools/edge_estimation/event_format.h"
#include "../../tools/edge_estimation/live_opq_pruner.h"
#include "../../tools/edge_estimation/backends/ivf_edge.h"
#include "../../tools/edge_estimation/query_store.h"

namespace {

uint64_t peakRssBytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS memory{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &memory, sizeof(memory))) return 0;
    return memory.PeakWorkingSetSize;
#else
    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage)) return 0;
#ifdef __APPLE__
    return usage.ru_maxrss;
#else
    return static_cast<uint64_t>(usage.ru_maxrss)*1024U;
#endif
#endif
}

struct Options {
    std::string index_path;
    std::string artifact_path;
    std::string query_path;
    std::string output_path;
    std::string latency_records_path;
    std::string result_records_path;
    size_t dimension = 0U;
    size_t query_start = 0U;
    size_t query_count = 0U;
    size_t k = 10U;
    size_t ef_search = 200U;
    size_t batch_size = 128U;
    size_t warmup_queries = 100U;
    size_t repeats = 1U;
    double beta = 1.0;
    bool no_prune = false;
};

std::string requireValue(int& i, int argc, char** argv) {
    if (i + 1 >= argc) throw std::invalid_argument("missing option value");
    return argv[++i];
}

size_t parseSize(const std::string& value, const char* name) {
    size_t consumed = 0U;
    const unsigned long long parsed = std::stoull(value, &consumed);
    if (value.empty() || value[0]=='-' || consumed != value.size() || parsed > std::numeric_limits<size_t>::max())
        throw std::invalid_argument(std::string("invalid ") + name);
    return static_cast<size_t>(parsed);
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        if (key == "--index-path") options.index_path = requireValue(i, argc, argv);
        else if (key == "--artifact-path") options.artifact_path = requireValue(i, argc, argv);
        else if (key == "--query-path") options.query_path = requireValue(i, argc, argv);
        else if (key == "--output") options.output_path = requireValue(i, argc, argv);
        else if (key == "--latency-records-output") options.latency_records_path = requireValue(i, argc, argv);
        else if (key == "--result-records-output") options.result_records_path = requireValue(i, argc, argv);
        else if (key == "--dimension") options.dimension = parseSize(requireValue(i, argc, argv), "dimension");
        else if (key == "--query-start") options.query_start = parseSize(requireValue(i, argc, argv), "query-start");
        else if (key == "--query-count") options.query_count = parseSize(requireValue(i, argc, argv), "query-count");
        else if (key == "--k") options.k = parseSize(requireValue(i, argc, argv), "k");
        else if (key == "--ef-search") options.ef_search = parseSize(requireValue(i, argc, argv), "ef-search");
        else if (key == "--batch-size") options.batch_size = parseSize(requireValue(i, argc, argv), "batch-size");
        else if (key == "--warmup-queries") options.warmup_queries = parseSize(requireValue(i, argc, argv), "warmup-queries");
        else if (key == "--repeats") options.repeats = parseSize(requireValue(i, argc, argv), "repeats");
        else if (key == "--no-prune") options.no_prune = true;
        else if (key == "--beta") options.beta = std::stod(requireValue(i, argc, argv));
        else if (key == "--help") {
            std::cout << "uq_ivf_performance_runner --index-path FILE --artifact-path DIR "
                      << "--query-path FILE --dimension N --query-count N --output FILE "
                      << "[--query-start N] [--k N] [--ef-search N] [--beta X] "
                      << "[--batch-size N] [--warmup-queries N] [--repeats N] [--no-prune] "
                      << "[--latency-records-output FILE] [--result-records-output FILE]\n";
            std::exit(0);
        } else throw std::invalid_argument("unknown option: " + key);
    }
    if (options.index_path.empty() || options.artifact_path.empty() ||
        options.query_path.empty() || options.output_path.empty() ||
        options.dimension == 0U || options.query_count == 0U || options.k == 0U ||
        options.ef_search == 0U || options.batch_size == 0U || options.repeats == 0U ||
        !std::isfinite(options.beta) || options.beta < 0.0)
        throw std::invalid_argument("incomplete IVF performance-runner contract");
    return options;
}

uint64_t mixChecksum(uint64_t hash, const std::priority_queue<
                     std::pair<float, hnswlib::labeltype> >& source) {
    std::priority_queue<std::pair<float, hnswlib::labeltype> > queue(source);
    while (!queue.empty()) {
        const std::pair<float, hnswlib::labeltype> value = queue.top();
        queue.pop();
        uint32_t distance_bits = 0U;
        std::memcpy(&distance_bits, &value.first, sizeof(distance_bits));
        const uint64_t words[] = {distance_bits, static_cast<uint64_t>(value.second)};
        for (size_t i = 0; i < 2U; ++i) {
            hash ^= words[i];
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

uint64_t percentile(const std::vector<uint64_t>& sorted, double fraction) {
    if (sorted.empty()) return 0U;
    return sorted[static_cast<size_t>(fraction * (sorted.size() - 1U))];
}

std::vector<uint64_t> queryIds(size_t begin, size_t count) {
    std::vector<uint64_t> ids(count);
    for (size_t i = 0; i < count; ++i) ids[i] = begin + i;
    return ids;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
#ifndef HNSWLIB_ENABLE_EDGE_ESTIMATION_ACTIVE
        throw std::runtime_error("binary lacks unified active-pruning support");
#elif !defined(HNSWLIB_ENABLE_EDGE_QUANT_V0)
        throw std::runtime_error("binary lacks layer-0 graph catalog support");
#else
        const std::shared_ptr<const uq::QueryStore> queries(
            new uq::QueryStore(options.query_path, static_cast<uint32_t>(options.dimension)));
        if (options.query_start > queries->count() || options.query_count > queries->count()-options.query_start)
            throw std::invalid_argument("query range exceeds fvecs row count");

        hnswlib::L2Space space(options.dimension);
        hnswlib::HierarchicalNSW<float> index(&space, options.index_path);
        index.setEf(options.ef_search);
        hnswlib::edge_estimation::EdgeCatalog catalog;
        if (!options.no_prune)
            catalog=hnswlib::edge_estimation::EdgeCatalog::fromLayer0Graph(index.getV0Layer0GraphView());
        uq::Header header{};
        header.dimension = static_cast<uint32_t>(options.dimension);
        if (!options.no_prune) header.identity = catalog.identityDigest();
        std::unique_ptr<uq::IvfEdgeArtifactKernel> kernel;
        std::unique_ptr<uq::LiveArtifactPruner<uq::IvfEdgeArtifactKernel>> pruner;
        if (!options.no_prune) {
            kernel.reset(new uq::IvfEdgeArtifactKernel(options.artifact_path, queries, header));
            kernel->verifyIndex(options.index_path);
            pruner.reset(new uq::LiveArtifactPruner<uq::IvfEdgeArtifactKernel>(*kernel, catalog, options.beta));
        }
        const std::vector<uint64_t> ids = queryIds(options.query_start, options.query_count);
        uint64_t checksum = 1469598103934665603ULL;
        const size_t warmup = std::min(options.warmup_queries, options.query_count);
        if (warmup != 0U) {
          for (size_t first=0; first<warmup; first+=options.batch_size) {
            const size_t last=std::min(warmup,first+options.batch_size);
            const std::vector<uint64_t> warmup_ids(ids.begin()+first, ids.begin()+last);
            if (!options.no_prune) kernel->prepareQueryBatch(warmup_ids, options.batch_size);
            for (size_t i = 0; i < warmup_ids.size(); ++i) {
                if (!options.no_prune) pruner->beginQuery(warmup_ids[i]);
                hnswlib::edge_estimation::ScopedActiveEdgePruner active(pruner.get());
                checksum = mixChecksum(checksum, index.searchKnn(queries->query(warmup_ids[i]), options.k));
            }
          }
        }
        if (pruner) pruner->resetMetrics();

        std::vector<uint64_t> latencies;
        std::vector<uint64_t> search_latencies;
        std::vector<uint64_t> batch_completion_latencies;
        std::vector<std::priority_queue<std::pair<float,hnswlib::labeltype>>> saved_results(options.query_count);
        latencies.reserve(options.repeats * options.query_count);
        search_latencies.reserve(options.repeats * options.query_count);
        uint64_t total_ns = 0U;
        uint64_t total_prepare_ns = 0U;
        for (size_t repeat = 0; repeat < options.repeats; ++repeat) {
          const size_t block_size = options.no_prune ? 1U : options.batch_size;
          for (size_t first=0; first<ids.size(); first+=block_size) {
            const size_t last=std::min(ids.size(), first+block_size);
            const std::vector<uint64_t> block(ids.begin()+first, ids.begin()+last);
            const std::chrono::steady_clock::time_point prepare_start = std::chrono::steady_clock::now();
            if (!options.no_prune) kernel->prepareQueryBatch(block, options.batch_size);
            const std::chrono::steady_clock::time_point prepare_end = std::chrono::steady_clock::now();
            const uint64_t prepare_ns = options.no_prune ? 0U : static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(prepare_end - prepare_start).count());
            total_prepare_ns += prepare_ns;
            total_ns += prepare_ns;
            const uint64_t amortized_prepare_ns = prepare_ns / block.size();
            uint64_t batch_completion_ns = prepare_ns;
            for (size_t i = first; i < last; ++i) {
                const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
                if (!options.no_prune) pruner->beginQuery(ids[i]);
                std::priority_queue<std::pair<float, hnswlib::labeltype> > result;
                {
                    hnswlib::edge_estimation::ScopedActiveEdgePruner active(pruner.get());
                    result = index.searchKnn(queries->query(ids[i]), options.k);
                }
                const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                const uint64_t search_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
                total_ns += search_ns;
                search_latencies.push_back(search_ns);
                latencies.push_back(search_ns + amortized_prepare_ns);
                batch_completion_ns += search_ns;
                batch_completion_latencies.push_back(batch_completion_ns);
                checksum = mixChecksum(checksum, result);
                if (repeat==0 && !options.result_records_path.empty()) saved_results[i]=result;
            }
          }
        }
        const uq::LiveOpqMetrics measured_metrics = pruner ? pruner->metrics() : uq::LiveOpqMetrics{};

        if (!options.latency_records_path.empty()) {
            std::ofstream records(options.latency_records_path.c_str());
            if (!records) throw std::runtime_error("cannot create latency records output");
            records << "repeat_id,query_id,latency_ns,search_latency_ns,batch_completion_service_ns\n";
            for (size_t sample = 0; sample < latencies.size(); ++sample)
                records << sample / options.query_count << ','
                        << options.query_start + sample % options.query_count << ','
                        << latencies[sample] << ',' << search_latencies[sample] << ',' << batch_completion_latencies[sample] << '\n';
        }

        if (!options.result_records_path.empty()) {
            std::ofstream records(options.result_records_path.c_str());
            if (!records) throw std::runtime_error("cannot create result records output");
            records << "query_id,rank,label,distance\n" << std::setprecision(17);
            for (size_t i = 0; i < options.query_count; ++i) {
                auto result = saved_results[i];
                std::vector<std::pair<float, hnswlib::labeltype> > ordered;
                while (!result.empty()) { ordered.push_back(result.top()); result.pop(); }
                std::reverse(ordered.begin(), ordered.end());
                for (size_t rank = 0; rank < ordered.size(); ++rank)
                    records << ids[i] << ',' << rank << ',' << ordered[rank].second
                            << ',' << ordered[rank].first << '\n';
            }
        }

        std::sort(latencies.begin(), latencies.end());
        std::sort(search_latencies.begin(), search_latencies.end());
        std::sort(batch_completion_latencies.begin(), batch_completion_latencies.end());
        const size_t measured_queries = options.repeats * options.query_count;
        const double qps = total_ns == 0U ? 0.0 : measured_queries * 1e9 / total_ns;
        std::ofstream output(options.output_path.c_str());
        if (!output) throw std::runtime_error("cannot create output JSON");
        output << "{\n"
               << "  \"schema_version\": 1,\n"
               << "  \"latency_semantics\": \"search_plus_amortized_preparation_not_request_latency\",\n"
               << "  \"kernel\": \"live_ivf_edge_batch_v1\",\n"
               << "  \"method\": \"" << (options.no_prune ? "hnsw" : kernel->method()) << "\",\n"
               << "  \"dimension\": " << options.dimension << ",\n"
               << "  \"query_start\": " << options.query_start << ",\n"
               << "  \"query_count\": " << options.query_count << ",\n"
               << "  \"warmup_queries\": " << warmup << ",\n"
               << "  \"repeats\": " << options.repeats << ",\n"
               << "  \"k\": " << options.k << ",\n"
               << "  \"ef_search\": " << options.ef_search << ",\n"
               << "  \"beta\": " << std::setprecision(17) << options.beta << ",\n"
               << "  \"batch_size\": " << options.batch_size << ",\n"
               << "  \"batch_engine\": \"" << (kernel ? kernel->batchPreparationEngine() : "none") << "\",\n"
               << "  \"measured_queries\": " << measured_queries << ",\n"
               << "  \"total_latency_ns\": " << total_ns << ",\n"
               << "  \"batch_prepare_total_ns\": " << total_prepare_ns << ",\n"
               << "  \"batch_prepare_ns_per_query\": " << (total_prepare_ns / measured_queries) << ",\n"
               << "  \"qps\": " << qps << ",\n"
               << "  \"latency_p50_ns\": " << percentile(latencies, 0.50) << ",\n"
               << "  \"latency_p95_ns\": " << percentile(latencies, 0.95) << ",\n"
               << "  \"latency_p99_ns\": " << percentile(latencies, 0.99) << ",\n"
               << "  \"search_latency_p50_ns\": " << percentile(search_latencies, 0.50) << ",\n"
               << "  \"search_latency_p95_ns\": " << percentile(search_latencies, 0.95) << ",\n"
               << "  \"batch_completion_service_p95_ns\": " << percentile(batch_completion_latencies, 0.95) << ",\n"
               << "  \"backend_bytes\": " << (kernel ? kernel->backendBytes() : 0U) << ",\n"
               << "  \"scratch_bytes\": " << (kernel ? kernel->scratchBytes() : 0U) << ",\n"
               << "  \"peak_rss_bytes\": " << peakRssBytes() << ",\n"
               << "  \"catalog_payload_bytes\": " << (options.no_prune ? 0U : 8ULL*(catalog.nodeCount()+1)+4ULL*catalog.edgeCount()) << ",\n"
               << "  \"attempted_estimates\": " << measured_metrics.attempted << ",\n"
               << "  \"valid_estimates\": " << measured_metrics.valid << ",\n"
               << "  \"pruned_estimates\": " << measured_metrics.pruned << ",\n"
               << "  \"eligible_exact_distance_count\": " << (options.no_prune ? "null" : std::to_string(measured_metrics.attempted-measured_metrics.pruned)) << ",\n"
               << "  \"fallback_estimates\": " << measured_metrics.fallback << ",\n"
               << "  \"threshold_unavailable\": " << measured_metrics.threshold_unavailable << ",\n"
               << "  \"result_checksum\": \"" << std::hex << checksum << std::dec << "\",\n"
               << "  \"latency_records_written\": " << (!options.latency_records_path.empty() ? "true" : "false") << ",\n"
               << "  \"result_records_written\": " << (!options.result_records_path.empty() ? "true" : "false") << "\n"
               << "}\n";
        if (!output) throw std::runtime_error("failed to write output JSON");
        return 0;
#endif
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 2;
    }
}
