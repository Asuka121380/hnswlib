#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#include "hnswlib/hnswlib.h"

namespace {

struct Options {
    std::string index_path;
    std::string sidecar_path;
    std::string query_path;
    std::string output_path;
    std::string method;
    std::string prefetch = "legacy";
    size_t dimension = 0U;
    size_t query_start = 0U;
    size_t query_count = 0U;
    size_t k = 10U;
    size_t ef_search = 200U;
    size_t warmup_queries = 100U;
    size_t repeats = 5U;
    double beta = 0.0;
    int cpu = -1;
};

std::string requireValue(int& i, int argc, char** argv) {
    if (i + 1 >= argc) {
        throw std::invalid_argument("missing option value");
    }
    return argv[++i];
}

size_t parseSize(const std::string& value, const char* name) {
    size_t consumed = 0U;
    const unsigned long long parsed = std::stoull(value, &consumed);
    if (consumed != value.size() ||
        parsed > std::numeric_limits<size_t>::max()) {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return static_cast<size_t>(parsed);
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        if (key == "--index-path")
            options.index_path = requireValue(i, argc, argv);
        else if (key == "--sidecar-path")
            options.sidecar_path = requireValue(i, argc, argv);
        else if (key == "--query-path")
            options.query_path = requireValue(i, argc, argv);
        else if (key == "--output")
            options.output_path = requireValue(i, argc, argv);
        else if (key == "--method")
            options.method = requireValue(i, argc, argv);
        else if (key == "--prefetch")
            options.prefetch = requireValue(i, argc, argv);
        else if (key == "--dimension")
            options.dimension = parseSize(
                requireValue(i, argc, argv), "dimension");
        else if (key == "--query-start")
            options.query_start = parseSize(
                requireValue(i, argc, argv), "query-start");
        else if (key == "--query-count")
            options.query_count = parseSize(
                requireValue(i, argc, argv), "query-count");
        else if (key == "--k")
            options.k = parseSize(requireValue(i, argc, argv), "k");
        else if (key == "--ef-search")
            options.ef_search = parseSize(
                requireValue(i, argc, argv), "ef-search");
        else if (key == "--warmup-queries")
            options.warmup_queries = parseSize(
                requireValue(i, argc, argv), "warmup-queries");
        else if (key == "--repeats")
            options.repeats = parseSize(
                requireValue(i, argc, argv), "repeats");
        else if (key == "--beta")
            options.beta = std::stod(requireValue(i, argc, argv));
        else if (key == "--cpu")
            options.cpu = std::stoi(requireValue(i, argc, argv));
        else if (key == "--help") {
            std::cout
                << "v0_performance_runner --method baseline|approx-no-retry|approx-retry "
                << "--index-path FILE --query-path FILE --dimension N "
                << "--query-count N --output FILE [--sidecar-path FILE] "
                << "[--beta X] [--prefetch legacy|gate] [--query-start N] "
                << "[--k N] [--ef-search N] [--warmup-queries N] "
                << "[--repeats N] [--cpu N]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + key);
        }
    }

    const bool approximate = options.method == "approx-no-retry" ||
        options.method == "approx-retry";
    if (options.index_path.empty() || options.query_path.empty() ||
        options.output_path.empty() || options.dimension == 0U ||
        options.query_count == 0U || options.k == 0U ||
        options.ef_search == 0U || options.repeats < 5U ||
        (options.method != "baseline" && !approximate)) {
        throw std::invalid_argument("incomplete performance-runner contract");
    }
    if (approximate &&
        (options.sidecar_path.empty() ||
         !std::isfinite(options.beta) || options.beta < 1.0)) {
        throw std::invalid_argument(
            "approximate method requires sidecar and beta >= 1");
    }
    if (options.prefetch != "legacy" && options.prefetch != "gate") {
        throw std::invalid_argument("prefetch must be legacy or gate");
    }
    return options;
}

std::vector<float> loadFvecs(const Options& options) {
    std::ifstream input(options.query_path.c_str(), std::ios::binary);
    if (!input) throw std::runtime_error("cannot open query fvecs");
    const uint64_t row_bytes = sizeof(int32_t) +
        static_cast<uint64_t>(options.dimension) * sizeof(float);
    input.seekg(
        static_cast<std::streamoff>(
            static_cast<uint64_t>(options.query_start) * row_bytes),
        std::ios::beg);
    if (!input) throw std::runtime_error("cannot seek query fvecs");

    std::vector<float> queries(
        options.query_count * options.dimension);
    for (size_t row = 0U; row < options.query_count; ++row) {
        int32_t dimension = 0;
        input.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
        if (!input || dimension != static_cast<int32_t>(options.dimension)) {
            throw std::runtime_error("query fvecs dimension mismatch");
        }
        input.read(
            reinterpret_cast<char*>(
                queries.data() + row * options.dimension),
            static_cast<std::streamsize>(
                options.dimension * sizeof(float)));
        if (!input) throw std::runtime_error("truncated query fvecs");
    }
    return queries;
}

void pinCurrentThread(int cpu) {
    if (cpu < 0) return;
#ifdef _WIN32
    if (cpu >= static_cast<int>(sizeof(DWORD_PTR) * 8U) ||
        SetThreadAffinityMask(
            GetCurrentThread(),
            static_cast<DWORD_PTR>(1) << cpu) == 0) {
        throw std::runtime_error("failed to set CPU affinity");
    }
#else
    (void)cpu;
    throw std::runtime_error(
        "--cpu is currently implemented only for Windows");
#endif
}

uint64_t mixChecksum(
    uint64_t hash,
    const std::priority_queue<
        std::pair<float, hnswlib::labeltype> >& source) {
    std::priority_queue<
        std::pair<float, hnswlib::labeltype> > queue(source);
    while (!queue.empty()) {
        const std::pair<float, hnswlib::labeltype> value = queue.top();
        queue.pop();
        uint32_t distance_bits = 0U;
        std::memcpy(&distance_bits, &value.first, sizeof(distance_bits));
        const uint64_t words[] = {
            static_cast<uint64_t>(distance_bits),
            static_cast<uint64_t>(value.second)
        };
        for (size_t i = 0U; i < 2U; ++i) {
            hash ^= words[i];
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

uint64_t percentile(
    const std::vector<uint64_t>& sorted,
    double fraction) {
    if (sorted.empty()) return 0U;
    const size_t index = static_cast<size_t>(
        fraction * static_cast<double>(sorted.size() - 1U));
    return sorted[index];
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        pinCurrentThread(options.cpu);
        const std::chrono::steady_clock::time_point query_load_start =
            std::chrono::steady_clock::now();
        const std::vector<float> queries = loadFvecs(options);
        const std::chrono::steady_clock::time_point query_load_end =
            std::chrono::steady_clock::now();
        hnswlib::L2Space space(options.dimension);
        const std::chrono::steady_clock::time_point index_load_start =
            std::chrono::steady_clock::now();
        hnswlib::HierarchicalNSW<float> index(
            &space, options.index_path);
        const std::chrono::steady_clock::time_point index_load_end =
            std::chrono::steady_clock::now();
        index.setEf(options.ef_search);

        const bool approximate = options.method != "baseline";
        hnswlib::V0ApproxPruningConfig config;
        uint64_t sidecar_load_ns = 0U;
        if (approximate) {
#ifndef HNSWLIB_ENABLE_V0_APPROX_REAL_PRUNING
            throw std::runtime_error(
                "binary lacks V0 approximate performance support");
#else
            const std::chrono::steady_clock::time_point sidecar_load_start =
                std::chrono::steady_clock::now();
            index.loadEdgeQuantV0Metadata(options.sidecar_path);
            const std::chrono::steady_clock::time_point sidecar_load_end =
                std::chrono::steady_clock::now();
            sidecar_load_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    sidecar_load_end - sidecar_load_start).count());
            config.beta = options.beta;
            config.retry_enabled = options.method == "approx-retry";
            config.prefetch_policy = options.prefetch == "gate" ?
                hnswlib::V0ApproxPrefetchPolicy::GateAware :
                hnswlib::V0ApproxPrefetchPolicy::LegacyVector;
#endif
        }

        const auto run_query = [&index, &options, &config, approximate](
            const float* query) {
            if (!approximate) return index.searchKnn(query, options.k);
#ifdef HNSWLIB_ENABLE_V0_APPROX_REAL_PRUNING
            return index.searchKnnV0ApproxFast(query, options.k, config);
#else
            return index.searchKnn(query, options.k);
#endif
        };

        uint64_t checksum = 1469598103934665603ULL;
        const size_t warmup = std::min(
            options.warmup_queries, options.query_count);
        for (size_t i = 0U; i < warmup; ++i) {
            checksum = mixChecksum(
                checksum,
                run_query(queries.data() + i * options.dimension));
        }

        std::vector<uint64_t> latencies;
        latencies.reserve(options.repeats * options.query_count);
        uint64_t total_ns = 0U;
        for (size_t repeat = 0U; repeat < options.repeats; ++repeat) {
            for (size_t i = 0U; i < options.query_count; ++i) {
                const float* query =
                    queries.data() + i * options.dimension;
                const std::chrono::steady_clock::time_point start =
                    std::chrono::steady_clock::now();
                const std::priority_queue<
                    std::pair<float, hnswlib::labeltype> > result =
                        run_query(query);
                const std::chrono::steady_clock::time_point end =
                    std::chrono::steady_clock::now();
                const uint64_t elapsed = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        end - start).count());
                total_ns += elapsed;
                latencies.push_back(elapsed);
                checksum = mixChecksum(checksum, result);
            }
        }
        std::sort(latencies.begin(), latencies.end());
        const size_t measured_queries =
            options.repeats * options.query_count;
        const double qps = total_ns == 0U ? 0.0 :
            static_cast<double>(measured_queries) * 1e9 /
                static_cast<double>(total_ns);
        const uint64_t query_load_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                query_load_end - query_load_start).count());
        const uint64_t index_load_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                index_load_end - index_load_start).count());

        std::ofstream output(options.output_path.c_str());
        if (!output) throw std::runtime_error("cannot create output JSON");
        output << "{\n"
            << "  \"schema_version\": 1,\n"
            << "  \"kernel\": \"raw_fast_v1\",\n"
            << "  \"compiler\": \"" << __VERSION__ << "\",\n"
            << "  \"single_thread_query\": true,\n"
            << "  \"strict_fp_contract\": "
            << (HNSWLIB_V0_STRICT_FP_CONTRACT_ENABLED ? "true" : "false")
            << ",\n"
            << "  \"performance_comparable_flags\": "
            << (HNSWLIB_PERFORMANCE_COMPARABLE_FLAGS_ENABLED ? "true" : "false")
            << ",\n"
            << "  \"method\": \"" << options.method << "\",\n"
            << "  \"prefetch\": \"" << options.prefetch << "\",\n"
            << "  \"dimension\": " << options.dimension << ",\n"
            << "  \"query_start\": " << options.query_start << ",\n"
            << "  \"query_count\": " << options.query_count << ",\n"
            << "  \"warmup_queries\": " << warmup << ",\n"
            << "  \"repeats\": " << options.repeats << ",\n"
            << "  \"k\": " << options.k << ",\n"
            << "  \"ef_search\": " << options.ef_search << ",\n"
            << "  \"beta\": " << std::setprecision(17)
            << options.beta << ",\n"
            << "  \"cpu\": " << options.cpu << ",\n"
            << "  \"measured_queries\": " << measured_queries << ",\n"
            << "  \"query_load_ns\": " << query_load_ns << ",\n"
            << "  \"index_load_ns\": " << index_load_ns << ",\n"
            << "  \"sidecar_load_ns\": " << sidecar_load_ns << ",\n"
            << "  \"total_latency_ns\": " << total_ns << ",\n"
            << "  \"qps\": " << qps << ",\n"
            << "  \"latency_p50_ns\": "
            << percentile(latencies, 0.50) << ",\n"
            << "  \"latency_p95_ns\": "
            << percentile(latencies, 0.95) << ",\n"
            << "  \"latency_p99_ns\": "
            << percentile(latencies, 0.99) << ",\n"
            << "  \"result_checksum\": \"" << std::hex
            << checksum << std::dec << "\",\n"
            << "  \"latency_samples_ns\": [";
        for (size_t i = 0U; i < latencies.size(); ++i) {
            if (i != 0U) output << ',';
            output << latencies[i];
        }
        output << "]\n"
            << "}\n";
        if (!output) throw std::runtime_error("failed to write output JSON");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 2;
    }
}
