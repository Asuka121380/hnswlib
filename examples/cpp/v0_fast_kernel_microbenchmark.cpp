#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hnswlib/hnswlib.h"

namespace {

struct Options {
    std::string sidecar_path;
    std::string output_path;
    size_t iterations = 10000U;
};

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        if (key == "--help") {
            std::cout << "v0_fast_kernel_microbenchmark --sidecar-path FILE "
                      << "--output FILE [--iterations N]\n";
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::invalid_argument("missing value");
        const std::string value(argv[++i]);
        if (key == "--sidecar-path") options.sidecar_path = value;
        else if (key == "--output") options.output_path = value;
        else if (key == "--iterations")
            options.iterations = static_cast<size_t>(std::stoull(value));
        else throw std::invalid_argument("unknown option: " + key);
    }
    if (options.sidecar_path.empty() || options.output_path.empty() ||
        options.iterations == 0U) {
        throw std::invalid_argument("incomplete microbenchmark contract");
    }
    return options;
}

template<typename Callable>
double measureNsPerOperation(size_t iterations, Callable callable) {
    const std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();
    for (size_t i = 0U; i < iterations; ++i) callable(i);
    const std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    const uint64_t elapsed = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count());
    return static_cast<double>(elapsed) / static_cast<double>(iterations);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        const hnswlib::V0OwnedSidecar owned =
            hnswlib::loadV0Sidecar(options.sidecar_path);
        const hnswlib::V0SidecarView sidecar = owned.view();
        const hnswlib::V0SidecarHeader& header = sidecar.header();
        if (header.directed_edge_count == 0U) {
            throw std::runtime_error("sidecar has no edge records");
        }
        std::vector<float> native_codebook(
            static_cast<size_t>(header.codebook.size / sizeof(float)));
        for (size_t i = 0U; i < native_codebook.size(); ++i)
            native_codebook[i] = sidecar.codebookCentroid(i);
        std::vector<float> query(header.dimension);
        std::vector<float> candidate(header.dimension);
        for (size_t i = 0U; i < query.size(); ++i) {
            query[i] = static_cast<float>(
                static_cast<int>(i % 31U) - 15) * 0.03125f;
            candidate[i] = static_cast<float>(
                static_cast<int>(i % 17U) - 8) * 0.0625f;
        }
        const hnswlib::V0EdgeRecordView edge = sidecar.edgeRecord(0U);
        volatile double sink = 0.0;

        const size_t lut_iterations =
            std::max<size_t>(10U, options.iterations / 100U);
        const double strict_lut_ns = measureNsPerOperation(
            lut_iterations, [&](size_t) {
                const hnswlib::V0QueryLut lut(query.data(), sidecar);
                sink += static_cast<double>(lut.tableBytes());
            });
        const double fast_lut_ns = measureNsPerOperation(
            lut_iterations, [&](size_t) {
                const hnswlib::V0ApproxQueryLut lut(
                    query.data(), header, native_codebook.data());
                sink += static_cast<double>(lut.tableBytes());
            });

        const hnswlib::V0QueryLut strict_lut(query.data(), sidecar);
        const hnswlib::V0ApproxQueryLut fast_lut(
            query.data(), header, native_codebook.data());
        const double strict_lookup_ns = measureNsPerOperation(
            options.iterations, [&](size_t) {
                sink += strict_lut.innerProductUpper(edge);
            });
        const double fast_lookup_ns = measureNsPerOperation(
            options.iterations, [&](size_t) {
                sink += fast_lut.innerProductUnchecked(edge);
            });

        const hnswlib::EdgeQuantV0QueryContext strict_context(
            query.data(), sidecar);
        const hnswlib::EdgeQuantV0ApproxQueryContext fast_context(
            query.data(), header, native_codebook.data());
        const double strict_estimator_ns = measureNsPerOperation(
            options.iterations, [&](size_t) {
                sink += strict_context.evaluateRaw(edge, 1000.0)
                    .approximate_squared_distance;
            });
        const double fast_estimator_ns = measureNsPerOperation(
            options.iterations, [&](size_t) {
                sink += fast_context.evaluateRawFast(edge, 1000.0)
                    .approximate_squared_distance;
            });

        const double exact_l2_ns = measureNsPerOperation(
            options.iterations, [&](size_t) {
                float distance = 0.0f;
                for (size_t d = 0U; d < query.size(); ++d) {
                    const float delta = query[d] - candidate[d];
                    distance += delta * delta;
                }
                sink += distance;
            });
        const double checked_record_ns = measureNsPerOperation(
            options.iterations, [&](size_t i) {
                const size_t edge_index = i %
                    static_cast<size_t>(header.directed_edge_count);
                sink += sidecar.edgeRecord(edge_index).edgeLength();
            });
        const double direct_record_ns = measureNsPerOperation(
            options.iterations, [&](size_t i) {
                const size_t edge_index = i %
                    static_cast<size_t>(header.directed_edge_count);
                sink += sidecar.edgeRecordUnchecked(edge_index)
                    .edgeLengthNativeUnchecked();
            });

        hnswlib::VisitedList state(1000000);
        const double state_reset_ns = measureNsPerOperation(
            options.iterations, [&](size_t) {
                state.reset();
                sink += state.curV;
            });
        const double state_mark_ns = measureNsPerOperation(
            options.iterations, [&](size_t i) {
                state.approx_pruned_mass[i % state.numelements] = state.curV;
                sink += state.approx_pruned_mass[i % state.numelements];
            });

        std::ofstream output(options.output_path.c_str());
        if (!output) throw std::runtime_error("cannot create output JSON");
        output << std::setprecision(17)
            << "{\n"
            << "  \"schema_version\": 1,\n"
            << "  \"kernel\": \"raw_fast_v1\",\n"
            << "  \"dimension\": " << header.dimension << ",\n"
            << "  \"pq_m\": " << header.pq_m << ",\n"
            << "  \"pq_ksub\": " << header.pq_ksub << ",\n"
            << "  \"iterations\": " << options.iterations << ",\n"
            << "  \"strict_lut_build_ns\": " << strict_lut_ns << ",\n"
            << "  \"fast_lut_build_ns\": " << fast_lut_ns << ",\n"
            << "  \"strict_lookup_ns\": " << strict_lookup_ns << ",\n"
            << "  \"fast_lookup_ns\": " << fast_lookup_ns << ",\n"
            << "  \"strict_estimator_ns\": " << strict_estimator_ns << ",\n"
            << "  \"fast_estimator_ns\": " << fast_estimator_ns << ",\n"
            << "  \"exact_l2_ns\": " << exact_l2_ns << ",\n"
            << "  \"checked_record_ns\": " << checked_record_ns << ",\n"
            << "  \"direct_record_ns\": " << direct_record_ns << ",\n"
            << "  \"state_reset_ns\": " << state_reset_ns << ",\n"
            << "  \"state_mark_ns\": " << state_mark_ns << ",\n"
            << "  \"state_bytes_per_query\": "
            << (sizeof(hnswlib::vl_type) * state.numelements) << ",\n"
            << "  \"optimizer_sink\": " << sink << "\n"
            << "}\n";
        return output ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 2;
    }
}
