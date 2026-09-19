#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "hnswlib/edge_quant_v0_residual.h"

namespace {
struct Options {
    uint32_t dimension = 960U, bits = 64U;
    size_t edges = 1U << 16, queries = 100U, blocks = 7U;
    std::string output;
};
Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        if (key == "--help") {
            std::cout << "v0_residual_kernel_microbenchmark --output JSON "
                         "[--dimension N] [--bits N] [--edges N] "
                         "[--queries N] [--blocks N]\n";
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::invalid_argument("missing benchmark value");
        const std::string value(argv[++i]);
        if (key == "--output") o.output = value;
        else if (key == "--dimension") o.dimension = std::stoul(value);
        else if (key == "--bits") o.bits = std::stoul(value);
        else if (key == "--edges") o.edges = std::stoull(value);
        else if (key == "--queries") o.queries = std::stoull(value);
        else if (key == "--blocks") o.blocks = std::stoull(value);
        else throw std::invalid_argument("unknown benchmark option: " + key);
    }
    if (o.output.empty() || !o.dimension || !o.edges || !o.queries || !o.blocks ||
        o.bits == 0U || o.bits % 8U || o.bits > 256U)
        throw std::invalid_argument("invalid benchmark contract");
    return o;
}
uint64_t elapsed(std::chrono::steady_clock::time_point start,
                 std::chrono::steady_clock::time_point stop) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        std::mt19937 rng(42U);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> matrix(static_cast<size_t>(o.bits) * o.dimension);
        std::vector<float> query(static_cast<size_t>(o.queries) * o.dimension);
        std::vector<float> candidates(o.edges * o.dimension);
        for (size_t i = 0; i < matrix.size(); ++i) matrix[i] = normal(rng);
        for (size_t i = 0; i < query.size(); ++i) query[i] = normal(rng);
        for (size_t i = 0; i < candidates.size(); ++i) candidates[i] = normal(rng);
        const size_t stride = o.bits / 8U + 9U;
        std::vector<uint8_t> records(o.edges * stride, 0U);
        for (size_t i = 0; i < records.size(); ++i)
            records[i] = static_cast<uint8_t>(rng() & 255U);
        volatile double sink = 0.0;
        std::ofstream output(o.output.c_str());
        if (!output) throw std::runtime_error("cannot open benchmark output");
        output << "{\"schema_version\":1,\"dimension\":" << o.dimension
               << ",\"bits\":" << o.bits << ",\"edges\":" << o.edges
               << ",\"queries\":" << o.queries << ",\"stride\":"
               << stride << ",\"blocks\":[";
        for (size_t block = 0; block < o.blocks; ++block) {
            const size_t query_id = block % o.queries;
            const float* q = query.data() + query_id * o.dimension;
            const auto setup_start = std::chrono::steady_clock::now();
            hnswlib::V0ResidualQueryContext context(
                q, matrix.data(), o.dimension, o.bits);
            const auto setup_stop = std::chrono::steady_clock::now();
            const auto correction_start = std::chrono::steady_clock::now();
            double correction_sum = 0.0;
            for (size_t edge = 0; edge < o.edges; ++edge)
                correction_sum += context.correct(1.0, records.data() + edge * stride,
                                                  0.01f, 0.1f);
            const auto correction_stop = std::chrono::steady_clock::now();
            const auto metadata_start = std::chrono::steady_clock::now();
            double metadata_sum = 0.0;
            for (size_t edge = 0; edge < o.edges; ++edge)
                metadata_sum += records[edge * stride] * 1e-6;
            const auto metadata_stop = std::chrono::steady_clock::now();
            const auto exact_start = std::chrono::steady_clock::now();
            double exact_sum = 0.0;
            for (size_t edge = 0; edge < o.edges; ++edge) {
                const float* v = candidates.data() + edge * o.dimension;
                float distance = 0.0f;
                for (uint32_t coordinate = 0; coordinate < o.dimension; ++coordinate) {
                    const float delta = q[coordinate] - v[coordinate];
                    distance += delta * delta;
                }
                exact_sum += distance;
            }
            const auto exact_stop = std::chrono::steady_clock::now();
            sink += correction_sum + metadata_sum + exact_sum;
            if (block) output << ',';
            output << "{\"block\":" << block
                   << ",\"setup_ns\":" << elapsed(setup_start, setup_stop)
                   << ",\"correction_total_ns\":" << elapsed(correction_start, correction_stop)
                   << ",\"metadata_total_ns\":" << elapsed(metadata_start, metadata_stop)
                   << ",\"exact_total_ns\":" << elapsed(exact_start, exact_stop)
                   << '}';
        }
        output << "],\"checksum\":" << std::setprecision(17) << sink << "}\n";
        if (!output) throw std::runtime_error("cannot write benchmark output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
