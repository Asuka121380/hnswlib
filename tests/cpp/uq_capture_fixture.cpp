#include <cstdint>
#include <fstream>
#include <random>
#include <vector>

#include "hnswlib/hnswlib.h"

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::string root(argv[1]);
    const uint32_t dimension = 8U;
    const size_t count = 96U;
    std::mt19937 rng(77U);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> data(count * dimension);
    for (size_t i = 0; i < data.size(); ++i) data[i] = normal(rng);
    hnswlib::L2Space space(dimension);
    hnswlib::HierarchicalNSW<float> index(&space, count, 8U, 40U, 77U);
    for (size_t i = 0; i < count; ++i)
        index.addPoint(&data[i * dimension], static_cast<hnswlib::labeltype>(1000U + i));
    index.saveIndex(root + "/index.bin");
    std::ofstream base(root + "/base.fvecs", std::ios::binary);
    for (size_t i = 0; i < count; ++i) {
        base.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
        base.write(reinterpret_cast<const char*>(&data[i * dimension]),
                   dimension * sizeof(float));
    }
    std::ofstream queries(root + "/queries.fvecs", std::ios::binary);
    for (size_t i = 0; i < 6U; ++i) {
        queries.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
        queries.write(reinterpret_cast<const char*>(&data[(i + 10U) * dimension]),
                      dimension * sizeof(float));
    }
    std::ofstream ids(root + "/query_ids.txt");
    ids << "0\n2\n5\n";
    return base && queries && ids ? 0 : 1;
}
