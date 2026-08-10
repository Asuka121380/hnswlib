#include <algorithm>
#include <iomanip>
#include <iostream>
#include <queue>
#include <utility>
#include <vector>

#include "hnswlib/hnswlib.h"

namespace {

typedef std::pair<float, hnswlib::labeltype> Result;

std::vector<Result> sortedResults(
    std::priority_queue<Result> queue) {
    std::vector<Result> values;
    while (!queue.empty()) {
        values.push_back(queue.top());
        queue.pop();
    }
    std::sort(values.begin(), values.end(),
        [](const Result& left, const Result& right) {
            if (left.first != right.first) return left.first < right.first;
            return left.second < right.second;
        });
    return values;
}

}  // namespace

int main() {
    const size_t dimension = 12U;
    const size_t count = 96U;
    const size_t k = 8U;
    std::vector<float> points(count * dimension);
    for (size_t row = 0; row < count; ++row) {
        for (size_t column = 0; column < dimension; ++column) {
            const int raw = static_cast<int>(
                (row * 37U + column * 19U + row * column * 3U) % 211U);
            points[row * dimension + column] =
                static_cast<float>(raw - 105) / 31.0f +
                static_cast<float>(row) * 0.0005f;
        }
    }

    hnswlib::L2Space space(dimension);
    hnswlib::HierarchicalNSW<float> index(&space, count, 16U, 100U, 73U);
    for (size_t row = 0; row < count; ++row) {
        index.addPoint(points.data() + row * dimension, row);
    }
    index.setEf(48U);

    const size_t query_rows[] = {3U, 17U, 42U, 71U};
    std::cout << std::setprecision(17);
    for (size_t query_index = 0;
         query_index < sizeof(query_rows) / sizeof(query_rows[0]);
         ++query_index) {
        const size_t row = query_rows[query_index];
        const std::vector<Result> results = sortedResults(
            index.searchKnn(points.data() + row * dimension, k));
        std::cout << "query=" << query_index << ",source=" << row;
        for (size_t i = 0; i < results.size(); ++i) {
            std::cout << ',' << results[i].second << ':' << results[i].first;
        }
        std::cout << '\n';
    }
    return 0;
}
