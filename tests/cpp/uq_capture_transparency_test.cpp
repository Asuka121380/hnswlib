#include <cmath>
#include <cstdio>
#include <iostream>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "hnswlib/hnswlib.h"
#include "hnswlib/edge_estimation/observer.h"
#include "tools/edge_estimation/capture_observer.h"

template<class Queue>
bool sameQueue(Queue left, Queue right) {
    if (left.size() != right.size()) return false;
    while (!left.empty()) {
        if (left.top() != right.top()) return false;
        left.pop(); right.pop();
    }
    return true;
}

int main() {
#if !defined(HNSWLIB_ENABLE_EDGE_ESTIMATION_CAPTURE) || !defined(HNSWLIB_ENABLE_EDGE_QUANT_V0)
    std::cerr << "capture and graph access flags are required" << std::endl;
    return 2;
#else
    const size_t dimension = 8U;
    const size_t count = 80U;
    std::mt19937 rng(19U);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> data(count * dimension);
    for (size_t i = 0; i < data.size(); ++i) data[i] = normal(rng);
    // Include an exact duplicate to exercise a zero-length graph edge when present.
    for (size_t d = 0; d < dimension; ++d) data[dimension + d] = data[d];

    hnswlib::L2Space space(dimension);
    hnswlib::HierarchicalNSW<float> index(&space, count, 8, 40, 19U);
    for (size_t i = 0; i < count; ++i)
        index.addPoint(&data[i * dimension], static_cast<hnswlib::labeltype>(1000U + i));
    index.setEf(30U);
    const hnswlib::edge_estimation::EdgeCatalog catalog =
        hnswlib::edge_estimation::EdgeCatalog::fromLayer0Graph(
            index.getV0Layer0GraphView());
    uq::DatasetCaptureObserver observer(catalog, std::vector<uint64_t>{17U});
    const float* query = &data[13U * dimension];
    const auto baseline = index.searchKnn(query, 7U);
    std::priority_queue<std::pair<float, hnswlib::labeltype> > captured;
    {
        hnswlib::edge_estimation::ScopedCaptureObserver scope(&observer);
        captured = index.searchKnn(query, 7U);
    }
    if (!sameQueue(baseline, captured))
        throw std::runtime_error("capture observer changed search results");
    if (observer.events().empty() || observer.labels().empty() ||
        observer.ranges().size() != 1U)
        throw std::runtime_error("capture observer produced an incomplete dataset");
    observer.write("uq_capture_events.bin", "uq_capture_labels.bin",
                   "uq_capture_ranges.bin", static_cast<uint32_t>(dimension));
    const std::vector<uq::EventRecord> events = uq::readEvents("uq_capture_events.bin");
    const std::vector<uq::LabelRecord> labels = uq::readLabels("uq_capture_labels.bin");
    const std::vector<uq::QueryRangeRecord> ranges = uq::readQueryRanges("uq_capture_ranges.bin");
    uq::validateDataset(events, labels, ranges);
    std::remove("uq_capture_events.bin"); std::remove("uq_capture_labels.bin");
    std::remove("uq_capture_ranges.bin");
    std::cout << "uq_capture_transparency_test_ok events=" << events.size() << std::endl;
    return 0;
#endif
}
