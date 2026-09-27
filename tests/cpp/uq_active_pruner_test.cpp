#include <cstdint>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include "hnswlib/hnswlib.h"
#include "hnswlib/edge_estimation/active_policy.h"

namespace {

class CountingNoPrune : public hnswlib::edge_estimation::ActiveEdgePruner {
 public:
    hnswlib::edge_estimation::PolicyDecision evaluate(
        const hnswlib::edge_estimation::ActivePruneRequest& request) override {
        ++calls;
        if (request.source_id == hnswlib::edge_estimation::kInvalidNodeId ||
            request.target_id == hnswlib::edge_estimation::kInvalidNodeId)
            throw std::runtime_error("active request omitted node identity");
        return hnswlib::edge_estimation::PolicyDecision{false, false};
    }
    uint64_t calls = 0U;
};

std::vector<std::pair<float, hnswlib::labeltype> > ordered(
    std::priority_queue<std::pair<float, hnswlib::labeltype> > values) {
    std::vector<std::pair<float, hnswlib::labeltype> > result;
    while (!values.empty()) { result.push_back(values.top()); values.pop(); }
    return result;
}

}  // namespace

int main() {
#ifndef HNSWLIB_ENABLE_EDGE_ESTIMATION_ACTIVE
    std::cerr << "active hook is not compiled\n";
    return 2;
#else
    hnswlib::L2Space space(4U);
    hnswlib::HierarchicalNSW<float> index(&space, 128U, 12U, 100U, 17U);
    std::vector<std::vector<float> > points(96U, std::vector<float>(4U));
    for (size_t i = 0; i < points.size(); ++i) {
        points[i][0] = static_cast<float>(i % 11U);
        points[i][1] = static_cast<float>((i * 3U) % 17U);
        points[i][2] = static_cast<float>((i * 7U) % 19U);
        points[i][3] = static_cast<float>((i * 13U) % 23U);
        index.addPoint(points[i].data(), static_cast<hnswlib::labeltype>(i));
    }
    index.setEf(48U);
    const float query[4] = {3.25f, 5.5f, 7.75f, 2.0f};
    const std::vector<std::pair<float, hnswlib::labeltype> > baseline =
        ordered(index.searchKnn(query, 10U));
    CountingNoPrune pruner;
    std::vector<std::pair<float, hnswlib::labeltype> > observed;
    {
        hnswlib::edge_estimation::ScopedActiveEdgePruner active(&pruner);
        observed = ordered(index.searchKnn(query, 10U));
    }
    if (pruner.calls == 0U) throw std::runtime_error("active hook was not invoked");
    if (baseline != observed) throw std::runtime_error("no-prune hook changed search result");
    std::cout << "ACTIVE_PRUNER_TEST=PASS calls=" << pruner.calls << '\n';
    return 0;
#endif
}
