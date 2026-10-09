#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "backends/rotated_pq.h"
#include "event_format.h"
#include "hnswlib/edge_estimation/active_policy.h"
#include "hnswlib/edge_estimation/graph_catalog.h"

namespace uq {

struct LiveOpqMetrics {
    uint64_t attempted = 0U;
    uint64_t valid = 0U;
    uint64_t pruned = 0U;
    uint64_t fallback = 0U;
    uint64_t threshold_unavailable = 0U;
    uint64_t nonfinite_score = 0U;
    uint64_t zero_length = 0U;
    uint64_t catalog_mismatch = 0U;
    uint64_t backend_exception = 0U;
};

// Bridges a full-graph rotated-PQ artifact into the generic HNSW active hook.
template<class K> auto isZeroLengthEdge(K& k, uint64_t id, int) -> decltype(k.zeroLengthEdge(id), bool()) {
    return k.zeroLengthEdge(id);
}
template<class K> bool isZeroLengthEdge(K&, uint64_t, long) { return false; }
// Invalid/mismatched estimates always fail open to the exact distance path.
template <typename Kernel, bool CollectMetrics = true>
class LiveArtifactPruner : public hnswlib::edge_estimation::ActiveEdgePruner {
 public:
    LiveArtifactPruner(Kernel& kernel,
                  const hnswlib::edge_estimation::EdgeCatalog& catalog,
                  double beta)
        : kernel_(kernel), catalog_(catalog), policy_(beta) {}

    void beginQuery(uint64_t query_id) {
        kernel_.prepareQuery(query_id);
        current_query_id_ = query_id;
    }

    void resetMetrics() { metrics_ = LiveOpqMetrics(); }
    const LiveOpqMetrics& metrics() const { return metrics_; }

    hnswlib::edge_estimation::PolicyDecision evaluate(
        const hnswlib::edge_estimation::ActivePruneRequest& request) override {
        if constexpr (CollectMetrics) ++metrics_.attempted;
        if (!request.threshold_valid) {
            if constexpr (CollectMetrics) { ++metrics_.threshold_unavailable; ++metrics_.fallback; }
            return hnswlib::edge_estimation::PolicyDecision{false, true};
        }

        try {
            const hnswlib::edge_estimation::EdgeId edge_id =
                catalog_.edgeId(request.source_id, request.neighbor_slot);
            if (catalog_.target(edge_id) != request.target_id) {
                if constexpr (!CollectMetrics) throw std::runtime_error("catalog mismatch in uninstrumented search");
                if constexpr (CollectMetrics) { ++metrics_.fallback; ++metrics_.catalog_mismatch; }
                return hnswlib::edge_estimation::PolicyDecision{false, true};
            }
            EventRecord event{};
            event.query_id = current_query_id_;
            event.edge_id = edge_id;
            event.source_id = request.source_id;
            event.target_id = request.target_id;
            event.neighbor_slot = request.neighbor_slot;
            event.source_degree = request.source_degree;
            event.d_current = request.d_current;
            event.threshold_before = request.threshold;
            const double estimate = kernel_.score(event);
            if (!std::isfinite(estimate)) {
                if constexpr (CollectMetrics) {
                    ++metrics_.fallback;
                    if (isZeroLengthEdge(kernel_, edge_id, 0)) ++metrics_.zero_length;
                    else ++metrics_.nonfinite_score;
                }
                return hnswlib::edge_estimation::PolicyDecision{false, true};
            }
            const hnswlib::edge_estimation::PolicyDecision decision =
                policy_.decide(
                    hnswlib::edge_estimation::EdgeScore(
                        estimate, hnswlib::edge_estimation::EstimateStatus::Valid),
                    request.threshold, request.threshold_valid);
            if constexpr (CollectMetrics) {
                if (decision.fallback) ++metrics_.fallback;
                else ++metrics_.valid;
                if (decision.prune) ++metrics_.pruned;
            }
            return decision;
        } catch (const std::exception&) {
            if constexpr (!CollectMetrics) throw;
            if constexpr (CollectMetrics) { ++metrics_.fallback; ++metrics_.backend_exception; }
            return hnswlib::edge_estimation::PolicyDecision{false, true};
        }
    }

 private:
    Kernel& kernel_;
    const hnswlib::edge_estimation::EdgeCatalog& catalog_;
    hnswlib::edge_estimation::ScaledThresholdPolicy policy_;
    uint64_t current_query_id_ = 0U;
    LiveOpqMetrics metrics_;
};

using LiveOpqPruner = LiveArtifactPruner<RotatedPqArtifactKernel>;

}  // namespace uq
