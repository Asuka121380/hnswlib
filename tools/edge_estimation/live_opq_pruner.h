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
};

// Bridges a full-graph rotated-PQ artifact into the generic HNSW active hook.
// Invalid/mismatched estimates always fail open to the exact distance path.
class LiveOpqPruner : public hnswlib::edge_estimation::ActiveEdgePruner {
 public:
    LiveOpqPruner(RotatedPqArtifactKernel& kernel,
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
        ++metrics_.attempted;
        if (!request.threshold_valid) {
            ++metrics_.threshold_unavailable;
            ++metrics_.fallback;
            return hnswlib::edge_estimation::PolicyDecision{false, true};
        }

        try {
            const hnswlib::edge_estimation::EdgeId edge_id =
                catalog_.edgeId(request.source_id, request.neighbor_slot);
            if (catalog_.target(edge_id) != request.target_id) {
                ++metrics_.fallback;
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
                ++metrics_.fallback;
                return hnswlib::edge_estimation::PolicyDecision{false, true};
            }
            const hnswlib::edge_estimation::PolicyDecision decision =
                policy_.decide(
                    hnswlib::edge_estimation::EdgeScore(
                        estimate, hnswlib::edge_estimation::EstimateStatus::Valid),
                    request.threshold, request.threshold_valid);
            if (decision.fallback) ++metrics_.fallback;
            else ++metrics_.valid;
            if (decision.prune) ++metrics_.pruned;
            return decision;
        } catch (const std::exception&) {
            ++metrics_.fallback;
            return hnswlib::edge_estimation::PolicyDecision{false, true};
        }
    }

 private:
    RotatedPqArtifactKernel& kernel_;
    const hnswlib::edge_estimation::EdgeCatalog& catalog_;
    hnswlib::edge_estimation::ScaledThresholdPolicy policy_;
    uint64_t current_query_id_ = 0U;
    LiveOpqMetrics metrics_;
};

}  // namespace uq
