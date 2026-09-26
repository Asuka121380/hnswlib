#include <cmath>
#include <iostream>
#include <stdexcept>

#include "hnswlib/edge_estimation/policy.h"
#include "hnswlib/edge_estimation/score_bridge.h"
#include "hnswlib/edge_estimation/active_policy.h"
#include "hnswlib/edge_estimation/ivf_residual.h"

int main() {
    using namespace hnswlib::edge_estimation;
    const DotEstimate dot(2.0, EstimateStatus::Valid);
    const EdgeScore score = bridgeDotToSquaredDistance(9.0, 3.0, dot);
    if (!score.valid() || std::fabs(score.squared_distance - 6.0) > 1e-12)
        throw std::runtime_error("dot bridge mismatch");
    const ScaledThresholdPolicy policy(1.0);
    if (policy.decide(score, 6.0, true).prune)
        throw std::runtime_error("tie must not prune");
    if (!policy.decide(EdgeScore(6.0001, EstimateStatus::Valid), 6.0, true).prune)
        throw std::runtime_error("strict greater-than failed");
    if (!policy.decide(EdgeScore(0.0, EstimateStatus::ZeroLength), 6.0, true).fallback)
        throw std::runtime_error("invalid score did not fallback");
    const DotEstimate combined = composeIvfResidualDot(
        DotEstimate(1.25, EstimateStatus::Valid),
        DotEstimate(-0.5, EstimateStatus::Valid));
    if (!combined.valid() || std::fabs(combined.value - 0.75) > 1e-12)
        throw std::runtime_error("IVF residual composition mismatch");
    const NoPrunePolicy no_prune;
    if (decideActiveEdge(no_prune, score, 0.0, true).prune)
        throw std::runtime_error("no-prune policy pruned");
    if (!decideActiveEdge(no_prune,
                          EdgeScore(0.0, EstimateStatus::NonFinite),
                          0.0, true).fallback)
        throw std::runtime_error("active invalid score did not fallback");
    std::cout << "uq_policy_test_ok" << std::endl;
    return 0;
}
