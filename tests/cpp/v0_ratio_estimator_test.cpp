#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "v0_sidecar_test_utils.h"
#include "hnswlib/edge_quant_v0_probabilistic_bound.h"

namespace {

void writeFormulaSidecar(const std::string& path) {
    hnswlib::V0SidecarWriteSpec spec;
    spec.dimension = 4U;
    spec.pq_m = 2U;
    spec.pq_nbits = 1U;
    spec.pq_ksub = 2U;
    spec.pq_dsub = 2U;
    spec.node_count = 1U;
    spec.directed_edge_count = 1U;
    spec.training_metadata_json = "{\"purpose\":\"ratio_formula\"}";
    spec.codebook_centroids.assign(8U, 0.0f);
    spec.codebook_centroids[2U] = 0.6f;
    spec.codebook_centroids[3U] = 0.8f;
    spec.codebook_centroids[7U] = 0.5f;
    spec.node_offsets.push_back(0U);
    spec.node_offsets.push_back(1U);
    hnswlib::V0EdgeRecord edge;
    edge.code.push_back(1U);
    edge.code.push_back(1U);
    edge.edge_length = 2.0;
    edge.direction_error = 0.5;
    edge.anchor_projection = 0.0;
    hnswlib::V0SidecarWriter writer(path, spec);
    writer.writeEdgeRecord(edge);
    writer.finalize();
}

void requireNear(double left, double right, double tolerance, const char* text) {
    v0_test::require(std::fabs(left - right) <= tolerance, text);
}

void testFormulaNormClippingAndFallback() {
    const std::string path = "v0_ratio_estimator_test.v0meta";
    v0_test::FileCleanup cleanup(path);
    writeFormulaSidecar(path);
    const hnswlib::V0OwnedSidecar owned = hnswlib::loadV0Sidecar(path);
    const hnswlib::V0SidecarView sidecar = owned.view();
    const hnswlib::V0EdgeRecordView edge = sidecar.edgeRecord(0U);
    const hnswlib::V0RatioCodebookNormLut norm_lut(sidecar);
    const float query[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const hnswlib::V0RatioEstimatorQueryContext context(
        query, sidecar, norm_lut);
    const hnswlib::V0RatioEstimate result =
        context.evaluate(edge, 25.0, 0.5);
    v0_test::require(result.eligible(), "manual ratio case was ineligible");
    requireNear(
        norm_lut.reconstructionNormSquared(edge),
        1.25,
        1.0e-7,
        "norm LUT differs from explicit decoded norm");
    const double expected_s = std::sqrt(1.25);
    const double expected_x_dot_r =
        1.0 * 0.6 + 2.0 * 0.8 + 4.0 * 0.5;
    const double expected_kappa = 2.0 / (2.0 * expected_s);
    const double expected_raw = expected_x_dot_r / (5.0 * expected_s);
    const double expected_ratio = expected_raw / expected_kappa;
    const double expected_distance =
        25.0 + 4.0 - 20.0 * expected_ratio;
    requireNear(result.reconstruction_norm, expected_s, 1.0e-7,
                "reconstruction norm mismatch");
    requireNear(result.kappa_meta, expected_kappa, 1.0e-7,
                "kappa_meta mismatch");
    requireNear(result.rho_hat_raw, expected_raw, 1.0e-7,
                "raw rho mismatch");
    requireNear(result.rho_hat_ratio, expected_ratio, 1.0e-7,
                "ratio rho mismatch");
    requireNear(result.estimated_squared_distance, expected_distance, 1.0e-6,
                "ratio distance mismatch");

    const float large_query[] = {10.0f, 20.0f, 30.0f, 40.0f};
    const hnswlib::V0RatioEstimatorQueryContext clipping_context(
        large_query, sidecar, norm_lut);
    const hnswlib::V0RatioEstimate clipped =
        clipping_context.evaluate(edge, 1.0, 0.5);
    v0_test::require(
        clipped.eligible() && clipped.rho_hat_ratio > 1.0 &&
            clipped.rho_hat_ratio_clipped == 1.0,
        "ratio projection was not clipped to one");

    const hnswlib::V0RatioEstimate zero_n =
        context.evaluate(edge, 0.0, 0.5);
    v0_test::require(!zero_n.eligible(), "zero n did not fail closed");
    const hnswlib::V0RatioEstimate high_kappa_gate =
        context.evaluate(edge, 25.0, 0.95);
    v0_test::require(
        high_kappa_gate.status ==
            hnswlib::V0RatioEstimatorStatus::KappaBelowMinimum,
        "kappa_min fallback did not trigger");
}

void testStrictDecisionAndCurrentFallback() {
    hnswlib::V0RatioEstimate estimate;
    estimate.status = hnswlib::V0RatioEstimatorStatus::Valid;
    estimate.estimated_squared_distance = 7.0;
    const hnswlib::V0RatioCalibrator calibrator =
        hnswlib::V0RatioCalibrator::forTesting("test", 2.0, 0.5);
    hnswlib::V0BoundResult current;
    current.status = hnswlib::V0BoundStatus::Valid;
    current.lower_bound = 4.0;
    const hnswlib::V0RatioBoundResult primary =
        hnswlib::evaluateV0RatioPrimaryBound(estimate, calibrator);
    v0_test::require(
        primary.ratio_eligible && !primary.current_lb_fallback &&
            primary.effective_lower_bound == 5.0,
        "eligible primary ratio bound unexpectedly used current LB");
    const hnswlib::V0RatioBoundResult bound =
        hnswlib::evaluateV0RatioBound(estimate, calibrator, current);
    v0_test::require(
        bound.ratio_lower_bound == 5.0 &&
            !bound.provesFartherThan(5.0) &&
            bound.provesFartherThan(4.999),
        "ratio decision is not strict at equality");

    estimate.status = hnswlib::V0RatioEstimatorStatus::NumericFailure;
    hnswlib::V0RatioBoundResult fallback =
        hnswlib::evaluateV0RatioPrimaryBound(estimate, calibrator);
    v0_test::require(
        !fallback.ratio_eligible && !fallback.effective_bound_valid,
        "invalid primary ratio bound did not fail closed");
    hnswlib::applyV0RatioCurrentBoundFallback(fallback, current);
    v0_test::require(
        fallback.current_lb_fallback &&
            fallback.effective_lower_bound == 4.0,
        "invalid ratio estimate did not use current LB fallback");

    hnswlib::V0RatioBoundResult unchanged = primary;
    hnswlib::applyV0RatioCurrentBoundFallback(unchanged, current);
    v0_test::require(
        unchanged.ratio_eligible && !unchanged.current_lb_fallback &&
            unchanged.effective_lower_bound == primary.effective_lower_bound,
        "current fallback overwrote an eligible ratio bound");

    hnswlib::V0BoundResult invalid_current;
    invalid_current.status = hnswlib::V0BoundStatus::NumericFailure;
    hnswlib::V0RatioBoundResult exact_fallback =
        hnswlib::evaluateV0RatioPrimaryBound(estimate, calibrator);
    hnswlib::applyV0RatioCurrentBoundFallback(
        exact_fallback, invalid_current);
    v0_test::require(
        !exact_fallback.effective_bound_valid &&
            !exact_fallback.current_lb_fallback,
        "invalid current LB did not preserve exact fallback");
}

void writeText(const std::string& path, const std::string& value) {
    std::ofstream out(path.c_str(), std::ios::binary);
    out << value;
    if (!out) throw std::runtime_error("cannot write ratio test JSON");
}

void testFrozenCalibratorShaValidation() {
    const std::string calibrator_path = "v0_ratio_calibrator_test.json";
    const std::string selected_path = "v0_ratio_selected_test.json";
    v0_test::FileCleanup calibrator_cleanup(calibrator_path);
    v0_test::FileCleanup selected_cleanup(selected_path);
    const std::string calibrator =
        "{\"format\":\"v0_ratio_probabilistic_calibrator\","
        "\"status\":\"supported\",\"calibrator_id\":\"query_alpha_1e-1\","
        "\"calibration_level\":\"query\","
        "\"estimator_formula_version\":\"v0_ratio_corrected_pq_distance_v1\","
        "\"fallback_policy\":\"probabilistic_estimator_unavailable_use_current_lb\","
        "\"quantile_decimal\":\"1.25\",\"quantile_hex\":\"0x1.4p+0\","
        "\"input_sha256\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\","
        "\"query_split_manifest_sha256\":\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\","
        "\"nominal_alpha\":0.1,\"quantile\":1.25,\"kappa_min\":0.5}";
    writeText(calibrator_path, calibrator);
    const std::string sha = hnswlib::edgeQuantV0Sha256Hex(
        hnswlib::computeV0FileSha256(calibrator_path));
    const std::string selected =
        "{\"format\":\"v0_ratio_phase4_selected_operating_points\","
        "\"decision\":\"GO_TO_PHASE4\",\"operating_points\":[{"
        "\"operating_point_id\": \"query_alpha_1e-1\","
        "\"calibrator_sha256\":\"" + sha + "\","
        "\"calibration_level\":\"query\",\"alpha\":0.1,"
        "\"estimator_formula_version\":\"v0_ratio_corrected_pq_distance_v1\","
        "\"fallback_policy\":\"probabilistic_estimator_unavailable_use_current_lb\","
        "\"clipping_policy\":\"clipped\","
        "\"quantile\":1.25,\"kappa_min\":0.5,"
        "\"diagnostic_only\":false,\"trusted\":true}]}";
    writeText(selected_path, selected);
    const hnswlib::V0RatioCalibrator loaded =
        hnswlib::V0RatioCalibrator::load(
            selected_path, calibrator_path, "query_alpha_1e-1");
    v0_test::require(
        loaded.validated() && loaded.quantile() == 1.25 &&
            loaded.calibratorSha256() == sha,
        "valid frozen calibrator did not load");
    writeText(calibrator_path, calibrator + "\n");
    bool rejected = false;
    try {
        (void)hnswlib::V0RatioCalibrator::load(
            selected_path, calibrator_path, "query_alpha_1e-1");
    } catch (const std::exception&) {
        rejected = true;
    }
    v0_test::require(rejected, "calibrator SHA mismatch was accepted");
}

}  // namespace

int main() {
#if !defined(HNSWLIB_ENABLE_EDGE_QUANT_V0) || \
    !defined(HNSWLIB_ENABLE_V0_RATIO_ESTIMATOR)
    std::cerr << "V0 ratio estimator is required" << std::endl;
    return 2;
#else
    testFormulaNormClippingAndFallback();
    testStrictDecisionAndCurrentFallback();
    testFrozenCalibratorShaValidation();
    std::cout << "v0_ratio_estimator_test_ok" << std::endl;
    return 0;
#endif
}
