#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

#include "edge_quant_v0.h"
#include "edge_quant_v0_ratio_estimator.h"

namespace hnswlib {

static const uint32_t V0_RATIO_SHADOW_SCHEMA_VERSION = 1U;
static const char* const V0_RATIO_FALLBACK_POLICY =
    "probabilistic_estimator_unavailable_use_current_lb";

namespace edge_quant_v0_ratio_detail {

inline std::string readText(const std::string& path) {
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open ratio configuration: " + path);
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (!input.good() && !input.eof()) {
        throw std::runtime_error("Cannot read ratio configuration: " + path);
    }
    return buffer.str();
}

inline std::string stringField(
    const std::string& text,
    const std::string& key) {
    const std::regex expression(
        "\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    std::smatch match;
    if (!std::regex_search(text, match, expression)) {
        throw std::runtime_error(
            "Ratio configuration is missing string field: " + key);
    }
    return match[1].str();
}

inline double numberField(
    const std::string& text,
    const std::string& key) {
    const std::regex expression(
        "\\\"" + key +
        "\\\"\\s*:\\s*([-+]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE][-+]?[0-9]+)?)");
    std::smatch match;
    if (!std::regex_search(text, match, expression)) {
        throw std::runtime_error(
            "Ratio configuration is missing numeric field: " + key);
    }
    const double value = std::stod(match[1].str());
    if (!std::isfinite(value)) {
        throw std::runtime_error(
            "Ratio configuration field is non-finite: " + key);
    }
    return value;
}

inline bool boolField(
    const std::string& text,
    const std::string& key) {
    const std::regex expression(
        "\\\"" + key + "\\\"\\s*:\\s*(true|false)");
    std::smatch match;
    if (!std::regex_search(text, match, expression)) {
        throw std::runtime_error(
            "Ratio configuration is missing boolean field: " + key);
    }
    return match[1].str() == "true";
}

inline std::string regexEscape(const std::string& value) {
    static const std::string special = R"(\.^$|()[]{}*+?)";
    std::string escaped;
    for (size_t i = 0U; i < value.size(); ++i) {
        if (special.find(value[i]) != std::string::npos) escaped.push_back('\\');
        escaped.push_back(value[i]);
    }
    return escaped;
}

inline std::string objectContainingId(
    const std::string& text,
    const std::string& id) {
    const std::regex id_expression(
        "\\\"operating_point_id\\\"\\s*:\\s*\\\"" +
        regexEscape(id) + "\\\"");
    std::smatch id_match;
    if (!std::regex_search(text, id_match, id_expression)) {
        throw std::runtime_error(
            "Selected operating point is absent: " + id);
    }
    const size_t token_position = static_cast<size_t>(id_match.position());
    size_t begin = token_position;
    while (begin > 0U && text[begin] != '{') {
        --begin;
    }
    if (text[begin] != '{') {
        throw std::runtime_error(
            "Selected operating point object is malformed");
    }
    bool in_string = false;
    bool escaped = false;
    size_t depth = 0U;
    for (size_t i = begin; i < text.size(); ++i) {
        const char c = text[i];
        if (in_string) {
            if (escaped) escaped = false;
            else if (c == '\\') escaped = true;
            else if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') in_string = true;
        else if (c == '{') ++depth;
        else if (c == '}') {
            if (--depth == 0U) {
                return text.substr(begin, i - begin + 1U);
            }
        }
    }
    throw std::runtime_error(
        "Selected operating point object is unterminated");
}

inline bool nearlyEqual(double left, double right) {
    return std::fabs(left - right) <=
        8.0 * std::numeric_limits<double>::epsilon() *
        (1.0 + std::max(std::fabs(left), std::fabs(right)));
}

}  // namespace edge_quant_v0_ratio_detail

class V0RatioCalibrator {
 public:
    static V0RatioCalibrator load(
        const std::string& selected_path,
        const std::string& calibrator_path,
        const std::string& operating_point_id) {
        const std::string selected =
            edge_quant_v0_ratio_detail::readText(selected_path);
        const std::string calibrator =
            edge_quant_v0_ratio_detail::readText(calibrator_path);
        if (edge_quant_v0_ratio_detail::stringField(selected, "format") !=
                "v0_ratio_phase4_selected_operating_points" ||
            edge_quant_v0_ratio_detail::stringField(selected, "decision") !=
                "GO_TO_PHASE4") {
            throw std::runtime_error(
                "Phase-3 selection is not approved for Phase 4");
        }
        const std::string selected_object =
            edge_quant_v0_ratio_detail::objectContainingId(
                selected, operating_point_id);
        const std::string expected_sha =
            edge_quant_v0_ratio_detail::stringField(
                selected_object, "calibrator_sha256");
        const std::string actual_sha = edgeQuantV0Sha256Hex(
            computeV0FileSha256(calibrator_path));
        if (actual_sha != expected_sha) {
            throw std::runtime_error(
                "Frozen calibrator SHA-256 does not match Phase-3 selection");
        }
        if (edge_quant_v0_ratio_detail::stringField(calibrator, "format") !=
                "v0_ratio_probabilistic_calibrator" ||
            edge_quant_v0_ratio_detail::stringField(calibrator, "status") !=
                "supported" ||
            edge_quant_v0_ratio_detail::stringField(
                calibrator, "calibrator_id") != operating_point_id) {
            throw std::runtime_error(
                "Calibrator identity or status is invalid");
        }

        V0RatioCalibrator result;
        result.operating_point_id_ = operating_point_id;
        result.calibration_level_ =
            edge_quant_v0_ratio_detail::stringField(
                calibrator, "calibration_level");
        result.formula_version_ =
            edge_quant_v0_ratio_detail::stringField(
                calibrator, "estimator_formula_version");
        result.fallback_policy_ =
            edge_quant_v0_ratio_detail::stringField(
                calibrator, "fallback_policy");
        result.quantile_decimal_ =
            edge_quant_v0_ratio_detail::stringField(
                calibrator, "quantile_decimal");
        result.quantile_hex_ =
            edge_quant_v0_ratio_detail::stringField(
                calibrator, "quantile_hex");
        result.calibration_dataset_sha256_ =
            edge_quant_v0_ratio_detail::stringField(
                calibrator, "input_sha256");
        result.query_split_manifest_sha256_ =
            edge_quant_v0_ratio_detail::stringField(
                calibrator, "query_split_manifest_sha256");
        result.nominal_alpha_ =
            edge_quant_v0_ratio_detail::numberField(
                calibrator, "nominal_alpha");
        result.quantile_ = edge_quant_v0_ratio_detail::numberField(
            calibrator, "quantile");
        result.kappa_min_ = edge_quant_v0_ratio_detail::numberField(
            calibrator, "kappa_min");
        result.diagnostic_only_ =
            edge_quant_v0_ratio_detail::boolField(
                selected_object, "diagnostic_only");
        result.trusted_ = edge_quant_v0_ratio_detail::boolField(
            selected_object, "trusted");
        result.calibrator_sha256_ = actual_sha;
        result.selected_sha256_ = edgeQuantV0Sha256Hex(
            computeV0FileSha256(selected_path));
        result.selected_path_ = selected_path;
        result.calibrator_path_ = calibrator_path;

        const double selected_alpha =
            edge_quant_v0_ratio_detail::numberField(
                selected_object, "alpha");
        const double selected_quantile =
            edge_quant_v0_ratio_detail::numberField(
                selected_object, "quantile");
        const double selected_kappa =
            edge_quant_v0_ratio_detail::numberField(
                selected_object, "kappa_min");
        if (result.formula_version_ !=
                V0_RATIO_ESTIMATOR_FORMULA_VERSION ||
            result.fallback_policy_ != V0_RATIO_FALLBACK_POLICY ||
            edge_quant_v0_ratio_detail::stringField(
                selected_object, "estimator_formula_version") !=
                    V0_RATIO_ESTIMATOR_FORMULA_VERSION ||
            edge_quant_v0_ratio_detail::stringField(
                selected_object, "fallback_policy") !=
                    V0_RATIO_FALLBACK_POLICY ||
            edge_quant_v0_ratio_detail::stringField(
                selected_object, "clipping_policy") != "clipped" ||
            result.calibration_level_ !=
                edge_quant_v0_ratio_detail::stringField(
                    selected_object, "calibration_level") ||
            !edge_quant_v0_ratio_detail::nearlyEqual(
                result.nominal_alpha_, selected_alpha) ||
            !edge_quant_v0_ratio_detail::nearlyEqual(
                result.quantile_, selected_quantile) ||
            !edge_quant_v0_ratio_detail::nearlyEqual(
                result.kappa_min_, selected_kappa) ||
            !(result.nominal_alpha_ > 0.0 && result.nominal_alpha_ < 1.0) ||
            !(result.quantile_ >= 0.0) ||
            !(result.kappa_min_ > 0.0)) {
            throw std::runtime_error(
                "Selected operating point and calibrator disagree");
        }
        result.validated_ = true;
        return result;
    }

    static V0RatioCalibrator forTesting(
        const std::string& id,
        double quantile,
        double kappa_min) {
        V0RatioCalibrator result;
        result.operating_point_id_ = id;
        result.calibration_level_ = "record";
        result.formula_version_ = V0_RATIO_ESTIMATOR_FORMULA_VERSION;
        result.fallback_policy_ = V0_RATIO_FALLBACK_POLICY;
        result.quantile_ = quantile;
        result.quantile_decimal_ = std::to_string(quantile);
        result.quantile_hex_ = "test-only";
        result.kappa_min_ = kappa_min;
        result.nominal_alpha_ = 0.1;
        result.trusted_ = true;
        result.validated_ = true;
        return result;
    }

    bool validated() const { return validated_; }
    const std::string& operatingPointId() const {
        return operating_point_id_;
    }
    const std::string& calibrationLevel() const {
        return calibration_level_;
    }
    const std::string& formulaVersion() const {
        return formula_version_;
    }
    const std::string& fallbackPolicy() const {
        return fallback_policy_;
    }
    const std::string& quantileDecimal() const {
        return quantile_decimal_;
    }
    const std::string& quantileHex() const { return quantile_hex_; }
    const std::string& calibratorSha256() const {
        return calibrator_sha256_;
    }
    const std::string& selectedSha256() const {
        return selected_sha256_;
    }
    const std::string& calibrationDatasetSha256() const {
        return calibration_dataset_sha256_;
    }
    const std::string& querySplitManifestSha256() const {
        return query_split_manifest_sha256_;
    }
    const std::string& selectedPath() const { return selected_path_; }
    const std::string& calibratorPath() const { return calibrator_path_; }
    double nominalAlpha() const { return nominal_alpha_; }
    double quantile() const { return quantile_; }
    double kappaMin() const { return kappa_min_; }
    bool diagnosticOnly() const { return diagnostic_only_; }
    bool trusted() const { return trusted_; }

 private:
    bool validated_ = false;
    bool diagnostic_only_ = false;
    bool trusted_ = false;
    std::string operating_point_id_;
    std::string calibration_level_;
    std::string formula_version_;
    std::string fallback_policy_;
    std::string quantile_decimal_;
    std::string quantile_hex_;
    std::string calibrator_sha256_;
    std::string selected_sha256_;
    std::string calibration_dataset_sha256_;
    std::string query_split_manifest_sha256_;
    std::string selected_path_;
    std::string calibrator_path_;
    double nominal_alpha_ = 0.0;
    double quantile_ = 0.0;
    double kappa_min_ = 0.0;
};

struct V0RatioBoundResult {
    V0RatioEstimate estimate;
    double quantile = 0.0;
    double ratio_lower_bound = 0.0;
    double effective_lower_bound = 0.0;
    bool ratio_eligible = false;
    bool current_lb_fallback = false;
    bool effective_bound_valid = false;

    bool provesFartherThan(double threshold) const {
        return effective_bound_valid && std::isfinite(threshold) &&
            threshold >= 0.0 && effective_lower_bound > threshold;
    }
};

inline V0RatioBoundResult evaluateV0RatioBound(
    const V0RatioEstimate& estimate,
    const V0RatioCalibrator& calibrator,
    const V0BoundResult& current_bound) {
    if (!calibrator.validated()) {
        throw std::invalid_argument(
            "V0 ratio calibrator has not passed validation");
    }
    V0RatioBoundResult result;
    result.estimate = estimate;
    result.quantile = calibrator.quantile();
    if (estimate.eligible()) {
        const double lower =
            estimate.estimated_squared_distance - calibrator.quantile();
        if (std::isfinite(lower)) {
            result.ratio_lower_bound = lower > 0.0 ? lower : 0.0;
            result.effective_lower_bound = result.ratio_lower_bound;
            result.ratio_eligible = true;
            result.effective_bound_valid = true;
            return result;
        }
    }
    if (current_bound.valid()) {
        result.effective_lower_bound = current_bound.lower_bound;
        result.current_lb_fallback = true;
        result.effective_bound_valid = true;
    }
    return result;
}

struct V0RatioQueryMetrics {
    uint64_t ratio_bound_evaluated = 0U;
    uint64_t ratio_eligible = 0U;
    uint64_t ratio_bound_pruned = 0U;
    uint64_t ratio_current_lb_fallback = 0U;
    uint64_t ratio_current_lb_fallback_pruned = 0U;
    uint64_t ratio_exact_fallback = 0U;
    uint64_t ratio_invalid_fallback = 0U;
    uint64_t exact_distance_saved = 0U;
    uint64_t oracle_prunable = 0U;
    uint64_t ratio_interval_violation = 0U;
    uint64_t ratio_false_prune = 0U;
    uint64_t visited_nodes = 0U;
    uint64_t candidate_expansions = 0U;
    uint64_t estimator_time_ns = 0U;
    uint64_t exact_distance_time_ns = 0U;

    void reset() { *this = V0RatioQueryMetrics(); }
};

#ifdef HNSWLIB_ENABLE_V0_RATIO_SHADOW
struct V0RatioShadowRecord {
    uint64_t query_id = 0U;
    uint64_t current_node_id = 0U;
    uint64_t candidate_id = 0U;
    uint32_t graph_layer = 0U;
    double current_squared_distance = 0.0;
    double threshold = 0.0;
    double current_lb = 0.0;
    bool current_lb_valid = false;
    bool current_would_prune = false;
    double edge_length = 0.0;
    double direction_error = 0.0;
    double reconstruction_norm = 0.0;
    double x_dot_reconstruction = 0.0;
    double kappa_meta = 0.0;
    bool ratio_eligible = false;
    std::string ratio_fallback_reason;
    double rho_hat_raw = 0.0;
    double rho_hat_ratio = 0.0;
    double ratio_estimated_squared_distance = 0.0;
    double ratio_quantile = 0.0;
    double ratio_lb = 0.0;
    double ratio_effective_lb = 0.0;
    bool ratio_used_current_fallback = false;
    bool ratio_would_prune = false;
    double exact_squared_distance = 0.0;
    bool oracle_would_prune = false;
    bool ratio_interval_violation = false;
    bool ratio_false_prune = false;
    std::string calibrator_id;
};

class V0RatioShadowCollector {
 public:
    virtual ~V0RatioShadowCollector() {}
    virtual void append(const V0RatioShadowRecord& record) = 0;
};
#endif

}  // namespace hnswlib
