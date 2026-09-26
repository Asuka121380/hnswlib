#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

#include "pq_packed_artifact.h"
#include "hnswlib/edge_estimation/score_bridge.h"
#include "../event_format.h"
#include "../query_store.h"

namespace uq {

class RaBitQArtifactKernel {
 public:
    RaBitQArtifactKernel(const std::filesystem::path& root,
                         std::shared_ptr<const QueryStore> queries,
                         const Header& event_header)
        : queries_(std::move(queries)), config_(readNativeConfig(root / "native.cfg")),
          dimension_(number("dimension")), edge_count_(number64("edge_count")),
          record_size_(number("record_size")), code_size_(number("code_size")),
          sign_bytes_(number("sign_bytes")), current_query_(~uint64_t(0)) {
        require("format", "uq-rabitq/1"); require("backend", "rabitq");
        require("coverage", "full_graph"); require("metric", "inner_product");
        require("query_bits", "0"); require("centroid", "zero");
        require("rotation_layout", "row_major_r_times_column");
        require("rotation_bias", "none");
        if (!queries_ || queries_->dimension() != dimension_ ||
            dimension_ != event_header.dimension || sign_bytes_ != (dimension_ + 7U) / 8U ||
            code_size_ != sign_bytes_ + 8U || record_size_ != 16U + code_size_ ||
            parseHexDigest(config_.at("catalog_identity")) != event_header.identity)
            throw std::runtime_error("RaBitQ artifact/dataset identity or shape mismatch");
        const std::filesystem::path rotation_path =
            checkedArtifactPath(root, config_.at("rotation"));
        const std::filesystem::path records_path =
            checkedArtifactPath(root, config_.at("records"));
        if (artifactSha256(rotation_path) != config_.at("rotation_sha256") ||
            artifactSha256(records_path) != config_.at("records_sha256"))
            throw std::runtime_error("artifact file hash mismatch");
        rotation_ = readArtifactFloats(rotation_path);
        if (rotation_.size() != static_cast<size_t>(dimension_) * dimension_)
            throw std::runtime_error("RaBitQ rotation size mismatch");
        records_ = detail::readFile(records_path.string());
        if (records_.size() != edge_count_ * record_size_)
            throw std::runtime_error("RaBitQ records size mismatch");
    }

    void prepareQuery(uint64_t query_id) {
        const float* query = queries_->query(query_id);
        rotated_query_.assign(dimension_, 0.0f);
        for (uint32_t row = 0; row < dimension_; ++row) {
            float sum = 0.0f;
            const size_t offset = static_cast<size_t>(row) * dimension_;
            for (uint32_t column = 0; column < dimension_; ++column)
                sum += rotation_[offset + column] * query[column];
            rotated_query_[row] = sum;
        }
        current_query_ = query_id;
    }
    void prepareSource(const EventRecord&) {}
    double score(const EventRecord& event) {
        if (current_query_ != event.query_id || event.edge_id >= edge_count_)
            return std::numeric_limits<double>::quiet_NaN();
        const uint8_t* record = records_.data() + event.edge_id * record_size_;
        const double length = detail::readF64(record);
        const double anchor = detail::readF64(record + 8U);
        const uint8_t* code = record + 16U;
        float or_minus_c_l2sqr = 0.0f, dp_multiplier = 0.0f;
        std::memcpy(&or_minus_c_l2sqr, code + sign_bytes_, 4U);
        std::memcpy(&dp_multiplier, code + sign_bytes_ + 4U, 4U);
        if (!(length > 0.0) || !std::isfinite(length) || !std::isfinite(anchor) ||
            !std::isfinite(or_minus_c_l2sqr) || !std::isfinite(dp_multiplier) ||
            std::fabs(or_minus_c_l2sqr) > 1e-6f || !(dp_multiplier >= 0.0f))
            return std::numeric_limits<double>::quiet_NaN();
        double signed_sum = 0.0;
        for (uint32_t d = 0; d < dimension_; ++d) {
            const bool positive = (code[d / 8U] >> (d % 8U)) & 1U;
            signed_sum += (positive ? 1.0 : -1.0) * rotated_query_[d];
        }
        const double qdot = signed_sum * static_cast<double>(dp_multiplier) /
                            std::sqrt(static_cast<double>(dimension_));
        const hnswlib::edge_estimation::EdgeScore score =
            hnswlib::edge_estimation::bridgeDotToSquaredDistance(
                event.d_current, length,
                hnswlib::edge_estimation::DotEstimate(
                    qdot - anchor, hnswlib::edge_estimation::EstimateStatus::Valid));
        return score.valid() ? score.squared_distance
                             : std::numeric_limits<double>::quiet_NaN();
    }
    uint64_t backendBytes() const { return rotation_.size() * 4U + records_.size(); }
    uint64_t scratchBytes() const { return rotated_query_.capacity() * 4U; }

 private:
    uint32_t number(const char* key) const {
        return static_cast<uint32_t>(std::stoul(config_.at(key)));
    }
    uint64_t number64(const char* key) const { return std::stoull(config_.at(key)); }
    void require(const char* key, const char* value) const {
        if (config_.at(key) != value) throw std::runtime_error("unsupported RaBitQ contract");
    }
    std::shared_ptr<const QueryStore> queries_;
    std::map<std::string, std::string> config_;
    uint32_t dimension_;
    uint64_t edge_count_;
    uint32_t record_size_, code_size_, sign_bytes_;
    uint64_t current_query_;
    std::vector<float> rotation_, rotated_query_;
    std::vector<uint8_t> records_;
};

}  // namespace uq
