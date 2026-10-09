#pragma once

#include "rotated_pq.h"
#include <fstream>
#include <sstream>

namespace uq {

// Two-level edge representation. Graph traversal, not inverted lists, generates candidates.
class IvfEdgeArtifactKernel {
 public:
    IvfEdgeArtifactKernel(const std::filesystem::path& root,
                         std::shared_ptr<const QueryStore> queries, const Header& header)
        : queries_(std::move(queries)), cfg_(readNativeConfig(root / "native.cfg")),
          d_(number("dimension")), m_(number("m")), bits_(number("nbits")),
          nc_(number("nlist")), qb_(number("qjl_bits")), stride_(number("record_size")),
          edges_(std::stoull(cfg_.at("edge_count"))), model_(m_, bits_) {
        std::ifstream seal(root / "complete.sha256");
        std::string digest; seal >> digest;
        if (!seal || digest != artifactSha256(root / "native.cfg") ||
            artifactSha256(root / "manifest.json") != cfg_.at("manifest_sha256"))
            throw std::runtime_error("IVF artifact publication/manifest identity mismatch");
        method_ = cfg_.at("backend");
        const bool opq = method_ == "ivf_opq", qjl = method_ == "ivf_pq_qjl";
        if (cfg_.at("format") != "uq-ivf-edge/1" || cfg_.at("coverage") != "full_graph" ||
            (!opq && !qjl && method_ != "ivf_pq") || !d_ || !nc_ || d_ % m_ ||
            !queries_ || queries_->dimension() != d_ || header.dimension != d_ ||
            parseHexDigest(cfg_.at("catalog_identity")) != header.identity ||
            number("rotation") != static_cast<uint32_t>(opq) ||
            (qjl ? (!qb_ || qb_ % 8U) : qb_ != 0U) ||
            stride_ != 20U + model_.packedCodeBytes() + (qjl ? 8U + qb_/8U : 0U))
            throw std::runtime_error("IVF shape/representation/catalog contract mismatch");
        centers_ = floats(root, "centers.f32le", static_cast<size_t>(nc_)*d_);
        codebook_ = floats(root, "codebook.f32le", static_cast<size_t>(model_.centroidCount())*d_);
        if (opq) rotation_ = floats(root, "rotation.f32le", static_cast<size_t>(d_)*d_);
        if (qjl) projection_ = floats(root, "projection.f32le", static_cast<size_t>(qb_)*d_);
        if (edges_ > std::numeric_limits<size_t>::max()/stride_ ||
            std::filesystem::file_size(root / "edges.bin") != edges_*stride_ ||
            artifactSha256(root / "edges.bin") != cfg_.at("edges.bin_sha256"))
            throw std::runtime_error("IVF records size/hash mismatch");
        records_ = detail::readFile((root / "edges.bin").string());
        for (uint64_t e=0; e<edges_; ++e) {
            const uint8_t* r = records_.data()+e*stride_;
            if (detail::readU32(r+16) >= nc_ || !std::isfinite(detail::readF64(r)) ||
                detail::readF64(r) < 0 || !std::isfinite(detail::readF64(r+8)))
                throw std::runtime_error("IVF invalid edge metadata");
            if (qb_ && (!std::isfinite(detail::readF64(r+20+model_.packedCodeBytes())) ||
                        detail::readF64(r+20+model_.packedCodeBytes()) < 0))
                throw std::runtime_error("IVF invalid QJL scale");
        }
    }

    void verifyIndex(const std::filesystem::path& path) const {
        if (artifactSha256(path) != cfg_.at("index_sha256"))
            throw std::runtime_error("IVF artifact belongs to another index/vector/label mapping");
    }
    const std::string& method() const { return method_; }
    void prepareSource(const EventRecord&) {}
    void prepareQuery(uint64_t id) {
        if (!prepared_) {
            prepareQueryBatch(std::vector<uint64_t>{id}, 1U);
            prepared_ = false; // scalar lifecycle remains scalar on the next query
        }
        slot_ = slots_.at(id); current_ = id;
    }
    void prepareQueryBatch(const std::vector<uint64_t>& ids, size_t batch_size) {
        if (ids.empty() || !batch_size) throw std::invalid_argument("empty IVF query batch");
        prepared_ = false; current_ = ~uint64_t(0);
        slots_.reset(ids, static_cast<size_t>(queries_->count()));
        const size_t ls = lutSize();
        luts_.resize(ids.size()*ls); coarse_.resize(ids.size()*nc_);
        signed_luts_.resize(ids.size()*qjlLutSize());
        const size_t bs = std::min(batch_size, ids.size());
        input_.resize(bs*d_); rotated_.resize(bs*d_); projected_.resize(bs*qb_);
        for (size_t first=0; first<ids.size(); first+=bs) {
            const size_t n = std::min(bs, ids.size()-first);
            for (size_t row=0; row<n; ++row)
                std::copy_n(queries_->query(ids[first+row]), d_, input_.data()+row*d_);
            const float* pq_input = input_.data();
            if (!rotation_.empty()) {
                detail::rotateQueryBatch(input_.data(), n, d_, rotation_.data(), rotated_.data());
                pq_input = rotated_.data();
            }
            detail::buildPqInnerProductTablesBatch(pq_input, n, d_, m_, model_.centroidCount(),
                d_/m_, codebook_.data(), luts_.data()+first*ls);
            // A one-subspace inner-product table is a rectangular GEMM q * centers^T.
            detail::buildPqInnerProductTablesBatch(input_.data(), n, d_, 1, nc_, d_,
                centers_.data(), coarse_.data()+first*nc_);
            if (qb_) {
                detail::buildPqInnerProductTablesBatch(input_.data(), n, d_, 1, qb_, d_,
                    projection_.data(), projected_.data());
                for (size_t row=0; row<n; ++row)
                    for (uint32_t group=0; group<qb_/4; ++group)
                        for (uint32_t mask=0; mask<16; ++mask) {
                            float sum=0;
                            for (uint32_t b=0; b<4; ++b)
                                sum += ((mask>>b)&1 ? 1.0f : -1.0f)*projected_[row*qb_+group*4+b];
                            signed_luts_[(first+row)*qjlLutSize()+group*16+mask] = sum;
                        }
            }
        }
        count_ = ids.size(); batch_ = bs; prepared_ = true;
    }
    void clearPreparedQueryBatch() { prepared_=false; current_=~uint64_t(0); }
    const char* batchPreparationEngine() const { return detail::batchRotationEngine(); }
    size_t preparedQueryCount() const { return count_; }
    size_t preparedBatchSize() const { return batch_; }

    bool zeroLengthEdge(uint64_t id) const {
        return id < edges_ && detail::readF64(records_.data() + id * stride_) == 0.0;
    }
    double score(const EventRecord& event) {
        if (event.query_id != current_ || event.edge_id >= edges_ || !std::isfinite(event.d_current))
            return std::numeric_limits<double>::quiet_NaN();
        const uint8_t* r = records_.data()+event.edge_id*stride_;
        const double length = detail::readF64(r);
        if (!(length>0)) return std::numeric_limits<double>::quiet_NaN();
        const uint32_t center = detail::readU32(r+16);
        const auto dot = model_.estimate(luts_.data()+slot_*lutSize(), lutSize(), r+20, model_.packedCodeBytes());
        if (!dot.valid()) return std::numeric_limits<double>::quiet_NaN();
        double result = event.d_current + detail::readF64(r+8) -
                        2*length*(coarse_[slot_*nc_+center]+dot.value);
        if (qb_) {
            const uint8_t* companion = r+20+model_.packedCodeBytes();
            const double scale = detail::readF64(companion);
            double correction=0;
            const float* table=signed_luts_.data()+slot_*qjlLutSize();
            for (uint32_t byte=0; byte<qb_/8; ++byte) {
                const uint8_t code = companion[8+byte];
                correction += table[(byte*2)*16+(code&15)] + table[(byte*2+1)*16+(code>>4)];
            }
            result -= 2*scale*correction;
        }
        return result;
    }
    uint64_t backendBytes() const {
        return records_.size()+4ULL*(centers_.size()+codebook_.size()+rotation_.size()+projection_.size());
    }
    uint64_t scratchBytes() const {
        return 4ULL*(luts_.capacity()+coarse_.capacity()+signed_luts_.capacity()+input_.capacity()+
                     rotated_.capacity()+projected_.capacity()) + slots_.bytes();
    }
 private:
    uint32_t number(const char* key) const {
        size_t used=0; const auto value=std::stoull(cfg_.at(key), &used);
        if (used!=cfg_.at(key).size() || cfg_.at(key).empty() || cfg_.at(key)[0]=='-' ||
            value>std::numeric_limits<uint32_t>::max() ||
            (std::string(key)=="nbits" && (value<1 || value>8)))
            throw std::runtime_error("IVF numeric config overflow");
        return static_cast<uint32_t>(value);
    }
    std::vector<float> floats(const std::filesystem::path& root, const char* name, size_t size) const {
        if (artifactSha256(root/name)!=cfg_.at(std::string(name)+"_sha256"))
            throw std::runtime_error("IVF model hash mismatch");
        auto values=readArtifactFloats(root/name);
        if (values.size()!=size) throw std::runtime_error("IVF model shape mismatch");
        for (float v: values) if (!std::isfinite(v)) throw std::runtime_error("IVF nonfinite model");
        return values;
    }
    size_t lutSize() const { return static_cast<size_t>(m_)*model_.centroidCount(); }
    size_t qjlLutSize() const { return static_cast<size_t>(qb_)*4; }
    std::shared_ptr<const QueryStore> queries_;
    std::map<std::string,std::string> cfg_;
    uint32_t d_,m_,bits_,nc_,qb_,stride_;
    uint64_t edges_;
    PackedPqModel model_;
    std::string method_;
    uint64_t current_=~uint64_t(0);
    bool prepared_=false;
    size_t slot_=0,count_=0,batch_=0;
    QuerySlots slots_;
    std::vector<float> centers_,codebook_,rotation_,projection_,luts_,coarse_,signed_luts_,input_,rotated_,projected_;
    std::vector<uint8_t> records_;
};
} // namespace uq
