#pragma once

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace uq {

// Immutable fvecs storage. Construction performs all file I/O so replay only
// measures backend query preparation (rotation/LUT construction) and scoring.
class QueryStore {
 public:
    QueryStore(const std::filesystem::path& path, uint32_t dimension)
        : dimension_(dimension) {
        if (dimension_ == 0U) throw std::invalid_argument("query dimension must be positive");
        const uint64_t stride = 4U + static_cast<uint64_t>(dimension_) * 4U;
        const uint64_t bytes = std::filesystem::file_size(path);
        if (bytes == 0U || bytes % stride != 0U)
            throw std::runtime_error("query fvecs file size mismatch");
        count_ = bytes / stride;
        values_.resize(static_cast<size_t>(count_) * dimension_);
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open query fvecs file");
        for (uint64_t row = 0; row < count_; ++row) {
            uint32_t stored = 0U;
            input.read(reinterpret_cast<char*>(&stored), sizeof(stored));
            float* destination = values_.data() + static_cast<size_t>(row) * dimension_;
            input.read(reinterpret_cast<char*>(destination),
                       static_cast<std::streamsize>(dimension_) * 4);
            if (!input || stored != dimension_)
                throw std::runtime_error("query fvec read failed");
            for (uint32_t column = 0; column < dimension_; ++column)
                if (!std::isfinite(destination[column]))
                    throw std::runtime_error("non-finite query value");
        }
        char trailing = 0;
        if (input.read(&trailing, 1))
            throw std::runtime_error("query fvecs contains trailing bytes");
    }

    const float* query(uint64_t query_id) const {
        if (query_id >= count_) throw std::runtime_error("query id out of range");
        return values_.data() + static_cast<size_t>(query_id) * dimension_;
    }

    uint32_t dimension() const { return dimension_; }
    uint64_t count() const { return count_; }
    uint64_t bytes() const { return static_cast<uint64_t>(values_.size()) * sizeof(float); }

 private:
    uint32_t dimension_;
    uint64_t count_ = 0U;
    std::vector<float> values_;
};

}  // namespace uq
