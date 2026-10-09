#pragma once

#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace uq {

inline size_t parseUnsignedSize(const std::string& value, const char* name) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument(std::string("invalid ") + name);
    size_t end = 0;
    const auto parsed = std::stoull(value, &end);
    if (end != value.size() || parsed > std::numeric_limits<size_t>::max())
        throw std::invalid_argument(std::string("invalid ") + name);
    return static_cast<size_t>(parsed);
}

inline std::vector<uint64_t> readQueryIds(const std::string& path, uint64_t count) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot read query IDs: " + path);
    std::vector<uint64_t> ids;
    std::unordered_set<uint64_t> seen;
    std::string word;
    while (stream >> word) {
        const auto id = parseUnsignedSize(word, "query ID");
        if (id >= count || !seen.insert(id).second)
            throw std::invalid_argument("duplicate or out-of-range query ID");
        ids.push_back(id);
    }
    if (stream.bad() || ids.empty()) throw std::invalid_argument("empty or unreadable query IDs");
    return ids;
}

inline std::vector<uint64_t> queryRange(size_t start, size_t count, size_t available) {
    if (!count || start > available || count > available - start)
        throw std::invalid_argument("query range exceeds available rows");
    std::vector<uint64_t> ids(count);
    for (size_t i = 0; i < count; ++i) ids[i] = start + i;
    return ids;
}

// Reset only touched IDs; the allocation is amortized across windows, not the LUT values.
class QuerySlots {
 public:
    void reset(const std::vector<uint64_t>& ids, size_t available) {
        if (slots_.size() != available) slots_.assign(available, absent());
        else for (auto id : touched_) slots_[id] = absent();
        touched_.clear();
        for (size_t i = 0; i < ids.size(); ++i) {
            const auto id = ids[i];
            if (id >= available || slots_[id] != absent())
                throw std::invalid_argument("invalid prepared query IDs");
            slots_[id] = i;
            touched_.push_back(id);
        }
    }
    size_t at(uint64_t id) const {
        if (id >= slots_.size() || slots_[id] == absent())
            throw std::runtime_error("query absent from prepared window");
        return slots_[id];
    }
    uint64_t bytes() const { return slots_.capacity()*sizeof(size_t) + touched_.capacity()*sizeof(uint64_t); }
 private:
    static size_t absent() { return std::numeric_limits<size_t>::max(); }
    std::vector<size_t> slots_;
    std::vector<uint64_t> touched_;
};

}  // namespace uq
