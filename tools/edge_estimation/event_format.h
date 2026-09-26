#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace uq {

static const uint32_t kSchemaVersion = 1U;
static const uint32_t kHeaderSize = 64U;
static const uint32_t kEventRecordSize = 80U;
static const uint32_t kLabelRecordSize = 16U;
static const uint32_t kQueryRangeRecordSize = 24U;

enum class EventKind : uint8_t {
    QueryBegin = 1,
    SourceBegin = 2,
    Candidate = 3,
    ExactOnly = 4,
    QueryEnd = 5
};

enum EventFlags : uint8_t {
    ThresholdValid = 1U,
    FirstVisit = 2U,
    ScoreSlotEligible = 4U
};

struct Header {
    std::array<uint8_t, 8> magic;
    uint32_t schema_version;
    uint32_t header_size;
    uint32_t record_size;
    uint32_t dimension;
    uint64_t record_count;
    std::array<uint8_t, 32> identity;
};

struct EventRecord {
    EventKind kind;
    uint8_t flags;
    int32_t graph_layer;
    uint64_t event_id;
    uint64_t query_id;
    uint64_t expansion_id;
    uint64_t edge_id;
    uint32_t source_id;
    uint32_t target_id;
    uint32_t neighbor_slot;
    uint32_t source_degree;
    double d_current;
    double threshold_before;
};

struct LabelRecord {
    uint64_t event_id;
    double exact_squared_distance;
};

struct QueryRangeRecord {
    uint64_t query_id;
    uint64_t begin_event;
    uint64_t event_count;
};

namespace detail {

inline uint32_t readU32(const uint8_t* bytes) {
    uint32_t value = 0U;
    for (size_t i = 0; i < 4U; ++i)
        value |= static_cast<uint32_t>(bytes[i]) << (8U * i);
    return value;
}

inline int32_t readI32(const uint8_t* bytes) {
    const uint32_t raw = readU32(bytes);
    int32_t value = 0;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}

inline uint64_t readU64(const uint8_t* bytes) {
    uint64_t value = 0U;
    for (size_t i = 0; i < 8U; ++i)
        value |= static_cast<uint64_t>(bytes[i]) << (8U * i);
    return value;
}

inline double readF64(const uint8_t* bytes) {
    const uint64_t raw = readU64(bytes);
    double value = 0.0;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}

inline void appendU32(std::vector<uint8_t>& out, uint32_t value) {
    for (size_t i = 0; i < 4U; ++i)
        out.push_back(static_cast<uint8_t>((value >> (8U * i)) & 0xffU));
}

inline void appendI32(std::vector<uint8_t>& out, int32_t value) {
    uint32_t raw = 0U;
    std::memcpy(&raw, &value, sizeof(raw));
    appendU32(out, raw);
}

inline void appendU64(std::vector<uint8_t>& out, uint64_t value) {
    for (size_t i = 0; i < 8U; ++i)
        out.push_back(static_cast<uint8_t>((value >> (8U * i)) & 0xffU));
}

inline void appendF64(std::vector<uint8_t>& out, double value) {
    uint64_t raw = 0U;
    std::memcpy(&raw, &value, sizeof(raw));
    appendU64(out, raw);
}

inline std::vector<uint8_t> readFile(const std::string& path) {
    std::ifstream input(path.c_str(), std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open file: " + path);
    const std::streamoff end = input.tellg();
    if (end < 0) throw std::runtime_error("cannot size file: " + path);
    input.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(end));
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char*>(&bytes[0]), end);
        if (!input) throw std::runtime_error("cannot read file: " + path);
    }
    return bytes;
}

inline void writeFile(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream output(path.c_str(), std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create file: " + path);
    if (!bytes.empty())
        output.write(reinterpret_cast<const char*>(&bytes[0]),
                     static_cast<std::streamsize>(bytes.size()));
    if (!output) throw std::runtime_error("cannot write file: " + path);
}

inline std::array<uint8_t, 8> magic(const char (&value)[9]) {
    std::array<uint8_t, 8> result;
    for (size_t i = 0; i < 8U; ++i) result[i] = static_cast<uint8_t>(value[i]);
    return result;
}

inline Header parseHeader(
    const std::vector<uint8_t>& bytes,
    const std::array<uint8_t, 8>& expected_magic,
    uint32_t expected_record_size) {
    if (bytes.size() < kHeaderSize) throw std::runtime_error("truncated header");
    Header header;
    std::copy(bytes.begin(), bytes.begin() + 8, header.magic.begin());
    header.schema_version = readU32(&bytes[8]);
    header.header_size = readU32(&bytes[12]);
    header.record_size = readU32(&bytes[16]);
    header.dimension = readU32(&bytes[20]);
    header.record_count = readU64(&bytes[24]);
    std::copy(bytes.begin() + 32, bytes.begin() + 64, header.identity.begin());
    if (header.magic != expected_magic || header.schema_version != kSchemaVersion ||
        header.header_size != kHeaderSize ||
        header.record_size != expected_record_size) {
        throw std::runtime_error("unsupported binary header");
    }
    if (header.record_count >
        (std::numeric_limits<uint64_t>::max() - kHeaderSize) /
            expected_record_size) {
        throw std::runtime_error("binary record count overflow");
    }
    const uint64_t expected_size = kHeaderSize +
        header.record_count * static_cast<uint64_t>(expected_record_size);
    if (expected_size != bytes.size()) throw std::runtime_error("binary size mismatch");
    return header;
}

inline void appendHeader(
    std::vector<uint8_t>& out,
    const std::array<uint8_t, 8>& file_magic,
    uint32_t record_size,
    uint32_t dimension,
    uint64_t record_count,
    const std::array<uint8_t, 32>& identity) {
    out.insert(out.end(), file_magic.begin(), file_magic.end());
    appendU32(out, kSchemaVersion);
    appendU32(out, kHeaderSize);
    appendU32(out, record_size);
    appendU32(out, dimension);
    appendU64(out, record_count);
    out.insert(out.end(), identity.begin(), identity.end());
}

}  // namespace detail

inline void validateEvents(const std::vector<EventRecord>& events) {
    bool in_query = false;
    bool in_source = false;
    uint64_t query_id = 0U;
    uint64_t expansion_id = 0U;
    for (size_t i = 0; i < events.size(); ++i) {
        const EventRecord& event = events[i];
        if (event.event_id != i) throw std::runtime_error("non-contiguous event id");
        if ((event.flags & ~static_cast<uint8_t>(7U)) != 0U)
            throw std::runtime_error("reserved event flag set");
        switch (event.kind) {
            case EventKind::QueryBegin:
                if (in_query || event.graph_layer != -1 || event.flags != 0U ||
                    event.expansion_id != std::numeric_limits<uint64_t>::max() ||
                    event.edge_id != std::numeric_limits<uint64_t>::max() ||
                    event.source_id != std::numeric_limits<uint32_t>::max() ||
                    event.target_id != std::numeric_limits<uint32_t>::max() ||
                    event.neighbor_slot != std::numeric_limits<uint32_t>::max() ||
                    event.source_degree != 0U || event.d_current != 0.0 ||
                    event.threshold_before != 0.0)
                    throw std::runtime_error("invalid QUERY_BEGIN");
                in_query = true; in_source = false; query_id = event.query_id;
                break;
            case EventKind::SourceBegin:
                if (!in_query || event.query_id != query_id || event.graph_layer != -1 ||
                    event.flags != 0U ||
                    event.expansion_id == std::numeric_limits<uint64_t>::max() ||
                    event.edge_id != std::numeric_limits<uint64_t>::max() ||
                    event.source_id == std::numeric_limits<uint32_t>::max() ||
                    event.target_id != std::numeric_limits<uint32_t>::max() ||
                    event.neighbor_slot != std::numeric_limits<uint32_t>::max() ||
                    !std::isfinite(event.d_current) || event.d_current < 0.0 ||
                    event.threshold_before != 0.0)
                    throw std::runtime_error("invalid SOURCE_BEGIN");
                in_source = true; expansion_id = event.expansion_id;
                break;
            case EventKind::Candidate:
                if (!in_query || !in_source || event.query_id != query_id ||
                    event.expansion_id != expansion_id || event.graph_layer != 0 ||
                    event.edge_id == std::numeric_limits<uint64_t>::max() ||
                    event.source_id == std::numeric_limits<uint32_t>::max() ||
                    event.target_id == std::numeric_limits<uint32_t>::max() ||
                    event.neighbor_slot == std::numeric_limits<uint32_t>::max() ||
                    event.neighbor_slot >= event.source_degree ||
                    (event.flags & FirstVisit) == 0U ||
                    ((event.flags & ScoreSlotEligible) != 0U &&
                     (event.flags & ThresholdValid) == 0U) ||
                    !std::isfinite(event.d_current) || event.d_current < 0.0)
                    throw std::runtime_error("invalid CANDIDATE");
                if ((event.flags & ThresholdValid) == 0U &&
                    event.threshold_before != 0.0)
                    throw std::runtime_error("invalid threshold encoding");
                if ((event.flags & ThresholdValid) != 0U &&
                    (!std::isfinite(event.threshold_before) || event.threshold_before < 0.0))
                    throw std::runtime_error("invalid threshold value");
                break;
            case EventKind::ExactOnly:
                if (!in_query || event.query_id != query_id || event.flags != 0U ||
                    event.edge_id != std::numeric_limits<uint64_t>::max() ||
                    event.target_id == std::numeric_limits<uint32_t>::max() ||
                    event.neighbor_slot != std::numeric_limits<uint32_t>::max() ||
                    event.threshold_before != 0.0)
                    throw std::runtime_error("invalid EXACT_ONLY");
                break;
            case EventKind::QueryEnd:
                if (!in_query || event.query_id != query_id || event.graph_layer != -1 ||
                    event.flags != 0U ||
                    event.expansion_id != std::numeric_limits<uint64_t>::max() ||
                    event.edge_id != std::numeric_limits<uint64_t>::max() ||
                    event.source_id != std::numeric_limits<uint32_t>::max() ||
                    event.target_id != std::numeric_limits<uint32_t>::max() ||
                    event.neighbor_slot != std::numeric_limits<uint32_t>::max() ||
                    event.source_degree != 0U || event.d_current != 0.0 ||
                    event.threshold_before != 0.0)
                    throw std::runtime_error("invalid QUERY_END");
                in_query = false; in_source = false;
                break;
            default:
                throw std::runtime_error("unknown event kind");
        }
    }
    if (in_query) throw std::runtime_error("unterminated query");
}

inline std::vector<EventRecord> readEvents(
    const std::string& path, Header* output_header = NULL) {
    const std::vector<uint8_t> bytes = detail::readFile(path);
    const Header header = detail::parseHeader(
        bytes, detail::magic("UQEV0001"), kEventRecordSize);
    std::vector<EventRecord> events;
    events.reserve(static_cast<size_t>(header.record_count));
    for (uint64_t i = 0; i < header.record_count; ++i) {
        const uint8_t* p = &bytes[static_cast<size_t>(kHeaderSize + i * kEventRecordSize)];
        if (detail::readU32(p) >> 16U != 0U)
            throw std::runtime_error("event reserved16 is non-zero");
        EventRecord event;
        event.kind = static_cast<EventKind>(p[0]);
        event.flags = p[1];
        event.graph_layer = detail::readI32(p + 4);
        event.event_id = detail::readU64(p + 8);
        event.query_id = detail::readU64(p + 16);
        event.expansion_id = detail::readU64(p + 24);
        event.edge_id = detail::readU64(p + 32);
        event.source_id = detail::readU32(p + 40);
        event.target_id = detail::readU32(p + 44);
        event.neighbor_slot = detail::readU32(p + 48);
        event.source_degree = detail::readU32(p + 52);
        event.d_current = detail::readF64(p + 56);
        event.threshold_before = detail::readF64(p + 64);
        if (detail::readU64(p + 72) != 0U)
            throw std::runtime_error("event reserved64 is non-zero");
        events.push_back(event);
    }
    validateEvents(events);
    if (output_header != NULL) *output_header = header;
    return events;
}

inline void writeEvents(
    const std::string& path,
    uint32_t dimension,
    const std::array<uint8_t, 32>& catalog_identity,
    const std::vector<EventRecord>& events) {
    validateEvents(events);
    std::vector<uint8_t> out;
    out.reserve(kHeaderSize + events.size() * kEventRecordSize);
    detail::appendHeader(out, detail::magic("UQEV0001"), kEventRecordSize,
                         dimension, events.size(), catalog_identity);
    for (size_t i = 0; i < events.size(); ++i) {
        const EventRecord& event = events[i];
        out.push_back(static_cast<uint8_t>(event.kind));
        out.push_back(event.flags);
        out.push_back(0U); out.push_back(0U);
        detail::appendI32(out, event.graph_layer);
        detail::appendU64(out, event.event_id);
        detail::appendU64(out, event.query_id);
        detail::appendU64(out, event.expansion_id);
        detail::appendU64(out, event.edge_id);
        detail::appendU32(out, event.source_id);
        detail::appendU32(out, event.target_id);
        detail::appendU32(out, event.neighbor_slot);
        detail::appendU32(out, event.source_degree);
        detail::appendF64(out, event.d_current);
        detail::appendF64(out, event.threshold_before);
        detail::appendU64(out, 0U);
    }
    detail::writeFile(path, out);
}

inline std::vector<LabelRecord> readLabels(
    const std::string& path, Header* output_header = NULL) {
    const std::vector<uint8_t> bytes = detail::readFile(path);
    const Header header = detail::parseHeader(
        bytes, detail::magic("UQLB0001"), kLabelRecordSize);
    std::vector<LabelRecord> labels;
    labels.reserve(static_cast<size_t>(header.record_count));
    uint64_t previous = 0U;
    for (uint64_t i = 0; i < header.record_count; ++i) {
        const uint8_t* p = &bytes[static_cast<size_t>(kHeaderSize + i * kLabelRecordSize)];
        LabelRecord label{detail::readU64(p), detail::readF64(p + 8)};
        if (!std::isfinite(label.exact_squared_distance) ||
            (i != 0U && label.event_id <= previous))
            throw std::runtime_error("invalid label record");
        previous = label.event_id;
        labels.push_back(label);
    }
    if (output_header != NULL) *output_header = header;
    return labels;
}

inline void writeLabels(
    const std::string& path,
    uint32_t dimension,
    const std::array<uint8_t, 32>& identity,
    const std::vector<LabelRecord>& labels) {
    std::vector<uint8_t> out;
    detail::appendHeader(out, detail::magic("UQLB0001"), kLabelRecordSize,
                         dimension, labels.size(), identity);
    uint64_t previous = 0U;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (!std::isfinite(labels[i].exact_squared_distance) ||
            (i != 0U && labels[i].event_id <= previous))
            throw std::runtime_error("invalid label record");
        previous = labels[i].event_id;
        detail::appendU64(out, labels[i].event_id);
        detail::appendF64(out, labels[i].exact_squared_distance);
    }
    detail::writeFile(path, out);
}

inline std::vector<QueryRangeRecord> readQueryRanges(
    const std::string& path, Header* output_header = NULL) {
    const std::vector<uint8_t> bytes = detail::readFile(path);
    const Header header = detail::parseHeader(
        bytes, detail::magic("UQQR0001"), kQueryRangeRecordSize);
    std::vector<QueryRangeRecord> ranges;
    uint64_t expected_begin = 0U;
    for (uint64_t i = 0; i < header.record_count; ++i) {
        const uint8_t* p = &bytes[static_cast<size_t>(kHeaderSize + i * kQueryRangeRecordSize)];
        QueryRangeRecord range{detail::readU64(p), detail::readU64(p + 8),
                               detail::readU64(p + 16)};
        if (range.begin_event != expected_begin ||
            range.event_count > std::numeric_limits<uint64_t>::max() - expected_begin)
            throw std::runtime_error("non-contiguous query ranges");
        expected_begin += range.event_count;
        ranges.push_back(range);
    }
    if (output_header != NULL) *output_header = header;
    return ranges;
}

inline void writeQueryRanges(
    const std::string& path,
    uint32_t dimension,
    const std::array<uint8_t, 32>& identity,
    const std::vector<QueryRangeRecord>& ranges) {
    std::vector<uint8_t> out;
    detail::appendHeader(out, detail::magic("UQQR0001"), kQueryRangeRecordSize,
                         dimension, ranges.size(), identity);
    uint64_t expected_begin = 0U;
    for (size_t i = 0; i < ranges.size(); ++i) {
        if (ranges[i].begin_event != expected_begin ||
            ranges[i].event_count > std::numeric_limits<uint64_t>::max() - expected_begin)
            throw std::runtime_error("non-contiguous query ranges");
        expected_begin += ranges[i].event_count;
        detail::appendU64(out, ranges[i].query_id);
        detail::appendU64(out, ranges[i].begin_event);
        detail::appendU64(out, ranges[i].event_count);
    }
    detail::writeFile(path, out);
}

inline void validateDataset(
    const std::vector<EventRecord>& events,
    const std::vector<LabelRecord>& labels,
    const std::vector<QueryRangeRecord>& ranges) {
    validateEvents(events);
    size_t label_index = 0U;
    for (size_t i = 0; i < events.size(); ++i) {
        const bool needs_label = events[i].kind == EventKind::Candidate ||
            events[i].kind == EventKind::ExactOnly;
        if (needs_label) {
            if (label_index >= labels.size() || labels[label_index].event_id != i)
                throw std::runtime_error("missing or mismatched event label");
            ++label_index;
        }
    }
    if (label_index != labels.size()) throw std::runtime_error("extra event label");
    uint64_t covered = 0U;
    for (size_t i = 0; i < ranges.size(); ++i) {
        if (ranges[i].begin_event != covered || ranges[i].event_count == 0U ||
            ranges[i].event_count > events.size() - covered)
            throw std::runtime_error("invalid query range coverage");
        const size_t begin = static_cast<size_t>(covered);
        const size_t end = begin + static_cast<size_t>(ranges[i].event_count);
        if (events[begin].kind != EventKind::QueryBegin ||
            events[end - 1U].kind != EventKind::QueryEnd ||
            events[begin].query_id != ranges[i].query_id ||
            events[end - 1U].query_id != ranges[i].query_id)
            throw std::runtime_error("query range boundary mismatch");
        covered += ranges[i].event_count;
    }
    if (covered != events.size()) throw std::runtime_error("query ranges do not cover events");
}

}  // namespace uq
