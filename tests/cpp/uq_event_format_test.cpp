#include <array>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "tools/edge_estimation/event_format.h"

namespace {

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

uq::EventRecord event(
    uq::EventKind kind, uint64_t id, uint64_t query, int32_t layer) {
    uq::EventRecord result{};
    result.kind = kind; result.event_id = id; result.query_id = query;
    result.graph_layer = layer;
    result.expansion_id = std::numeric_limits<uint64_t>::max();
    result.edge_id = std::numeric_limits<uint64_t>::max();
    result.source_id = std::numeric_limits<uint32_t>::max();
    result.target_id = std::numeric_limits<uint32_t>::max();
    result.neighbor_slot = std::numeric_limits<uint32_t>::max();
    return result;
}

void expectFailure(const std::string& path) {
    try { (void)uq::readEvents(path); }
    catch (const std::exception&) { return; }
    throw std::runtime_error("corrupt event file was accepted");
}

}  // namespace

int main() {
    const std::string events_path = "uq_events_test.bin";
    const std::string labels_path = "uq_labels_test.bin";
    const std::string ranges_path = "uq_ranges_test.bin";
    const std::string corrupt_path = "uq_events_corrupt_test.bin";
    std::array<uint8_t, 32> identity{};
    identity[0] = 17U;

    std::vector<uq::EventRecord> events;
    events.push_back(event(uq::EventKind::QueryBegin, 0, 41, -1));
    uq::EventRecord source = event(uq::EventKind::SourceBegin, 1, 41, -1);
    source.expansion_id = 0; source.source_id = 7; source.source_degree = 2;
    source.d_current = 3.5; events.push_back(source);
    uq::EventRecord candidate = event(uq::EventKind::Candidate, 2, 41, 0);
    candidate.flags = uq::ThresholdValid | uq::FirstVisit | uq::ScoreSlotEligible;
    candidate.expansion_id = 0; candidate.edge_id = 19; candidate.source_id = 7;
    candidate.target_id = 9; candidate.neighbor_slot = 1; candidate.source_degree = 2;
    candidate.d_current = 3.5; candidate.threshold_before = 8.0;
    events.push_back(candidate);
    events.push_back(event(uq::EventKind::QueryEnd, 3, 41, -1));
    const std::vector<uq::LabelRecord> labels{{2, 9.25}};
    const std::vector<uq::QueryRangeRecord> ranges{{41, 0, 4}};

    uq::writeEvents(events_path, 16, identity, events);
    uq::writeLabels(labels_path, 16, identity, labels);
    uq::writeQueryRanges(ranges_path, 16, identity, ranges);
    uq::Header header;
    const std::vector<uq::EventRecord> decoded = uq::readEvents(events_path, &header);
    const std::vector<uq::LabelRecord> decoded_labels = uq::readLabels(labels_path);
    const std::vector<uq::QueryRangeRecord> decoded_ranges = uq::readQueryRanges(ranges_path);
    uq::validateDataset(decoded, decoded_labels, decoded_ranges);
    require(header.record_size == 80U, "wrong event record size");
    require(decoded[2].edge_id == 19U, "event roundtrip failed");

    {
        std::ifstream input(events_path.c_str(), std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                                std::istreambuf_iterator<char>());
        bytes.pop_back();
        std::ofstream output(corrupt_path.c_str(), std::ios::binary);
        output.write(&bytes[0], static_cast<std::streamsize>(bytes.size()));
    }
    expectFailure(corrupt_path);
    std::remove(events_path.c_str()); std::remove(labels_path.c_str());
    std::remove(ranges_path.c_str()); std::remove(corrupt_path.c_str());
    std::cout << "uq_event_format_test_ok" << std::endl;
    return 0;
}
