#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "hnswlib/hnswlib.h"

namespace {
struct Options {
    std::string index, sidecar, queries, input, events, edges;
    uint32_t dimension = 0U;
};
Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        if (key == "--help") {
            std::cout << "v0_residual_exporter --index FILE --sidecar FILE "
                         "--queries FVECS --input CSV --events CSV --edges BIN "
                         "--dimension N\n";
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::invalid_argument("missing exporter option value");
        const std::string value(argv[++i]);
        if (key == "--index") o.index = value;
        else if (key == "--sidecar") o.sidecar = value;
        else if (key == "--queries") o.queries = value;
        else if (key == "--input") o.input = value;
        else if (key == "--events") o.events = value;
        else if (key == "--edges") o.edges = value;
        else if (key == "--dimension") o.dimension = std::stoul(value);
        else throw std::invalid_argument("unknown exporter option: " + key);
    }
    if (!o.dimension || o.index.empty() || o.sidecar.empty() ||
        o.queries.empty() || o.input.empty() || o.events.empty() ||
        o.edges.empty()) throw std::invalid_argument("incomplete exporter options");
    return o;
}
std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> fields;
    size_t first = 0U;
    while (true) {
        const size_t end = line.find(',', first);
        fields.push_back(line.substr(first, end - first));
        if (end == std::string::npos) return fields;
        first = end + 1U;
    }
}
size_t column(const std::vector<std::string>& header, const std::string& name) {
    for (size_t i = 0; i < header.size(); ++i)
        if (header[i] == name) return i;
    throw std::runtime_error("missing input column: " + name);
}
std::vector<float> readQueries(const Options& o) {
    std::ifstream in(o.queries.c_str(), std::ios::binary);
    if (!in) throw std::runtime_error("cannot open query fvecs");
    std::vector<float> rows;
    while (true) {
        int32_t dimension = 0;
        in.read(reinterpret_cast<char*>(&dimension), 4U);
        if (in.eof() && in.gcount() == 0) break;
        if (!in || dimension != static_cast<int32_t>(o.dimension))
            throw std::runtime_error("query fvecs dimension mismatch");
        const size_t old = rows.size();
        rows.resize(old + o.dimension);
        in.read(reinterpret_cast<char*>(rows.data() + old),
                static_cast<std::streamsize>(o.dimension * 4U));
        if (!in) throw std::runtime_error("truncated query fvecs");
    }
    return rows;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        if (!hnswlib::edge_quant_v0_detail::nativeIsLittleEndian())
            throw std::runtime_error("exporter requires little-endian host");
        hnswlib::L2Space space(o.dimension);
        const hnswlib::DISTFUNC<float> exact_l2 = space.get_dist_func();
        const void* exact_l2_param = space.get_dist_func_param();
        hnswlib::HierarchicalNSW<float> index(&space, o.index);
        index.loadEdgeQuantV0Metadata(o.sidecar);
        const hnswlib::EdgeQuantV0Metadata& meta = index.getEdgeQuantV0Metadata();
        const hnswlib::V0Layer0GraphView graph = index.getV0Layer0GraphView();
        const hnswlib::V0SidecarHeader& sh = meta.header();
        if (sh.dimension != o.dimension) throw std::runtime_error("dimension mismatch");
        const std::vector<float> queries = readQueries(o);
        std::ifstream input(o.input.c_str());
        std::ofstream events(o.events.c_str());
        std::ofstream edges(o.edges.c_str(), std::ios::binary);
        if (!input || !events || !edges) throw std::runtime_error("cannot open exporter files");
        std::string line;
        if (!std::getline(input, line)) throw std::runtime_error("empty event CSV");
        const std::vector<std::string> header = split(line);
        const size_t query_col = column(header, "query_id");
        const size_t source_col = column(header, "current_node_id");
        const size_t target_col = column(header, "candidate_id");
        const size_t layer_col = column(header, "graph_layer");
        const size_t tau_col = column(header, "threshold");
        const size_t old_exact_col = column(header, "exact_squared_distance");
        const size_t old_current_col = column(header, "current_squared_distance");
        uint64_t edge_count = 0U, event_count = 0U;
        edges.write("V0RGE002", 8U);
        edges.write(reinterpret_cast<const char*>(&o.dimension), 4U);
        edges.write(reinterpret_cast<const char*>(&edge_count), 8U);
        edges.write(reinterpret_cast<const char*>(&sh.pq_m), 4U);
        events << "source_row,query_id,current_internal_id,candidate_internal_id,"
                  "neighbor_slot,edge_ordinal,edge_index,tau,D_exact_cpp,Dc_exact_cpp,"
                  "Dhat_pq_cpp,edge_eligible,old_exact,old_current\n";
        events << std::setprecision(17);
        std::unordered_map<uint64_t, uint64_t> unique;
        std::unordered_map<uint64_t,
            std::unique_ptr<hnswlib::EdgeQuantV0ApproxQueryContext> > raw_queries;
        std::vector<float> direction(o.dimension);
        std::vector<double> z(o.dimension);
        while (std::getline(input, line)) {
            if (line.empty()) continue;
            const std::vector<std::string> fields = split(line);
            if (fields.size() != header.size())
                throw std::runtime_error("malformed event CSV row");
            if (std::stoul(fields[layer_col]) != 0U)
                throw std::runtime_error("only layer zero events are supported");
            const uint64_t query_id = std::stoull(fields[query_col]);
            const uint64_t source_id = std::stoull(fields[source_col]);
            const uint64_t target_id = std::stoull(fields[target_col]);
            if (query_id >= queries.size() / o.dimension ||
                source_id >= graph.nodeCount() || target_id >= graph.nodeCount())
                throw std::runtime_error("event ID out of range");
            const hnswlib::V0Layer0NeighborSpan neighbors = graph.neighbors(
                static_cast<hnswlib::tableint>(source_id));
            size_t slot = neighbors.size;
            for (size_t i = 0; i < neighbors.size; ++i) {
                if (neighbors.ids[i] == target_id) {
                    if (slot != neighbors.size)
                        throw std::runtime_error("ambiguous repeated neighbor in trace");
                    slot = i;
                }
            }
            if (slot == neighbors.size)
                throw std::runtime_error("trace edge absent from frozen graph");
            const uint64_t ordinal = meta.edgeIndex(
                static_cast<hnswlib::tableint>(source_id), slot);
            const hnswlib::V0EdgeRecordView edge = meta.edgeRecord(
                static_cast<hnswlib::tableint>(source_id), slot);
            const float* c = graph.floatVector(static_cast<hnswlib::tableint>(source_id));
            const float* v = graph.floatVector(static_cast<hnswlib::tableint>(target_id));
            const float* q = queries.data() + query_id * o.dimension;
            const float dc = exact_l2(q, c, exact_l2_param);
            const float d = exact_l2(q, v, exact_l2_param);
            if (raw_queries.find(query_id) == raw_queries.end())
                raw_queries[query_id].reset(new hnswlib::EdgeQuantV0ApproxQueryContext(
                    q, sh, meta.nativeCodebookData()));
            const hnswlib::V0RawEstimateResult estimate =
                raw_queries[query_id]->evaluateRawFast(edge, dc);
            std::unordered_map<uint64_t, uint64_t>::const_iterator found = unique.find(ordinal);
            uint64_t edge_index = 0U;
            if (found == unique.end()) {
                edge_index = edge_count++;
                unique[ordinal] = edge_index;
                const double length = edge.edgeLengthNativeUnchecked();
                const double anchor = edge.anchorProjectionNativeUnchecked();
                const uint8_t* code = edge.codeDataUnchecked();
                double delta_norm = 0.0, centroid_dot = 0.0;
                for (uint32_t j = 0; j < o.dimension; ++j) {
                    const uint32_t sub = j / sh.pq_dsub;
                    const size_t flat = (static_cast<size_t>(sub) * sh.pq_ksub +
                        code[sub]) * sh.pq_dsub + j % sh.pq_dsub;
                    direction[j] = meta.nativeCodebookData()[flat];
                    const double delta = static_cast<double>(v[j]) - c[j];
                    z[j] = delta - length * direction[j];
                    delta_norm += delta * delta;
                    centroid_dot += static_cast<double>(c[j]) * direction[j];
                }
                const double bias = length * length - delta_norm +
                    2.0 * length * (anchor - centroid_dot);
                const uint32_t source32 = static_cast<uint32_t>(source_id);
                const uint32_t target32 = static_cast<uint32_t>(target_id);
                const uint32_t slot32 = static_cast<uint32_t>(slot);
                const uint32_t flags32 = edge.flags();
                edges.write(reinterpret_cast<const char*>(&ordinal), 8U);
                edges.write(reinterpret_cast<const char*>(&source32), 4U);
                edges.write(reinterpret_cast<const char*>(&target32), 4U);
                edges.write(reinterpret_cast<const char*>(&slot32), 4U);
                edges.write(reinterpret_cast<const char*>(&flags32), 4U);
                edges.write(reinterpret_cast<const char*>(&length), 8U);
                edges.write(reinterpret_cast<const char*>(&anchor), 8U);
                edges.write(reinterpret_cast<const char*>(&bias), 8U);
                edges.write(reinterpret_cast<const char*>(c), o.dimension * 4U);
                edges.write(reinterpret_cast<const char*>(z.data()), o.dimension * 8U);
                edges.write(reinterpret_cast<const char*>(code), sh.pq_m);
                if (!edges) throw std::runtime_error("cannot write edge geometry");
            } else edge_index = found->second;
            events << event_count << ',' << query_id << ',' << source_id << ','
                   << target_id << ',' << slot << ',' << ordinal << ','
                   << edge_index << ',' << std::stod(fields[tau_col]) << ','
                   << d << ',' << dc << ',';
            if (estimate.valid()) events << estimate.approximate_squared_distance;
            events << ',' << (estimate.valid() ? 1 : 0) << ','
                   << std::stod(fields[old_exact_col]) << ','
                   << std::stod(fields[old_current_col]) << '\n';
            ++event_count;
        }
        edges.seekp(12U);
        edges.write(reinterpret_cast<const char*>(&edge_count), 8U);
        if (!events || !edges) throw std::runtime_error("cannot finish exporter output");
        std::cout << "events=" << event_count << " unique_edges=" << edge_count << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
