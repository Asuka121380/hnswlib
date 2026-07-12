#pragma once

#include "baseline_trace.h"
#include <iomanip>
#include <ostream>

namespace hnswlib {

class BaselineTraceCsvWriter {
 public:
    static void writeQueryHeader(std::ostream& out) {
        out << "query_id,requested_k,ef_search,ef_effective,dimension,seed,"
            << "n_entry_distance,n_upper_edge_scan,n_upper_dist,n_base_entry_distance,"
            << "n_edge_scan,n_duplicate,n_unique_neighbor,n_dist,n_expanded,"
            << "n_inserted_candidate,n_inserted_result,n_threshold_changed,"
            << "n_expanded_later,n_final_topk,n_strict_state_neutral,trace_records_written,"
            << "n_exact_calls_total,baseline_query_latency_ns,trace_query_latency_ns,"
            << "exact_distance_time_ns,recall_at_k\n";
    }

    static void writeQuery(std::ostream& out, const BaselineQuerySummary& value) {
        out << std::setprecision(17)
            << value.query_id << ',' << value.requested_k << ',' << value.ef_search << ','
            << value.ef_effective << ',' << value.dimension << ',' << value.seed << ','
            << value.n_entry_distance << ',' << value.n_upper_edge_scan << ',' << value.n_upper_dist << ','
            << value.n_base_entry_distance << ',' << value.n_edge_scan << ',' << value.n_duplicate << ','
            << value.n_unique_neighbor << ',' << value.n_dist << ',' << value.n_expanded << ','
            << value.n_inserted_candidate << ',' << value.n_inserted_result << ','
            << value.n_threshold_changed << ',' << value.n_expanded_later << ',' << value.n_final_topk << ','
            << value.n_strict_state_neutral << ',' << value.trace_records_written << ','
            << value.nExactCallsTotal() << ',' << value.baseline_query_latency_ns << ','
            << value.trace_query_latency_ns << ',' << value.exact_distance_time_ns << ','
            << value.recall_at_k << '\n';
    }

    static void writeDcoHeader(std::ostream& out) {
        out << "query_id,graph_layer,expansion_index,dco_index,current_node_id,current_node_label,"
            << "neighbor_id,neighbor_label,dist_qc,dist_qd,threshold_before,threshold_after,"
            << "threshold_valid_before,threshold_valid_after,candidate_queue_size_before,"
            << "candidate_queue_size_after,result_queue_size_before,result_queue_size_after,"
            << "inserted_candidate_queue,inserted_result_queue,threshold_changed,expanded_later,"
            << "in_final_topk,geometry_valid,edge_length_cd,edge_dot_qcd,edge_cosine_qcd,"
            << "triangle_lower_bound,absolute_margin,relative_margin,is_negative_at_evaluation,"
            << "is_state_neutral,search_progress_fraction,threshold_stability_indicator\n";
    }

    static void writeDco(std::ostream& out, const BaselineDcoRecord& value) {
        out << std::setprecision(17)
            << value.query_id << ',' << value.graph_layer << ',' << value.expansion_index << ','
            << value.dco_index << ',' << value.current_node_id << ',' << value.current_node_label << ','
            << value.neighbor_id << ',' << value.neighbor_label << ',' << value.dist_qc << ','
            << value.dist_qd << ',' << value.threshold_before << ',' << value.threshold_after << ','
            << value.threshold_valid_before << ',' << value.threshold_valid_after << ','
            << value.candidate_queue_size_before << ',' << value.candidate_queue_size_after << ','
            << value.result_queue_size_before << ',' << value.result_queue_size_after << ','
            << value.inserted_candidate_queue << ',' << value.inserted_result_queue << ','
            << value.threshold_changed << ',' << value.expanded_later << ',' << value.in_final_topk << ','
            << value.geometry_valid << ',' << value.edge_length_cd << ',' << value.edge_dot_qcd << ','
            << value.edge_cosine_qcd << ',' << value.triangle_lower_bound << ',' << value.absolute_margin << ','
            << value.relative_margin << ',' << value.is_negative_at_evaluation << ','
            << value.is_state_neutral << ',' << value.search_progress_fraction << ','
            << value.threshold_stability_indicator << '\n';
    }
};

}  // namespace hnswlib
