#pragma once
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif
#include "query_selection.h"
#include "query_store.h"
#include "distance_counter_space.h"
#include "live_opq_pruner.h"

namespace uq { namespace performance {
using Clock = std::chrono::steady_clock;
inline uint64_t elapsed(Clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-t).count();
}
inline uint64_t peakRss() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS m{};
    return GetProcessMemoryInfo(GetCurrentProcess(), &m, sizeof(m)) ? m.PeakWorkingSetSize : 0;
#else
    struct rusage m{};
    if(getrusage(RUSAGE_SELF,&m)) return 0;
#ifdef __APPLE__
    return m.ru_maxrss;
#else
    return uint64_t(m.ru_maxrss)*1024;
#endif
#endif
}
struct Options {
    std::string index,artifact,queries,ids,warmup_ids,output,results,latency,metrics="basic";
    size_t dimension=0,start=0,count=0,k=10,ef=200,window=128,chunk=128,warmup=100,repeats=1;
    size_t max_result_bytes=1024ULL*1024*1024;
    double beta=1.0;
    bool no_prune=false,range=false,schedule=false,batch_alias=false;
};
inline Options parse(int argc,char** argv) {
    Options o; std::set<std::string> seen;
    for(int i=1;i<argc;++i) {
        const std::string key=argv[i];
        if(!seen.insert(key).second) throw std::invalid_argument("duplicate option: "+key);
        if(key=="--help") {
            std::cout<<"--index-path FILE [--artifact-path DIR | --no-prune] --query-path FILE --dimension D "
                "(--query-ids FILE | --query-start N --query-count N) --output FILE "
                "[--warmup-query-ids FILE] [--warmup-queries N] [--k N] [--ef-search N] [--beta X] "
                "[--prepare-window N|all --compute-chunk N|all] [--batch-size N] [--repeats N] "
                "[--metrics-level off|basic|diagnostic] [--max-result-bytes N] "
                "[--result-records-output FILE] [--latency-records-output FILE]\n";
            std::exit(0);
        }
        if(key=="--no-prune") {o.no_prune=true;continue;}
        if(++i>=argc) throw std::invalid_argument("missing value: "+key);
        const std::string v=argv[i];
        if(key=="--index-path") o.index=v;
        else if(key=="--artifact-path") o.artifact=v;
        else if(key=="--query-path") o.queries=v;
        else if(key=="--query-ids") o.ids=v;
        else if(key=="--warmup-query-ids") o.warmup_ids=v;
        else if(key=="--output") o.output=v;
        else if(key=="--result-records-output") o.results=v;
        else if(key=="--latency-records-output") o.latency=v;
        else if(key=="--metrics-level") o.metrics=v;
        else if(key=="--beta") {
            size_t end=0;o.beta=std::stod(v,&end);
            if(end!=v.size()) throw std::invalid_argument("invalid beta");
        } else if(key=="--prepare-window" || key=="--compute-chunk") {
            size_t n=v=="all"?0:parseUnsignedSize(v,key.c_str());
            if(!n && v!="all") throw std::invalid_argument("window/chunk must be positive or all");
            if(key=="--prepare-window") o.window=n; else o.chunk=n;
            o.schedule=true;
        } else {
            size_t n=parseUnsignedSize(v,key.c_str());
            if(key=="--dimension") o.dimension=n;
            else if(key=="--query-start") {o.start=n;o.range=true;}
            else if(key=="--query-count") {o.count=n;o.range=true;}
            else if(key=="--k") o.k=n;
            else if(key=="--ef-search") o.ef=n;
            else if(key=="--warmup-queries") o.warmup=n;
            else if(key=="--repeats") o.repeats=n;
            else if(key=="--max-result-bytes") o.max_result_bytes=n;
            else if(key=="--batch-size") {
                if(!n) throw std::invalid_argument("batch must be positive");
                o.window=o.chunk=n;o.batch_alias=true;
            } else throw std::invalid_argument("unknown option: "+key);
        }
    }
    if(o.index.empty() || o.queries.empty() || o.output.empty() || !o.dimension ||
       !o.k || o.ef<o.k || !o.repeats || (!o.no_prune && o.artifact.empty()) ||
       !std::isfinite(o.beta) || o.beta<0 || (!o.ids.empty() && o.range) ||
       (o.batch_alias && o.schedule) || (o.metrics!="off" && o.metrics!="basic" && o.metrics!="diagnostic"))
        throw std::invalid_argument("invalid performance contract");
    if(o.chunk && o.window && o.chunk>o.window)
        throw std::invalid_argument("compute chunk exceeds preparation window");
    return o;
}
struct Json {
    std::ostream& out; bool first=true;
    explicit Json(std::ostream& o):out(o) {out<<"{"<<std::setprecision(17);}
    void key(const char* k) {if(!first)out<<",";first=false;out<<"\n\""<<k<<"\":";}
    template<class T> void number(const char* k,T value) {key(k);out<<value;}
    void text(const char* k,const std::string& value) {key(k);out<<std::quoted(value);}
    void boolean(const char* k,bool value) {key(k);out<<(value?"true":"false");}
    void nullable(const char* k,uint64_t value,bool present) {key(k);if(present)out<<value;else out<<"null";}
    void finish() {out<<"\n}\n";if(!out)throw std::runtime_error("JSON write failed");}
};
struct Neighbor {hnswlib::labeltype label=0;float distance=0;};
struct Sample {uint64_t repeat,query,exact;};
struct Repeat {uint64_t service=0,prepare=0,search=0;};
inline size_t product(size_t a,size_t b) {
    if(b && a>std::numeric_limits<size_t>::max()/b) throw std::length_error("buffer size overflow");
    return a*b;
}
inline uint64_t checksum(const std::vector<Neighbor>& values,size_t first,size_t count) {
    uint64_t hash=1469598103934665603ULL;
    for(size_t i=first;i<first+count;++i) {
        uint32_t bits;std::memcpy(&bits,&values[i].distance,sizeof(bits));
        for(uint64_t word:{uint64_t(values[i].label),uint64_t(bits)}){hash^=word;hash*=1099511628211ULL;}
    }
    return hash;
}
template<class Kernel,bool Collect> int execute(const Options& o) {
    const auto q0=Clock::now();
    auto queries=std::make_shared<const QueryStore>(o.queries,uint32_t(o.dimension));
    const auto ids=o.ids.empty()?queryRange(o.start,o.count,queries->count()):readQueryIds(o.ids,queries->count());
    const auto warm=o.warmup_ids.empty()?std::vector<uint64_t>(ids.begin(),ids.begin()+std::min(o.warmup,ids.size()))
                                       :readQueryIds(o.warmup_ids,queries->count());
    if(!o.warmup_ids.empty()) {
        std::set<uint64_t> measured(ids.begin(),ids.end());
        for(auto id:warm)if(measured.count(id))throw std::invalid_argument("warmup overlaps measured IDs");
    }
    const uint64_t query_load=elapsed(q0);
    hnswlib::L2Space l2(o.dimension);DistanceCounterSpace counted(l2);
    const bool diagnostic=o.metrics=="diagnostic";
    const auto g0=Clock::now();
    hnswlib::HierarchicalNSW<float> index(diagnostic?static_cast<hnswlib::SpaceInterface<float>*>(&counted):&l2,o.index);
    index.setEf(o.ef);
    if(index.getCurrentElementCount()<o.k)throw std::invalid_argument("k exceeds index size");
    const uint64_t graph_load=elapsed(g0),graph_rss=peakRss();
    hnswlib::edge_estimation::EdgeCatalog catalog;
    std::unique_ptr<Kernel> kernel;
    std::unique_ptr<LiveArtifactPruner<Kernel,Collect>> pruner;
    const auto a0=Clock::now();
    const uint64_t hash_before=artifactHashNanoseconds();
    if(!o.no_prune) {
        catalog=hnswlib::edge_estimation::EdgeCatalog::fromLayer0Graph(index.getV0Layer0GraphView());
        Header h{};h.dimension=uint32_t(o.dimension);h.identity=catalog.identityDigest();
        kernel.reset(new Kernel(o.artifact,queries,h));kernel->verifyIndex(o.index);
        pruner.reset(new LiveArtifactPruner<Kernel,Collect>(*kernel,catalog,o.beta));
    }
    const uint64_t backend_load=o.no_prune?0:elapsed(a0),backend_rss=peakRss();
    const uint64_t backend_hash=artifactHashNanoseconds()-hash_before;
    const size_t executions=product(o.repeats,ids.size()),neighbors=product(executions,o.k);
    const size_t result_bytes=product(neighbors,sizeof(Neighbor));
    if(result_bytes>o.max_result_bytes)throw std::length_error("result buffer exceeds max-result-bytes");
    std::vector<Neighbor> saved(neighbors);
    std::vector<Sample> samples(diagnostic?executions:0);
    std::vector<uint64_t> latency(o.latency.empty()?0:executions);
    std::vector<Repeat> times(o.repeats);uint64_t scratch_peak=0;
    auto pass=[&](const std::vector<uint64_t>& selection,size_t repeat,bool measure) {
        if(selection.empty())return;
        const size_t window=o.window?std::min(o.window,selection.size()):selection.size();
        const auto wall=Clock::now();
        for(size_t first=0;first<selection.size();first+=window) {
            const size_t last=std::min(selection.size(),first+window);
            const std::vector<uint64_t> block(selection.begin()+first,selection.begin()+last);
            if(kernel) {
                const auto t=Clock::now();
                kernel->prepareQueryBatch(block,o.chunk?std::min(o.chunk,block.size()):block.size());
                if(measure)times[repeat].prepare+=elapsed(t);
                scratch_peak=std::max(scratch_peak,kernel->scratchBytes());
            }
            const auto t=Clock::now();
            for(size_t i=first;i<last;++i) {
                const auto per_query=latency.empty()?Clock::time_point{}:Clock::now();
                if(diagnostic)counted.reset();
                if(pruner)pruner->beginQuery(selection[i]);
                hnswlib::edge_estimation::ScopedActiveEdgePruner active(pruner.get());
                auto result=index.searchKnn(queries->query(selection[i]),o.k);
                if(result.size()!=o.k)throw std::runtime_error("incomplete top-k");
                if(measure) {
                    const size_t sample=repeat*ids.size()+i;
                    for(size_t rank=o.k;rank>0;--rank) {
                        auto n=result.top();result.pop();saved[sample*o.k+rank-1]={n.second,n.first};
                    }
                    if(diagnostic)samples[sample]={repeat,selection[i],counted.calls()};
                    if(!latency.empty())latency[sample]=elapsed(per_query);
                }
            }
            if(measure)times[repeat].search+=elapsed(t);
            if(kernel)kernel->clearPreparedQueryBatch();
        }
        if(measure)times[repeat].service=elapsed(wall);
    };
    pass(warm,0,false);
    const uint64_t warmup_rss=peakRss();
    if(pruner)pruner->resetMetrics();
    for(size_t r=0;r<o.repeats;++r)pass(ids,r,true);
    const auto m=pruner?pruner->metrics():LiveOpqMetrics{};
    if(m.catalog_mismatch || m.backend_exception)throw std::runtime_error("unexpected backend/catalog fallback");
    uint64_t wall=0,prepare=0,search=0,exact=0;
    for(auto t:times){wall+=t.service;prepare+=t.prepare;search+=t.search;}
    for(auto s:samples)exact+=s.exact;
    if(!wall || prepare+search>wall)throw std::runtime_error("invalid timing components");
    std::vector<uint64_t> hashes;
    for(size_t r=0;r<o.repeats;++r)hashes.push_back(checksum(saved,r*ids.size()*o.k,ids.size()*o.k));
    const bool stable=std::all_of(hashes.begin(),hashes.end(),[&](uint64_t h){return h==hashes.front();});
    if(!o.results.empty()) {
        std::ofstream out(o.results);out<<"repeat_id,query_id,rank,label,distance\n"<<std::setprecision(17);
        for(size_t r=0;r<o.repeats;++r)for(size_t i=0;i<ids.size();++i)for(size_t rank=0;rank<o.k;++rank){
            const auto n=saved[(r*ids.size()+i)*o.k+rank];
            out<<r<<','<<ids[i]<<','<<rank<<','<<n.label<<','<<n.distance<<'\n';
        }
        if(!out)throw std::runtime_error("result write failed");
    }
    if(diagnostic) {
        std::ofstream out(o.output+".diagnostic.csv");out<<"repeat_id,query_id,exact_l2_calls\n";
        for(auto s:samples)out<<s.repeat<<','<<s.query<<','<<s.exact<<'\n';
        if(!out)throw std::runtime_error("diagnostic write failed");
    }
    if(!o.latency.empty()) {
        std::ofstream out(o.latency);out<<"repeat_id,query_id,search_materialize_ns\n";
        for(size_t i=0;i<latency.size();++i)out<<i/ids.size()<<','<<ids[i%ids.size()]<<','<<latency[i]<<'\n';
        if(!out)throw std::runtime_error("latency write failed");
    }
    std::ofstream out(o.output);Json j(out);
    j.number("schema_version",2);j.text("method",kernel?kernel->method():"hnsw");
    j.text("kernel","common_window_service_v2");j.text("latency_semantics","offline_full_service_wall");
    j.number("dimension",o.dimension);j.number("k",o.k);j.number("ef_search",o.ef);
    if(o.no_prune)j.nullable("beta",0,false);else j.number("beta",o.beta);
    j.number("query_count",ids.size());j.number("n_unique_queries",ids.size());
    j.number("repeats",o.repeats);j.number("inner_repeats",o.repeats);
    j.number("measured_queries",executions);j.number("n_query_executions",executions);
    j.number("warmup_queries",warm.size());j.boolean("explicit_warmup",!o.warmup_ids.empty());
    if(o.window)j.number("prepare_window",o.window);else j.text("prepare_window","all");
    if(o.chunk)j.number("compute_chunk",o.chunk);else j.text("compute_chunk","all");
    j.number("effective_prepare_window",o.window?std::min(o.window,ids.size()):ids.size());
    j.number("batch_size",o.chunk);j.boolean("legacy_batch_alias",o.batch_alias);
    j.text("batch_engine",kernel?kernel->batchPreparationEngine():"none");
    j.text("metrics_level",o.metrics);j.boolean("timing_instrumented",!latency.empty());
    j.number("service_wall_ns",wall);j.number("total_latency_ns",wall);
    j.number("prepare_wall_ns",prepare);j.number("batch_prepare_total_ns",prepare);
    j.number("batch_prepare_ns_per_query",double(prepare)/executions);
    j.number("search_and_materialize_wall_ns",search);j.number("other_wall_ns",wall-prepare-search);
    j.number("qps_service",executions*1e9/wall);j.number("qps",executions*1e9/wall);
    j.number("query_load_ns",query_load);j.number("graph_load_ns",graph_load);
    j.number("backend_load_and_validation_ns",backend_load);j.number("query_bytes",queries->bytes());
    j.number("backend_file_hash_ns",backend_hash);
    j.number("backend_load_and_structural_validation_ns",backend_load-backend_hash);
    j.text("load_page_cache_state","unspecified");
    j.number("result_buffer_bytes",result_bytes);j.number("backend_bytes",kernel?kernel->backendBytes():0);
    j.number("scratch_bytes",scratch_peak);
    j.number("catalog_payload_bytes",kernel?8ULL*(catalog.nodeCount()+1)+4ULL*catalog.edgeCount():0);
    j.nullable("peak_rss_after_graph_bytes",graph_rss,graph_rss!=0);
    j.nullable("peak_rss_after_backend_bytes",backend_rss,backend_rss!=0);
    j.nullable("peak_rss_after_warmup_bytes",warmup_rss,warmup_rss!=0);
    const auto rss=peakRss();j.nullable("peak_rss_bytes",rss,rss!=0);
    j.nullable("exact_l2_calls",exact,diagnostic);
    const bool present=Collect && !o.no_prune;
    j.nullable("attempted_estimates",m.attempted,present);j.nullable("valid_estimates",m.valid,present);
    j.nullable("pruned_estimates",m.pruned,present);j.nullable("fallback_estimates",m.fallback,present);
    j.nullable("threshold_unavailable",m.threshold_unavailable,present);
    j.nullable("nonfinite_score",m.nonfinite_score,present);j.nullable("catalog_mismatch",m.catalog_mismatch,present);
    j.nullable("backend_exception",m.backend_exception,present);
    j.nullable("zero_length",m.zero_length,present);
    j.nullable("hop_count",0,false); // No hook currently exposes actual graph hops.
    j.nullable("eligible_exact_distance_count",m.attempted-m.pruned,present);
    j.boolean("results_stable_across_repeats",stable);
    j.text("result_checksum",std::to_string(hashes.front()));
    j.boolean("timed_results_written",!o.results.empty());j.boolean("result_records_written",!o.results.empty());
    j.boolean("latency_records_written",!o.latency.empty());
    j.key("repeat_metrics");out<<'[';
    for(size_t r=0;r<times.size();++r){
        if(r)out<<',';Json child(out);child.number("repeat_id",r);
        child.number("service_wall_ns",times[r].service);child.number("prepare_wall_ns",times[r].prepare);
        child.number("search_and_materialize_wall_ns",times[r].search);
        child.text("result_checksum",std::to_string(hashes[r]));child.finish();
    }
    out<<']';j.finish();return 0;
}
template<class Kernel> int main(int argc,char** argv) {
    try {const auto o=parse(argc,argv);return o.metrics=="off"?execute<Kernel,false>(o):execute<Kernel,true>(o);}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
}} // namespace uq::performance
