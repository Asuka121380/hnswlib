// Fixed-order PQ/exact memory replay. See README.md for interpretation limits.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <array>
#include "counters.h"
#ifdef __linux__
#include <sched.h>
#endif
void cacheReplayPQ(const uint8_t*, double);
#include "hnswlib/hnswlib.h"

namespace cr {
struct Event { uint64_t kind, id; double current; }; // Q=0, PQ=1, exact=2
static_assert(sizeof(Event)==24, "trace event ABI");
std::vector<Event>* recording=nullptr;
const uint8_t* edge0=nullptr;
uint64_t edge_stride=0, edge_count=0;
std::unordered_map<const void*, uint64_t> nodes;
void require(bool yes, const std::string& msg) { if(!yes) throw std::runtime_error(msg); }
struct Space : hnswlib::SpaceInterface<float> {
    hnswlib::L2Space base;
    explicit Space(size_t d):base(d){}
    size_t get_data_size() override {return base.get_data_size();}
    void* get_dist_func_param() override {return this;}
    hnswlib::DISTFUNC<float> get_dist_func() override {return distance;}
    static float distance(const void* a,const void* b,const void* p) {
        Space* self=(Space*)p;
        if(recording) {
            auto it=nodes.find(b);
            require(it!=nodes.end(),"exact operand not an index vector");
            recording->push_back({2,it->second,0});
        }
        return self->base.get_dist_func()(a,b,self->base.get_dist_func_param());
    }
};
using Args=std::map<std::string,std::string>;
std::string get(const Args&a,const std::string&k,const std::string&d="") {
    auto i=a.find(k);return i==a.end()?d:i->second;
}
uint64_t number(const Args&a,const std::string&k,uint64_t d) {
    std::string s=get(a,k,std::to_string(d));
    require(!s.empty()&&s[0]!='-',"invalid unsigned option "+k);
    size_t pos=0;auto v=std::stoull(s,&pos);require(pos==s.size(),"invalid number "+k);return v;
}
std::vector<float> queries(const std::string& path,size_t dim,size_t start,size_t count) {
    std::ifstream in(path,std::ios::binary);require(bool(in),"cannot open queries");
    in.seekg(uint64_t(start)*(dim+1)*4);
    std::vector<float> q(dim*count);
    for(size_t i=0;i<count;++i) {
        uint32_t d=0;in.read((char*)&d,4);require(d==dim,"fvecs dimension/truncation");
        in.read((char*)&q[i*dim],dim*4);require(bool(in),"truncated fvecs");
        for(size_t j=0;j<dim;++j) require(std::isfinite(q[i*dim+j]),"nonfinite query");
    }return q;
}
template<class T> void write(std::ostream&o,const T&v){o.write((const char*)&v,sizeof(v));}
template<class T> void read(std::istream&i,T&v){i.read((char*)&v,sizeof(v));require(bool(i),"truncated trace");}
void save(const std::string&p,const std::vector<Event>&e,size_t d,size_t count) {
    std::ofstream o(p,std::ios::binary);require(bool(o),"cannot write trace");
    o.write("PQREPL01",8);write(o,uint64_t(d));write(o,uint64_t(count));write(o,uint64_t(e.size()));
    o.write((const char*)e.data(),e.size()*sizeof(Event));require(bool(o),"trace write failed");
}
std::vector<Event> load(const std::string&p,size_t d,size_t count) {
    std::ifstream in(p,std::ios::binary|std::ios::ate);require(bool(in),"cannot open trace");
    auto bytes=in.tellg();in.seekg(0);char magic[8];in.read(magic,8);
    require(bool(in)&&std::memcmp(magic,"PQREPL01",8)==0,"trace magic mismatch");
    uint64_t dd,cc,n;read(in,dd);read(in,cc);read(in,n);
    require(dd==d&&cc==count,"trace query contract mismatch");
    require(bytes>=32 && n==uint64_t(bytes-std::streamoff(32))/24 && (bytes-std::streamoff(32))%24==0,"trace size mismatch");
    std::vector<Event> e(n);in.read((char*)e.data(),n*24);require(bool(in),"trace read failed");return e;
}
struct Lut {
    std::vector<float> storage;float* p;
    size_t m,k,spacing;
    Lut(size_t mm,size_t kk,size_t ss):storage(mm*kk*ss+16),m(mm),k(kk),spacing(ss) {
        p=(float*)((reinterpret_cast<uintptr_t>(storage.data())+63)&~uintptr_t(63));
    }
    void build(const float*q,const float*book,size_t original_k,size_t ds) {
        for(size_t j=0;j<m;++j)for(size_t c=0;c<k;++c) {
            float sum=0;for(size_t x=0;x<ds;++x)sum+=q[j*ds+x]*book[(j*original_k+c)*ds+x];
            p[(j*k+c)*spacing]=sum;
        }
    }
};
volatile double sink=0;
}
void cacheReplayPQ(const uint8_t*code,double current) {
    if(!cr::recording)return;
    auto delta=uintptr_t(code)-uintptr_t(cr::edge0);
    cr::require(cr::edge_stride && delta%cr::edge_stride==0 && delta/cr::edge_stride<cr::edge_count,"PQ edge address mismatch");
    cr::recording->push_back({1,delta/cr::edge_stride,current});
}
int main(int argc,char**argv) {try {
    using namespace cr;
    Args a;
    const std::vector<std::string> allowed={"mode","index","sidecar","queries","dimension","query-start","query-count","trace","output","ef","k","beta","retry","prefetch","repeats","warmups","spacing","fold-bits","cpu","counters"};
    for(int i=1;i<argc;++i) {
        std::string key=argv[i];if(key=="--help") {
            std::cout<<"Use run_replay.py build|record|replay|smoke. See README.md.\n";return 0;
        }
        require(key.substr(0,2)=="--" && i+1<argc,"expected --key value");key=key.substr(2);
        require(std::find(allowed.begin(),allowed.end(),key)!=allowed.end(),"unknown option "+key);
        require(a.emplace(key,argv[++i]).second,"duplicate option "+key);
    }
    auto mode=get(a,"mode");require(mode=="record"||mode=="replay","mode must be record or replay");
    const size_t d=number(a,"dimension",960),count=number(a,"query-count",100),start=number(a,"query-start",0);
    require(d>0&&count>0,"dimension/count must be positive");
    int cpu=std::stoi(get(a,"cpu","-1"));
#ifdef __linux__
    if(cpu>=0){require(cpu<CPU_SETSIZE,"cpu out of range");cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(cpu,&mask);require(sched_setaffinity(0,sizeof(mask),&mask)==0,"CPU affinity failed");}
#else
    require(cpu==-1,"--cpu currently requires Linux");
#endif
    auto q=queries(get(a,"queries"),d,start,count);
    Space space(d);hnswlib::HierarchicalNSW<float> index(&space,get(a,"index"));
    index.loadEdgeQuantV0Metadata(get(a,"sidecar"));
    const auto& metadata=index.getEdgeQuantV0Metadata();const auto& h=metadata.header();auto view=metadata.view();
    require(h.dimension==d && h.directed_edge_count>0,"sidecar dimension/edges invalid");
    edge0=view.edgeRecord(0).codeDataUnchecked();edge_stride=h.edge_record_stride;edge_count=h.directed_edge_count;
    std::vector<Event> events;
    if(mode=="record") {
        for(size_t i=0;i<index.getCurrentElementCount();++i)nodes[index.getDataByInternalId(i)]=i;
        hnswlib::V0ApproxPruningConfig config;config.beta=std::stod(get(a,"beta","1.4"));
        require(config.valid(),"invalid beta");auto retry=number(a,"retry",1);require(retry<=1,"retry must be 0 or 1");config.retry_enabled=retry;
        auto prefetch=get(a,"prefetch","legacy");require(prefetch=="legacy"||prefetch=="gate","invalid prefetch");
        config.prefetch_policy=prefetch=="gate"?hnswlib::V0ApproxPrefetchPolicy::GateAware:hnswlib::V0ApproxPrefetchPolicy::LegacyVector;
        auto ef=number(a,"ef",500),k=number(a,"k",10);require(ef>=k&&k>0,"require ef>=k>0");index.setEf(ef);
        // Record each query once; recording timing is never used for performance.
        for(size_t i=0;i<count;++i){events.push_back({0,i,0});recording=&events;auto result=index.searchKnnV0ApproxFast(&q[i*d],k,config);recording=nullptr;
            auto reference=index.searchKnnV0ApproxFast(&q[i*d],k,config);
            require(result.size()==reference.size(),"recording changed result count");
            while(!result.empty()){require(result.top()==reference.top(),"recording changed search results");result.pop();reference.pop();}}
        save(get(a,"trace"),events,d,count);
    } else events=load(get(a,"trace"),d,count);
    size_t nq=0,npq=0,nexact=0;bool active=false;
    std::vector<size_t> boundaries;
    for(size_t ei=0;ei<events.size();++ei){const auto&e=events[ei];
        if(e.kind==0){require(e.id==nq&&e.id<count,"query ordering invalid");boundaries.push_back(ei);++nq;active=true;}
        else if(e.kind==1){require(active&&e.id<edge_count&&std::isfinite(e.current)&&e.current>=0,"invalid PQ event");++npq;}
        else if(e.kind==2){require(active&&e.id<index.getCurrentElementCount(),"invalid exact event");++nexact;}
        else require(false,"unknown event kind");
    }
    require(nq==count&&npq>0&&nexact>0,"trace must contain queries, PQ and exact events");
    boundaries.push_back(events.size());
    size_t spacing=number(a,"spacing",1),bits=number(a,"fold-bits",h.pq_nbits);
    require(spacing==1||spacing==4||spacing==16,"spacing must be 1, 4 or 16");
    require(bits>0&&bits<=h.pq_nbits&&bits<=8 && h.pq_ksub==(1u<<h.pq_nbits),"invalid bit contract");
    const size_t kk=1u<<bits;
    // Folded codes are explicitly a cache sensitivity surrogate, NOT trained low-bit PQ.
    Lut lut(h.pq_m,kk,spacing);const float*book=metadata.nativeCodebookData();
    size_t repeats=number(a,"repeats",7),warmups=number(a,"warmups",2);require(repeats>0,"repeats must be positive");
    std::vector<double> times,checksums;
    std::vector<std::array<uint64_t,3>> hardware;
    ReplayCounters counters(number(a,"counters",0)!=0);
    if(mode=="replay")for(size_t r=0;r<warmups+repeats;++r) {
        counters.reset();
        double sum=0,elapsed=0;
        for(size_t qi=0;qi<count;++qi) {
            size_t begin=boundaries[qi],end=boundaries[qi+1];
            const float*query=&q[events[begin].id*d];
            lut.build(query,book,h.pq_ksub,h.pq_dsub); // same query lifecycle; outside timed region
            counters.start();
            auto t0=std::chrono::steady_clock::now();
            for(size_t j=begin+1;j<end;++j) {
                const Event&e=events[j];
                if(e.kind==1){auto edge=view.edgeRecord(e.id);const uint8_t*code=edge.codeDataUnchecked();float dot=0;
                    for(size_t m=0;m<h.pq_m;++m)dot+=lut.p[(m*kk+(code[m]&(kk-1)))*spacing];
                    double len=edge.edgeLengthNativeUnchecked(),anchor=edge.anchorProjectionNativeUnchecked();
                    sum+=e.current+len*len-2*len*(double(dot)-anchor);
                }else sum+=space.base.get_dist_func()(query,index.getDataByInternalId(e.id),space.base.get_dist_func_param());
            }
            auto t1=std::chrono::steady_clock::now();counters.stop();
            elapsed+=std::chrono::duration<double,std::nano>(t1-t0).count();
        }
        sink=sum;require(std::isfinite(sum),"nonfinite checksum");
        counters.collect();
        if(r>=warmups){times.push_back(elapsed);checksums.push_back(sum);hardware.push_back({counters.values[0],counters.values[1],counters.values[2]});}
    }
    std::ofstream out(get(a,"output"));require(bool(out),"cannot write result");
    out<<std::setprecision(17)<<"{\"schema\":1,\"mode\":\""<<mode<<"\",\"queries\":"<<nq<<",\"pq_events\":"<<npq<<",\"exact_events\":"<<nexact
       <<",\"m\":"<<h.pq_m<<",\"source_bits\":"<<h.pq_nbits<<",\"lookup_bits\":"<<bits<<",\"folded_surrogate\":"<<(bits<h.pq_nbits?"true":"false")
       <<",\"spacing\":"<<spacing<<",\"lut_bytes\":"<<h.pq_m*kk*spacing*4<<",\"lut_build_timed\":false,\"elapsed_ns\":[";
    for(size_t i=0;i<times.size();++i){if(i)out<<',';out<<times[i];}out<<"],\"checksums\":[";
    for(size_t i=0;i<checksums.size();++i){if(i)out<<',';out<<checksums[i];}
    out<<"],\"counter_names\":[\"cycles\",\"instructions\",\"L1D_read_misses\"],\"counters\":";
    if(!counters.enabled)out<<"null";else{out<<'[';for(size_t i=0;i<hardware.size();++i){if(i)out<<',';out<<'['<<hardware[i][0]<<','<<hardware[i][1]<<','<<hardware[i][2]<<']';}out<<']';}
    out<<"}\n";require(bool(out),"result write failed");
    return 0;
}catch(const std::exception&e){cr::recording=nullptr;std::cerr<<e.what()<<'\n';return 1;}}
