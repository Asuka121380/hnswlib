// Synthetic correctness fixture only. Not a performance or recall dataset.
#include <fstream>
#include <random>
#include <vector>
#include "hnswlib/hnswlib.h"
void cacheReplayPQ(const uint8_t*, double) {}
int main(int argc,char**argv){
    if(argc!=2)return 1;std::string dir=argv[1];const uint32_t d=64,m=8,k=256,n=160;
    std::mt19937 rng(1234);std::normal_distribution<float> normal(0,1);
    hnswlib::L2Space space(d);hnswlib::HierarchicalNSW<float> index(&space,n,8,60,1234);
    std::vector<float> x(d);for(size_t i=0;i<n;++i){for(auto&v:x)v=normal(rng);index.addPoint(x.data(),i);}
    index.saveIndex(dir+"/index.bin");
    auto graph=index.getV0Layer0GraphView();hnswlib::V0SidecarWriteSpec spec;
    spec.dimension=d;spec.pq_m=m;spec.pq_nbits=8;spec.pq_ksub=k;spec.pq_dsub=d/m;spec.node_count=n;
    spec.training_metadata_json="{\"purpose\":\"synthetic cache replay correctness fixture\"}";
    spec.codebook_centroids.resize(k*d);for(auto&v:spec.codebook_centroids)v=normal(rng)*0.1f;
    spec.node_offsets.push_back(0);
    for(size_t i=0;i<n;++i){spec.directed_edge_count+=graph.neighbors(i).size;spec.node_offsets.push_back(spec.directed_edge_count);}
    spec.base_index_sha256=index.getV0SerializedIndexFingerprint();spec.adjacency_sha256=graph.adjacencyFingerprint();
    hnswlib::V0SidecarWriter writer(dir+"/sidecar.bin",spec);
    for(size_t i=0;i<spec.directed_edge_count;++i){hnswlib::V0EdgeRecord r;r.code.resize(m);for(auto&c:r.code)c=rng()%k;
        r.edge_length=8;r.direction_error=2;r.anchor_projection=0;r.numeric_padding=1e-6;writer.writeEdgeRecord(r);}
    writer.finalize();std::ofstream out(dir+"/queries.fvecs",std::ios::binary);
    for(size_t i=0;i<5;++i){for(auto&v:x)v=normal(rng);out.write((char*)&d,4);out.write((char*)x.data(),d*4);}
}
