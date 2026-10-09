#include "../../tools/edge_estimation/backends/ivf_edge.h"
#include "../../tools/edge_estimation/performance_loop.h"
int main(int argc, char** argv) {
    return uq::performance::main<uq::IvfEdgeArtifactKernel>(argc, argv);
}