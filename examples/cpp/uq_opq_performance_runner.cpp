#include "../../tools/edge_estimation/performance_loop.h"
int main(int argc, char** argv) {
    return uq::performance::main<uq::RotatedPqArtifactKernel>(argc, argv);
}