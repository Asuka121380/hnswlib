#pragma once

#include "edge_quant_v0_graph_access.h"
#include "edge_quant_v0_sampler.h"

namespace hnswlib {

// Stage 1 feature-isolation scaffold. Query-time V0 state is introduced in
// later implementation stages without changing the baseline search API.
struct EdgeQuantV0QueryContext {};

}  // namespace hnswlib
