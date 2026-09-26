#pragma once

#include "optional_backend.h"

namespace uq {
inline UnavailableBackend saqBackend() {
    return UnavailableBackend("saq", "optional SAQ dependency/required ISA is unavailable");
}
}  // namespace uq
