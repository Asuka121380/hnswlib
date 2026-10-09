#pragma once
#include "hnswlib/hnswlib.h"

namespace uq {
// Diagnostic only: preserve the selected L2 implementation and count actual calls.
class DistanceCounterSpace : public hnswlib::SpaceInterface<float> {
 public:
    explicit DistanceCounterSpace(hnswlib::SpaceInterface<float>& space)
        : size_(space.get_data_size()), original_(space.get_dist_func()),
          parameter_(space.get_dist_func_param()) {}
    size_t get_data_size() override { return size_; }
    hnswlib::DISTFUNC<float> get_dist_func() override { return countDistance; }
    void* get_dist_func_param() override { return this; }
    void reset() { calls_ = 0; }
    uint64_t calls() const { return calls_; }
 private:
    static float countDistance(const void* a, const void* b, const void* context) {
        auto* self = const_cast<DistanceCounterSpace*>(static_cast<const DistanceCounterSpace*>(context));
        ++self->calls_;
        return self->original_(a, b, self->parameter_);
    }
    size_t size_;
    hnswlib::DISTFUNC<float> original_;
    void* parameter_;
    uint64_t calls_ = 0;
};
}  // namespace uq
