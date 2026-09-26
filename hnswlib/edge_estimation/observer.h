#pragma once

#include <cstddef>
#include <cstdint>

namespace hnswlib {
namespace edge_estimation {

class CaptureObserver {
 public:
    virtual ~CaptureObserver() {}
    virtual void onQueryBegin() = 0;
    virtual void onExactOnly(
        int32_t graph_layer,
        uint32_t target_id,
        double exact_squared_distance) = 0;
    virtual void onSourceBegin(
        uint64_t expansion_id,
        uint32_t source_id,
        uint32_t source_degree,
        double d_current) = 0;
    virtual void onCandidateBefore(
        uint64_t expansion_id,
        uint64_t edge_id,
        uint32_t source_id,
        uint32_t target_id,
        uint32_t neighbor_slot,
        uint32_t source_degree,
        double d_current,
        double threshold_before,
        bool threshold_valid,
        bool score_slot_eligible) = 0;
    virtual void onCandidateExact(
        uint64_t edge_id,
        double exact_squared_distance) = 0;
    virtual void onQueryEnd() = 0;
};

inline CaptureObserver*& activeCaptureObserverSlot() {
    static thread_local CaptureObserver* observer = NULL;
    return observer;
}

inline CaptureObserver* activeCaptureObserver() {
    return activeCaptureObserverSlot();
}

class ScopedCaptureObserver {
 public:
    explicit ScopedCaptureObserver(CaptureObserver* observer)
        : previous_(activeCaptureObserverSlot()) {
        activeCaptureObserverSlot() = observer;
    }
    ~ScopedCaptureObserver() { activeCaptureObserverSlot() = previous_; }

 private:
    ScopedCaptureObserver(const ScopedCaptureObserver&);
    ScopedCaptureObserver& operator=(const ScopedCaptureObserver&);
    CaptureObserver* previous_;
};

}  // namespace edge_estimation
}  // namespace hnswlib
