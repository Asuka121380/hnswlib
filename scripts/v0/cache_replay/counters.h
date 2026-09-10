#pragma once
#include <cstdint>
#include <stdexcept>
#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

// Optional diagnostic run. Counters exclude index loading and LUT construction.
// Do not compare its timings to --counters 0: syscalls perturb query boundaries.
struct ReplayCounters {
    bool enabled;
    int fds[3]={-1,-1,-1};
    uint64_t values[3]={0,0,0};
    explicit ReplayCounters(bool on):enabled(on) {
        if(!on)return;
#ifdef __linux__
        for(int i=0;i<3;++i) {
            perf_event_attr a{};a.size=sizeof(a);a.disabled=1;a.exclude_kernel=1;a.exclude_hv=1;
            a.type=i==2?PERF_TYPE_HW_CACHE:PERF_TYPE_HARDWARE;
            a.config=i==0?PERF_COUNT_HW_CPU_CYCLES:i==1?PERF_COUNT_HW_INSTRUCTIONS:
                (PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ<<8) | (PERF_COUNT_HW_CACHE_RESULT_MISS<<16));
            a.read_format=PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
            fds[i]=int(syscall(SYS_perf_event_open,&a,0,-1,-1,0));
            if(fds[i]<0){closeAll();throw std::runtime_error("perf_event_open failed: check perf permissions/event support, or use --counters 0");}
        }
#else
        throw std::runtime_error("hardware counters require Linux; use --counters 0");
#endif
    }
    void reset(){
#ifdef __linux__
        if(enabled)for(int fd:fds)if(ioctl(fd,PERF_EVENT_IOC_RESET,0)<0)throw std::runtime_error("perf reset failed");
#endif
    }
    void start(){
#ifdef __linux__
        if(enabled)for(int fd:fds)if(ioctl(fd,PERF_EVENT_IOC_ENABLE,0)<0)throw std::runtime_error("perf enable failed");
#endif
    }
    void stop(){
#ifdef __linux__
        if(enabled)for(int fd:fds)if(ioctl(fd,PERF_EVENT_IOC_DISABLE,0)<0)throw std::runtime_error("perf disable failed");
#endif
    }
    void collect(){
#ifdef __linux__
        if(enabled)for(int i=0;i<3;++i){uint64_t v[3];
            if(::read(fds[i],v,sizeof(v))!=sizeof(v)||v[2]==0||v[2]!=v[1])
                throw std::runtime_error("perf counters unavailable or multiplexed; rerun with fewer competing profilers");
            values[i]=v[0];
        }
#endif
    }
    void closeAll(){
#ifdef __linux__
        for(int&fd:fds)if(fd>=0){::close(fd);fd=-1;}
#endif
    }
    ~ReplayCounters(){closeAll();}
};
