#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>
#include <functional>

namespace axiom {

// Prevent dead-code elimination by forcing compiler to treat the value as read/modified
template <typename T>
inline void doNotOptimizeAway(T const& value) {
#if defined(__clang__) || defined(__GNUC__)
    asm volatile("" : : "r,m"(value) : "memory");
#else
    volatile const T* p = &value;
    (void)p;
#endif
}

template <typename T>
inline void clobberMemory() {
#if defined(__clang__) || defined(__GNUC__)
    asm volatile("" : : : "memory");
#endif
}

// High-resolution clock wrapper
struct BenchmarkClock {
    using Clock = std::chrono::steady_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using DurationNs = std::chrono::duration<double, std::nano>;

    static TimePoint now() {
        return Clock::now();
    }

    static double durationMs(TimePoint start, TimePoint end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    static double durationNs(TimePoint start, TimePoint end) {
        return DurationNs(end - start).count();
    }
};

struct ExecutionStats {
    std::string workload_name;
    std::string candidate_id;
    int size;
    int warmup_runs;
    int measurement_runs;
    std::vector<double> run_times_ms;
    double min_ms;
    double max_ms;
    double median_ms;
    double mean_ms;
    double stddev_ms;
    double iqr_ms;
    double gflops;
    bool passed_correctness;
    double max_rel_error;
};

} // namespace axiom
