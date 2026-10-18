#include "axiom_utils.hpp"
#include <cstring>
#include <vector>
#include <thread>
#include <algorithm>

namespace axiom {

/**
 * Multithreaded Matrix Multiplication using C++20 standard concurrency.
 * Partitions row iteration i across T worker threads.
 * Since each thread writes to a disjoint set of rows C[i * N ... (i+1) * N - 1],
 * execution is lock-free with zero false-sharing between threads.
 */
void matmul_multithreaded_custom(const float* A, const float* B, float* C, int N, int num_threads) {
    std::memset(C, 0, sizeof(float) * N * N);

    if (num_threads <= 1 || N < 16) {
        matmul_interchanged(A, B, C, N);
        return;
    }

    // Cap thread count to hardware concurrency or matrix row count
    const int actual_threads = std::min({num_threads, N, static_cast<int>(std::thread::hardware_concurrency())});
    if (actual_threads <= 1) {
        matmul_interchanged(A, B, C, N);
        return;
    }

    std::vector<std::jthread> workers;
    workers.reserve(actual_threads);

    const int rows_per_thread = (N + actual_threads - 1) / actual_threads;

    for (int t = 0; t < actual_threads; ++t) {
        const int i_start = t * rows_per_thread;
        const int i_end = std::min(i_start + rows_per_thread, N);
        if (i_start >= i_end) break;

        workers.emplace_back([A, B, C, N, i_start, i_end]() {
            for (int i = i_start; i < i_end; ++i) {
                const int row_offset_A = i * N;
                const int row_offset_C = i * N;
                for (int k = 0; k < N; ++k) {
                    const float a_ik = A[row_offset_A + k];
                    const int row_offset_B = k * N;

                    #pragma clang loop vectorize(enable) interleave(enable)
                    for (int j = 0; j < N; ++j) {
                        C[row_offset_C + j] += a_ik * B[row_offset_B + j];
                    }
                }
            }
        });
    }
    // std::jthread automatically joins upon destruction
}

void matmul_multithreaded(const float* A, const float* B, float* C, int N) {
    // Default multithreaded run uses 4 threads (or detected hardware concurrency)
    matmul_multithreaded_custom(A, B, C, N, 4);
}

} // namespace axiom
