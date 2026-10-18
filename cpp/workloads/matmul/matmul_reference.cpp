#include "axiom_utils.hpp"
#include <cstring>

namespace axiom {

/**
 * Baseline Naive Matrix Multiplication (i-j-k loop order).
 * Row-major layout: A[i][k] is contiguous, but B[k][j] has stride N.
 * As N increases, accessing B column-wise causes frequent cache line misses
 * and inhibits auto-vectorization.
 */
void matmul_reference(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

} // namespace axiom
