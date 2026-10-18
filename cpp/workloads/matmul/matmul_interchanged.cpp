#include "axiom_utils.hpp"
#include <cstring>

namespace axiom {

/**
 * Loop-Interchanged Matrix Multiplication (i-k-j loop order).
 * Innermost loop iterates over 'j':
 * - A[i * N + k] is scalar and invariant across the inner loop (held in register).
 * - B[k * N + j] is accessed with unit stride (contiguous row access).
 * - C[i * N + j] is accessed with unit stride and accumulated.
 * This dramatically improves spatial locality and enables LLVM auto-vectorization
 * using NEON SIMD vector instructions on ARM64.
 */
void matmul_interchanged(const float* A, const float* B, float* C, int N) {
    // Clear output matrix C
    std::memset(C, 0, sizeof(float) * N * N);

    for (int i = 0; i < N; ++i) {
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
}

} // namespace axiom
