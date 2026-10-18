#include "axiom_utils.hpp"
#include <cstring>
#include <algorithm>

namespace axiom {

/**
 * Cache-Blocked Matrix Multiplication (Tiled i-k-j loop order).
 * Partitions N x N matrices into B x B sub-blocks where B = 64.
 * 64 x 64 floats = 16 KB per block. Three active sub-blocks consume ~48 KB,
 * fitting comfortably into the Apple M4 L1 Data Cache (64 KB).
 * This eliminates capacity cache misses on large matrices (N >= 512).
 */
void matmul_blocked(const float* A, const float* B, float* C, int N) {
    std::memset(C, 0, sizeof(float) * N * N);

    constexpr int BLOCK_SIZE = 64;

    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        const int i_max = std::min(ii + BLOCK_SIZE, N);
        for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
            const int k_max = std::min(kk + BLOCK_SIZE, N);
            for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                const int j_max = std::min(jj + BLOCK_SIZE, N);

                // Micro-kernel inside block: i-k-j order
                for (int i = ii; i < i_max; ++i) {
                    const int row_offset_A = i * N;
                    const int row_offset_C = i * N;
                    for (int k = kk; k < k_max; ++k) {
                        const float a_ik = A[row_offset_A + k];
                        const int row_offset_B = k * N;
                        
                        #pragma clang loop vectorize(enable) interleave(enable)
                        for (int j = jj; j < j_max; ++j) {
                            C[row_offset_C + j] += a_ik * B[row_offset_B + j];
                        }
                    }
                }
            }
        }
    }
}

} // namespace axiom
