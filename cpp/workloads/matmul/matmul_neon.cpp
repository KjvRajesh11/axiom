#include "axiom_utils.hpp"
#include <cstring>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define AXIOM_HAS_ARM_NEON 1
#else
#define AXIOM_HAS_ARM_NEON 0
#endif

namespace axiom {

/**
 * Explicit ARM NEON Matrix Multiplication (4x4 Register-Blocked Outer-Product Microkernel).
 * Utilizes NEON vector registers:
 * - 4 vector registers for accumulator C (c00_c30, c01_c31, c02_c32, c03_c33)
 * - 1 vector register for row of B
 * - Broadcast scalar values of A into NEON vector lanes
 * 
 * If compiled on non-ARM target, gracefully falls back to portable interchanged loop.
 */
void matmul_neon(const float* A, const float* B, float* C, int N) {
#if AXIOM_HAS_ARM_NEON
    std::memset(C, 0, sizeof(float) * N * N);

    // Process 4 rows and 4 columns at a time (4x4 register blocking)
    const int N_block = (N / 4) * 4;

    for (int i = 0; i < N_block; i += 4) {
        for (int k = 0; k < N; ++k) {
            // Load 4 values from column k of A across 4 rows: A[i+0][k], A[i+1][k], A[i+2][k], A[i+3][k]
            const float32x4_t a0 = vdupq_n_f32(A[(i + 0) * N + k]);
            const float32x4_t a1 = vdupq_n_f32(A[(i + 1) * N + k]);
            const float32x4_t a2 = vdupq_n_f32(A[(i + 2) * N + k]);
            const float32x4_t a3 = vdupq_n_f32(A[(i + 3) * N + k]);

            for (int j = 0; j < N_block; j += 4) {
                // Load 4 consecutive elements from row k of B: B[k][j ... j+3]
                const float32x4_t b_vec = vld1q_f32(&B[k * N + j]);

                // Load existing accumulators for C[(i+0..3)][j..j+3]
                float32x4_t c0 = vld1q_f32(&C[(i + 0) * N + j]);
                float32x4_t c1 = vld1q_f32(&C[(i + 1) * N + j]);
                float32x4_t c2 = vld1q_f32(&C[(i + 2) * N + j]);
                float32x4_t c3 = vld1q_f32(&C[(i + 3) * N + j]);

                // Fused multiply-accumulate: C += A * B
                c0 = vfmaq_f32(c0, a0, b_vec);
                c1 = vfmaq_f32(c1, a1, b_vec);
                c2 = vfmaq_f32(c2, a2, b_vec);
                c3 = vfmaq_f32(c3, a3, b_vec);

                // Store back to C
                vst1q_f32(&C[(i + 0) * N + j], c0);
                vst1q_f32(&C[(i + 1) * N + j], c1);
                vst1q_f32(&C[(i + 2) * N + j], c2);
                vst1q_f32(&C[(i + 3) * N + j], c3);
            }

            // Cleanup remaining columns if N is not divisible by 4
            for (int j = N_block; j < N; ++j) {
                const float b_val = B[k * N + j];
                C[(i + 0) * N + j] += A[(i + 0) * N + k] * b_val;
                C[(i + 1) * N + j] += A[(i + 1) * N + k] * b_val;
                C[(i + 2) * N + j] += A[(i + 2) * N + k] * b_val;
                C[(i + 3) * N + j] += A[(i + 3) * N + k] * b_val;
            }
        }
    }

    // Cleanup remaining rows if N is not divisible by 4
    for (int i = N_block; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            const float a_ik = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
#else
    // Portable fallback for non-ARM targets
    matmul_interchanged(A, B, C, N);
#endif
}

} // namespace axiom
