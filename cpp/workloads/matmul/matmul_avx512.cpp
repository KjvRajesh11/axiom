#include "axiom_cpu_features.hpp"
#include "axiom_utils.hpp"
#include <cstring>
#include <iostream>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace axiom {

// Forward declaration of portable fallback
void matmul_interchanged(const float* A, const float* B, float* C, int N);

#if defined(__AVX512F__)
/**
 * Native AVX-512 Matrix Multiplication Kernel.
 * Uses 512-bit ZMM registers processing 16 single-precision floats per vector.
 * - Broadcasts A[i * N + k] into a 512-bit vector with _mm512_set1_ps.
 * - Loads contiguous slices of B[k * N + j] with _mm512_loadu_ps.
 * - Accumulates into C[i * N + j] using fused multiply-add (_mm512_fmadd_ps).
 * - Handles tails (N % 16 != 0) scalar-wise for complete bounds safety.
 */
void matmul_avx512_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    std::memset(C, 0, sizeof(float) * N * N);

    for (int i = 0; i < N; ++i) {
        const int row_offset_A = i * N;
        const int row_offset_C = i * N;

        for (int k = 0; k < N; ++k) {
            const float a_val = A[row_offset_A + k];
            const __m512 v_a = _mm512_set1_ps(a_val);
            const int row_offset_B = k * N;

            int j = 0;
            // Unroll by 32 floats (two 512-bit vectors)
            for (; j + 31 < N; j += 32) {
                __m512 vb0 = _mm512_loadu_ps(&B[row_offset_B + j]);
                __m512 vb1 = _mm512_loadu_ps(&B[row_offset_B + j + 16]);
                __m512 vc0 = _mm512_loadu_ps(&C[row_offset_C + j]);
                __m512 vc1 = _mm512_loadu_ps(&C[row_offset_C + j + 16]);

                vc0 = _mm512_fmadd_ps(v_a, vb0, vc0);
                vc1 = _mm512_fmadd_ps(v_a, vb1, vc1);

                _mm512_storeu_ps(&C[row_offset_C + j], vc0);
                _mm512_storeu_ps(&C[row_offset_C + j + 16], vc1);
            }

            // Single 16-float vector step
            for (; j + 15 < N; j += 16) {
                __m512 vb = _mm512_loadu_ps(&B[row_offset_B + j]);
                __m512 vc = _mm512_loadu_ps(&C[row_offset_C + j]);
                vc = _mm512_fmadd_ps(v_a, vb, vc);
                _mm512_storeu_ps(&C[row_offset_C + j], vc);
            }

            // Tail handling for non-multiples of 16
            for (; j < N; ++j) {
                C[row_offset_C + j] += a_val * B[row_offset_B + j];
            }
        }
    }
}
#endif

/**
 * Safe Dispatch Entry Point.
 * Inspects CPUID features at runtime:
 * - If AVX-512 is supported by the CPU and enabled in build: dispatches to native kernel.
 * - Otherwise: safely routes to high-efficiency portable fallback without executing illegal instructions.
 */
void matmul_avx512_dispatched(const float* A, const float* B, float* C, int N) {
#if defined(__AVX512F__)
    static const bool cpu_has_avx512 = detect_cpu_features().has_avx512f;
    if (cpu_has_avx512) {
        matmul_avx512_kernel(A, B, C, N);
        return;
    }
#endif
    // Safe portable fallback
    matmul_interchanged(A, B, C, N);
}

bool matmul_avx512_is_supported_on_host() {
#if defined(__AVX512F__)
    return detect_cpu_features().has_avx512f;
#else
    return false;
#endif
}

} // namespace axiom
