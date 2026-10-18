#include "axiom_utils.hpp"
#include <cstring>

namespace axiom {

/**
 * Controlled Performance Regression Candidate.
 * Intentionally iterates with a pathological column-strided traversal pattern
 * combined with unnecessary intermediate writes, demonstrating that Axiom's
 * Decision Engine reliably detects and rejects statistically significant performance regressions.
 */
void matmul_regressed(const float* A, const float* B, float* C, int N) {
    std::memset(C, 0, sizeof(float) * N * N);

    // j-k-i traversal order: worst possible spatial locality for row-major matrices
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < N; ++k) {
            const float b_kj = B[k * N + j];
            for (int i = 0; i < N; ++i) {
                // Column strided read on A (stride N) and column strided write on C (stride N)
                C[i * N + j] += A[i * N + k] * b_kj;
            }
        }
    }
}

} // namespace axiom
