#include "axiom_utils.hpp"
#include <cstring>

namespace axiom {

/**
 * Optimized 2D Convolution with Loop Interchange (Kernel over Image Columns).
 * Inverting loop order allows kernel weights to be held in scalar registers,
 * streaming over output columns with contiguous memory writes and enabling SIMD.
 */
void conv2d_row_major(const float* input, int H, int W, const float* kernel, int K, float* output) {
    const int out_H = H - K + 1;
    const int out_W = W - K + 1;

    std::memset(output, 0, sizeof(float) * out_H * out_W);

    for (int r = 0; r < out_H; ++r) {
        float* out_row = output + r * out_W;
        for (int kr = 0; kr < K; ++kr) {
            const float* in_row = input + (r + kr) * W;
            const float* k_row = kernel + kr * K;

            for (int kc = 0; kc < K; ++kc) {
                const float w = k_row[kc];
                const float* in_ptr = in_row + kc;

                #pragma clang loop vectorize(enable) interleave(enable)
                for (int c = 0; c < out_W; ++c) {
                    out_row[c] += in_ptr[c] * w;
                }
            }
        }
    }
}

} // namespace axiom
