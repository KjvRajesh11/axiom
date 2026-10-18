#include "axiom_utils.hpp"

namespace axiom {

/**
 * Baseline Naive 2D Image Convolution.
 * Computes output[r, c] = sum_{kr, kc} input[r + kr, c + kc] * kernel[kr, kc]
 * Valid padding (output size is (H - K + 1) x (W - K + 1)).
 */
void conv2d_reference(const float* input, int H, int W, const float* kernel, int K, float* output) {
    const int out_H = H - K + 1;
    const int out_W = W - K + 1;

    for (int r = 0; r < out_H; ++r) {
        for (int c = 0; c < out_W; ++c) {
            float sum = 0.0f;
            for (int kr = 0; kr < K; ++kr) {
                for (int kc = 0; kc < K; ++kc) {
                    sum += input[(r + kr) * W + (c + kc)] * kernel[kr * K + kc];
                }
            }
            output[r * out_W + c] = sum;
        }
    }
}

} // namespace axiom
