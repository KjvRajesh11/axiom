#include "axiom_utils.hpp"
#include <cstring>
#include <algorithm>

namespace axiom {

/**
 * Tiled 2D Convolution.
 * Partitions output columns into blocks of size 64 floats (256 bytes)
 * to maximize L1 cache line residency when processing large image resolutions.
 */
void conv2d_tiled(const float* input, int H, int W, const float* kernel, int K, float* output) {
    const int out_H = H - K + 1;
    const int out_W = W - K + 1;

    std::memset(output, 0, sizeof(float) * out_H * out_W);

    constexpr int TILE_COLS = 64;

    for (int r = 0; r < out_H; ++r) {
        float* out_row = output + r * out_W;

        for (int c_start = 0; c_start < out_W; c_start += TILE_COLS) {
            const int c_end = std::min(c_start + TILE_COLS, out_W);

            for (int kr = 0; kr < K; ++kr) {
                const float* in_row = input + (r + kr) * W;
                const float* k_row = kernel + kr * K;

                for (int kc = 0; kc < K; ++kc) {
                    const float w = k_row[kc];
                    const float* in_ptr = in_row + kc;

                    #pragma clang loop vectorize(enable) interleave(enable)
                    for (int c = c_start; c < c_end; ++c) {
                        out_row[c] += in_ptr[c] * w;
                    }
                }
            }
        }
    }
}

} // namespace axiom
