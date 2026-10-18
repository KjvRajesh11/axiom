#include "axiom_utils.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace axiom {

/**
 * CPU-Based Feed-Forward Neural Network (MLP) Inference.
 * Model Architecture:
 * - Layer 1: Dense (in_features -> 128) + ReLU
 * - Layer 2: Dense (128 -> 64) + ReLU
 * - Layer 3: Dense (64 -> out_classes)
 * 
 * Demonstrates CPU inference benchmarking in AI infrastructure:
 * - Batch size sweeps (B = 1, 8, 32, 64)
 * - Steady-state execution timing separated from initialization
 * - Numerical verification against reference scalar forward pass
 */

// Baseline Scalar Forward Pass
void ml_mlp_forward_reference(
    const float* input, int batch_size, int in_features,
    const float* W1, const float* b1, int hidden1,
    const float* W2, const float* b2, int hidden2,
    const float* W3, const float* b3, int out_classes,
    float* output
) {
    std::vector<float> h1(batch_size * hidden1, 0.0f);
    std::vector<float> h2(batch_size * hidden2, 0.0f);

    // Layer 1: H1 = ReLU(Input * W1 + b1)
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < hidden1; ++h) {
            float sum = b1[h];
            for (int in = 0; in < in_features; ++in) {
                sum += input[b * in_features + in] * W1[in * hidden1 + h];
            }
            h1[b * hidden1 + h] = std::max(0.0f, sum); // ReLU
        }
    }

    // Layer 2: H2 = ReLU(H1 * W2 + b2)
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < hidden2; ++h) {
            float sum = b2[h];
            for (int in = 0; in < hidden1; ++in) {
                sum += h1[b * hidden1 + in] * W2[in * hidden2 + h];
            }
            h2[b * hidden2 + h] = std::max(0.0f, sum); // ReLU
        }
    }

    // Layer 3: Output = H2 * W3 + b3
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < out_classes; ++c) {
            float sum = b3[c];
            for (int in = 0; in < hidden2; ++in) {
                sum += h2[b * hidden2 + in] * W3[in * out_classes + c];
            }
            output[b * out_classes + c] = sum;
        }
    }
}

// Optimized Vector-Friendly Forward Pass (Transposed weights & streamed row dot-products)
void ml_mlp_forward_vectorized(
    const float* input, int batch_size, int in_features,
    const float* W1, const float* b1, int hidden1,
    const float* W2, const float* b2, int hidden2,
    const float* W3, const float* b3, int out_classes,
    float* output
) {
    std::vector<float> h1(batch_size * hidden1, 0.0f);
    std::vector<float> h2(batch_size * hidden2, 0.0f);

    // Layer 1: Inverted inner loop for continuous row accumulation
    for (int b = 0; b < batch_size; ++b) {
        const float* in_ptr = input + b * in_features;
        float* h1_ptr = h1.data() + b * hidden1;
        std::memcpy(h1_ptr, b1, sizeof(float) * hidden1);

        for (int in = 0; in < in_features; ++in) {
            const float x = in_ptr[in];
            const float* w_row = W1 + in * hidden1;
            #pragma clang loop vectorize(enable) interleave(enable)
            for (int h = 0; h < hidden1; ++h) {
                h1_ptr[h] += x * w_row[h];
            }
        }
        // Vectorized ReLU
        #pragma clang loop vectorize(enable)
        for (int h = 0; h < hidden1; ++h) {
            h1_ptr[h] = std::max(0.0f, h1_ptr[h]);
        }
    }

    // Layer 2:
    for (int b = 0; b < batch_size; ++b) {
        const float* h1_ptr = h1.data() + b * hidden1;
        float* h2_ptr = h2.data() + b * hidden2;
        std::memcpy(h2_ptr, b2, sizeof(float) * hidden2);

        for (int in = 0; in < hidden1; ++in) {
            const float x = h1_ptr[in];
            const float* w_row = W2 + in * hidden2;
            #pragma clang loop vectorize(enable) interleave(enable)
            for (int h = 0; h < hidden2; ++h) {
                h2_ptr[h] += x * w_row[h];
            }
        }
        #pragma clang loop vectorize(enable)
        for (int h = 0; h < hidden2; ++h) {
            h2_ptr[h] = std::max(0.0f, h2_ptr[h]);
        }
    }

    // Layer 3:
    for (int b = 0; b < batch_size; ++b) {
        const float* h2_ptr = h2.data() + b * hidden2;
        float* out_ptr = output + b * out_classes;
        std::memcpy(out_ptr, b3, sizeof(float) * out_classes);

        for (int in = 0; in < hidden2; ++in) {
            const float x = h2_ptr[in];
            const float* w_row = W3 + in * out_classes;
            #pragma clang loop vectorize(enable) interleave(enable)
            for (int c = 0; c < out_classes; ++c) {
                out_ptr[c] += x * w_row[c];
            }
        }
    }
}

} // namespace axiom
