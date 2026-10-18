#include "axiom_utils.hpp"
#include <cstdint>
#include <numeric>

namespace axiom {

/**
 * Sequential Memory Traversal (Unit Stride).
 * Reads every element contiguously: buffer[0], buffer[1], buffer[2], ...
 * Maximizes hardware prefetching and spatial cache line utilization.
 */
float memory_traverse_sequential(const float* buffer, size_t num_elements, int iterations) {
    float sum = 0.0f;
    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < num_elements; ++i) {
            sum += buffer[i];
        }
    }
    return sum;
}

/**
 * Strided Memory Traversal (Non-unit stride, e.g. stride = 32 floats = 128 bytes).
 * Exactly 1 access per 128-byte cache line, defeating spatial prefetch benefit.
 */
float memory_traverse_strided(const float* buffer, size_t num_elements, size_t stride_elements, int iterations) {
    float sum = 0.0f;
    if (stride_elements == 0) stride_elements = 1;

    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t s = 0; s < stride_elements; ++s) {
            for (size_t i = s; i < num_elements; i += stride_elements) {
                sum += buffer[i];
            }
        }
    }
    return sum;
}

} // namespace axiom
