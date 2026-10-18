#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>

namespace axiom {

struct VerificationResult {
    bool passed;
    double max_absolute_diff;
    double max_relative_diff;
    size_t mismatch_count;
    double tolerance_abs;
    double tolerance_rel;
};

// Deterministic pseudo-random initialization using fixed seed
inline void init_matrix_f32(std::vector<float>& mat, int rows, int cols, uint32_t seed = 42) {
    mat.resize(rows * cols);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < mat.size(); ++i) {
        mat[i] = dist(rng);
    }
}

inline void init_vector_f32(std::vector<float>& vec, size_t size, uint32_t seed = 42) {
    vec.resize(size);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = dist(rng);
    }
}

// Numerical tolerance verification:
// |actual - expected| <= tol_abs + tol_rel * |expected|
inline VerificationResult verify_results_f32(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    double tol_abs = 1e-4,
    double tol_rel = 1e-4
) {
    VerificationResult res{true, 0.0, 0.0, 0, tol_abs, tol_rel};
    if (actual.size() != expected.size()) {
        res.passed = false;
        res.mismatch_count = actual.size() > expected.size() ? actual.size() : expected.size();
        return res;
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        double act = static_cast<double>(actual[i]);
        double exp = static_cast<double>(expected[i]);
        double abs_diff = std::abs(act - exp);
        double denom = std::max(std::abs(exp), 1e-12);
        double rel_diff = abs_diff / denom;

        if (abs_diff > res.max_absolute_diff) {
            res.max_absolute_diff = abs_diff;
        }
        if (rel_diff > res.max_relative_diff) {
            res.max_relative_diff = rel_diff;
        }

        double allowed_diff = tol_abs + tol_rel * std::abs(exp);
        if (abs_diff > allowed_diff) {
            res.passed = false;
            res.mismatch_count++;
        }
    }
    return res;
}

// Workload function prototypes
// 1. Matrix Multiplication (C = A * B, matrices N x N, row-major)
void matmul_reference(const float* A, const float* B, float* C, int N);
void matmul_interchanged(const float* A, const float* B, float* C, int N);
void matmul_blocked(const float* A, const float* B, float* C, int N);
void matmul_neon(const float* A, const float* B, float* C, int N);
void matmul_multithreaded(const float* A, const float* B, float* C, int N);
void matmul_multithreaded_custom(const float* A, const float* B, float* C, int N, int num_threads);
void matmul_regressed(const float* A, const float* B, float* C, int N);
void matmul_avx512_dispatched(const float* A, const float* B, float* C, int N);
bool matmul_avx512_is_supported_on_host();

// 2. 2D Convolution (Output = Input convolved with Kernel K x K)
void conv2d_reference(const float* input, int H, int W, const float* kernel, int K, float* output);
void conv2d_row_major(const float* input, int H, int W, const float* kernel, int K, float* output);
void conv2d_tiled(const float* input, int H, int W, const float* kernel, int K, float* output);

// 3. Memory Locality & Working-Set Traversal
float memory_traverse_sequential(const float* buffer, size_t num_elements, int iterations);
float memory_traverse_strided(const float* buffer, size_t num_elements, size_t stride_elements, int iterations);

// 4. CPU-Based ML Feed-Forward Neural Network Inference
void ml_mlp_forward_reference(
    const float* input, int batch_size, int in_features,
    const float* W1, const float* b1, int hidden1,
    const float* W2, const float* b2, int hidden2,
    const float* W3, const float* b3, int out_classes,
    float* output
);

void ml_mlp_forward_vectorized(
    const float* input, int batch_size, int in_features,
    const float* W1, const float* b1, int hidden1,
    const float* W2, const float* b2, int hidden2,
    const float* W3, const float* b3, int out_classes,
    float* output
);

} // namespace axiom
