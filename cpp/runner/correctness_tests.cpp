#include "axiom_benchmark.hpp"
#include "axiom_utils.hpp"
#include <iostream>
#include <vector>
#include <string>

using namespace axiom;

int main() {
    int total_tests = 0;
    int failed_tests = 0;

    std::cout << "=== Running Axiom 2.1 C++ Workload Correctness Tests ===\n";

    // 1. Matrix multiplication tests across sizes
    std::vector<int> matmul_sizes = {8, 16, 32, 64, 128, 256};
    for (int N : matmul_sizes) {
        std::vector<float> A, B, C_ref, C_interchanged, C_blocked, C_neon, C_multithreaded, C_regressed, C_avx512;
        init_matrix_f32(A, N, N, 123 + N);
        init_matrix_f32(B, N, N, 456 + N);
        C_ref.resize(N * N, 0.0f);
        C_interchanged.resize(N * N, 0.0f);
        C_blocked.resize(N * N, 0.0f);
        C_neon.resize(N * N, 0.0f);
        C_multithreaded.resize(N * N, 0.0f);
        C_regressed.resize(N * N, 0.0f);
        C_avx512.resize(N * N, 0.0f);

        matmul_reference(A.data(), B.data(), C_ref.data(), N);
        matmul_interchanged(A.data(), B.data(), C_interchanged.data(), N);
        matmul_blocked(A.data(), B.data(), C_blocked.data(), N);
        matmul_neon(A.data(), B.data(), C_neon.data(), N);
        matmul_multithreaded(A.data(), B.data(), C_multithreaded.data(), N);
        matmul_regressed(A.data(), B.data(), C_regressed.data(), N);
        matmul_avx512_dispatched(A.data(), B.data(), C_avx512.data(), N);

        // Test interchanged
        total_tests++;
        if (!verify_results_f32(C_interchanged, C_ref, 1e-4, 1e-4).passed) {
            std::cerr << "FAIL: matmul_interchanged N=" << N << "\n";
            failed_tests++;
        } else {
            std::cout << "PASS: matmul_interchanged N=" << N << "\n";
        }

        // Test blocked
        total_tests++;
        if (!verify_results_f32(C_blocked, C_ref, 1e-4, 1e-4).passed) {
            std::cerr << "FAIL: matmul_blocked N=" << N << "\n";
            failed_tests++;
        } else {
            std::cout << "PASS: matmul_blocked N=" << N << "\n";
        }

        // Test explicit ARM NEON
        total_tests++;
        if (!verify_results_f32(C_neon, C_ref, 1e-4, 1e-4).passed) {
            std::cerr << "FAIL: matmul_neon N=" << N << "\n";
            failed_tests++;
        } else {
            std::cout << "PASS: matmul_neon N=" << N << "\n";
        }

        // Test multithreaded
        total_tests++;
        if (!verify_results_f32(C_multithreaded, C_ref, 1e-4, 1e-4).passed) {
            std::cerr << "FAIL: matmul_multithreaded N=" << N << "\n";
            failed_tests++;
        } else {
            std::cout << "PASS: matmul_multithreaded N=" << N << "\n";
        }

        // Test regressed (must be mathematically correct, despite being slow)
        total_tests++;
        if (!verify_results_f32(C_regressed, C_ref, 1e-4, 1e-4).passed) {
            std::cerr << "FAIL: matmul_regressed N=" << N << "\n";
            failed_tests++;
        } else {
            std::cout << "PASS: matmul_regressed N=" << N << "\n";
        }

        // Test AVX-512 safe dispatch (native on x86-64 AVX-512, portable fallback on others)
        total_tests++;
        if (!verify_results_f32(C_avx512, C_ref, 1e-4, 1e-4).passed) {
            std::cerr << "FAIL: matmul_avx512_dispatched N=" << N << "\n";
            failed_tests++;
        } else {
            std::cout << "PASS: matmul_avx512_dispatched N=" << N << "\n";
        }
    }

    // 1b. Non-power-of-two tail bounds tests for AVX-512 / vector loops
    for (int N : {17, 33, 49, 65}) {
        std::vector<float> A, B, C_ref(N * N, 0.0f), C_avx512(N * N, 0.0f);
        init_matrix_f32(A, N, N, 777 + N);
        init_matrix_f32(B, N, N, 888 + N);
        matmul_reference(A.data(), B.data(), C_ref.data(), N);
        matmul_avx512_dispatched(A.data(), B.data(), C_avx512.data(), N);
        total_tests++;
        if (!verify_results_f32(C_avx512, C_ref, 1e-4, 1e-4).passed) {
            std::cerr << "FAIL: matmul_avx512 tail bounds N=" << N << "\n";
            failed_tests++;
        } else {
            std::cout << "PASS: matmul_avx512 tail bounds N=" << N << "\n";
        }
    }

    // 2. 2D Convolution tests across sizes
    std::vector<int> conv_sizes = {32, 64, 128};
    for (int S : conv_sizes) {
        int H = S, W = S, K = 5;
        int out_H = H - K + 1;
        int out_W = W - K + 1;

        std::vector<float> input, kernel, out_ref, out_row_major, out_tiled;
        init_matrix_f32(input, H, W, 789 + S);
        init_matrix_f32(kernel, K, K, 321 + S);
        out_ref.resize(out_H * out_W, 0.0f);
        out_row_major.resize(out_H * out_W, 0.0f);
        out_tiled.resize(out_H * out_W, 0.0f);

        conv2d_reference(input.data(), H, W, kernel.data(), K, out_ref.data());
        conv2d_row_major(input.data(), H, W, kernel.data(), K, out_row_major.data());
        conv2d_tiled(input.data(), H, W, kernel.data(), K, out_tiled.data());

        total_tests++;
        if (!verify_results_f32(out_row_major, out_ref, 1e-4, 1e-4).passed) {
            std::cerr << "FAIL: conv2d_row_major size=" << S << "\n";
            failed_tests++;
        } else {
            std::cout << "PASS: conv2d_row_major size=" << S << "\n";
        }

        total_tests++;
        if (!verify_results_f32(out_tiled, out_ref, 1e-4, 1e-4).passed) {
            std::cerr << "FAIL: conv2d_tiled size=" << S << "\n";
            failed_tests++;
        } else {
            std::cout << "PASS: conv2d_tiled size=" << S << "\n";
        }
    }

    // 3. CPU ML Inference Tests (MLP forward pass) across batch sizes
    std::vector<int> ml_batches = {1, 8, 32};
    for (int B : ml_batches) {
        int in_features = 64, hidden1 = 128, hidden2 = 64, out_classes = 10;
        std::vector<float> input, W1, b1, W2, b2, W3, b3, out_ref, out_vec;
        init_matrix_f32(input, B, in_features, 100 + B);
        init_matrix_f32(W1, in_features, hidden1, 200 + B);
        init_vector_f32(b1, hidden1, 300 + B);
        init_matrix_f32(W2, hidden1, hidden2, 400 + B);
        init_vector_f32(b2, hidden2, 500 + B);
        init_matrix_f32(W3, hidden2, out_classes, 600 + B);
        init_vector_f32(b3, out_classes, 700 + B);

        out_ref.resize(B * out_classes, 0.0f);
        out_vec.resize(B * out_classes, 0.0f);

        ml_mlp_forward_reference(
            input.data(), B, in_features,
            W1.data(), b1.data(), hidden1,
            W2.data(), b2.data(), hidden2,
            W3.data(), b3.data(), out_classes,
            out_ref.data()
        );

        ml_mlp_forward_vectorized(
            input.data(), B, in_features,
            W1.data(), b1.data(), hidden1,
            W2.data(), b2.data(), hidden2,
            W3.data(), b3.data(), out_classes,
            out_vec.data()
        );

        total_tests++;
        auto v = verify_results_f32(out_vec, out_ref, 1e-4, 1e-4);
        if (!v.passed) {
            std::cerr << "FAIL: ml_mlp_forward_vectorized batch=" << B 
                      << " max_abs=" << v.max_absolute_diff << "\n";
            failed_tests++;
        } else {
            std::cout << "PASS: ml_mlp_forward_vectorized batch=" << B << "\n";
        }
    }

    // 4. Memory Locality Sum equivalence test
    {
        size_t num_elements = 1024;
        std::vector<float> buf(num_elements);
        init_vector_f32(buf, num_elements, 999);
        float sum_seq = memory_traverse_sequential(buf.data(), num_elements, 1);
        float sum_strided = memory_traverse_strided(buf.data(), num_elements, 32, 1);

        total_tests++;
        if (std::abs(sum_seq - sum_strided) > 1e-3 * std::abs(sum_seq)) {
            std::cerr << "FAIL: memory_traverse sum mismatch\n";
            failed_tests++;
        } else {
            std::cout << "PASS: memory_traverse sum equivalence\n";
        }
    }

    std::cout << "\nResults: " << (total_tests - failed_tests) << "/" << total_tests << " tests passed.\n";
    return (failed_tests == 0) ? 0 : 1;
}
