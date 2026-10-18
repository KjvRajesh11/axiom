#include "axiom_benchmark.hpp"
#include "axiom_utils.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>

using namespace axiom;

// Helper to calculate statistics
static void calculate_stats(
    const std::vector<double>& times_ms,
    double& min_val,
    double& max_val,
    double& median_val,
    double& mean_val,
    double& stddev_val,
    double& iqr_val
) {
    if (times_ms.empty()) return;
    std::vector<double> sorted = times_ms;
    std::sort(sorted.begin(), sorted.end());

    min_val = sorted.front();
    max_val = sorted.back();

    size_t n = sorted.size();
    if (n % 2 == 0) {
        median_val = 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    } else {
        median_val = sorted[n / 2];
    }

    double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    mean_val = sum / n;

    double sq_sum = 0.0;
    for (double t : sorted) {
        sq_sum += (t - mean_val) * (t - mean_val);
    }
    stddev_val = std::sqrt(sq_sum / n);

    // IQR = Q3 - Q1
    double q1 = sorted[n / 4];
    double q3 = sorted[(3 * n) / 4];
    iqr_val = q3 - q1;
}

int main(int argc, char** argv) {
    // Arguments: <workload> <candidate> <size> <warmup> <runs>
    // Example: axiom_benchmark_runner matmul neon 256 3 15
    std::string workload = (argc > 1) ? argv[1] : "matmul";
    std::string candidate = (argc > 2) ? argv[2] : "reference";
    int size = (argc > 3) ? std::stoi(argv[3]) : 256;
    int warmup = (argc > 4) ? std::stoi(argv[4]) : 3;
    int runs = (argc > 5) ? std::stoi(argv[5]) : 15;

    std::vector<double> run_times_ms;
    run_times_ms.reserve(runs);

    bool correctness_passed = true;
    double max_rel_err = 0.0;
    double max_abs_err = 0.0;
    double gflops = 0.0;

    if (workload == "matmul") {
        int N = size;
        std::vector<float> A, B, C_ref, C_cand;
        init_matrix_f32(A, N, N, 100);
        init_matrix_f32(B, N, N, 200);
        C_ref.resize(N * N, 0.0f);
        C_cand.resize(N * N, 0.0f);

        // Compute reference for correctness
        matmul_reference(A.data(), B.data(), C_ref.data(), N);
        doNotOptimizeAway(C_ref.data());

        // Warmup runs
        for (int w = 0; w < warmup; ++w) {
            std::fill(C_cand.begin(), C_cand.end(), 0.0f);
            clobberMemory<float>();
            if (candidate == "reference") {
                matmul_reference(A.data(), B.data(), C_cand.data(), N);
            } else if (candidate == "interchanged") {
                matmul_interchanged(A.data(), B.data(), C_cand.data(), N);
            } else if (candidate == "blocked") {
                matmul_blocked(A.data(), B.data(), C_cand.data(), N);
            } else if (candidate == "neon") {
                matmul_neon(A.data(), B.data(), C_cand.data(), N);
            } else if (candidate == "multithreaded") {
                matmul_multithreaded(A.data(), B.data(), C_cand.data(), N);
            } else if (candidate == "regressed") {
                matmul_regressed(A.data(), B.data(), C_cand.data(), N);
            } else if (candidate == "avx512") {
                matmul_avx512_dispatched(A.data(), B.data(), C_cand.data(), N);
            } else {
                std::cerr << "Unknown matmul candidate: " << candidate << "\n";
                return 1;
            }
            doNotOptimizeAway(C_cand.data());
        }

        // Measurement runs
        for (int r = 0; r < runs; ++r) {
            std::fill(C_cand.begin(), C_cand.end(), 0.0f);
            clobberMemory<float>();
            auto t0 = BenchmarkClock::now();
            if (candidate == "reference") {
                matmul_reference(A.data(), B.data(), C_cand.data(), N);
            } else if (candidate == "interchanged") {
                matmul_interchanged(A.data(), B.data(), C_cand.data(), N);
            } else if (candidate == "blocked") {
                matmul_blocked(A.data(), B.data(), C_cand.data(), N);
            } else if (candidate == "neon") {
                matmul_neon(A.data(), B.data(), C_cand.data(), N);
            } else if (candidate == "multithreaded") {
                matmul_multithreaded(A.data(), B.data(), C_cand.data(), N);
            } else if (candidate == "regressed") {
                matmul_regressed(A.data(), B.data(), C_cand.data(), N);
            } else if (candidate == "avx512") {
                matmul_avx512_dispatched(A.data(), B.data(), C_cand.data(), N);
            }
            doNotOptimizeAway(C_cand.data());
            auto t1 = BenchmarkClock::now();
            run_times_ms.push_back(BenchmarkClock::durationMs(t0, t1));
        }

        auto verif = verify_results_f32(C_cand, C_ref, 1e-4, 1e-4);
        correctness_passed = verif.passed;
        max_abs_err = verif.max_absolute_diff;
        max_rel_err = verif.max_relative_diff;

        double total_flops = 2.0 * static_cast<double>(N) * N * N;
        double min_ms, max_ms, median_ms, mean_ms, stddev_ms, iqr_ms;
        calculate_stats(run_times_ms, min_ms, max_ms, median_ms, mean_ms, stddev_ms, iqr_ms);
        if (median_ms > 0.0) {
            gflops = (total_flops / (median_ms * 1e-3)) / 1e9;
        }

    } else if (workload == "conv2d") {
        int H = size, W = size, K = 5;
        int out_H = H - K + 1, out_W = W - K + 1;

        std::vector<float> input, kernel, out_ref, out_cand;
        init_matrix_f32(input, H, W, 101);
        init_matrix_f32(kernel, K, K, 202);
        out_ref.resize(out_H * out_W, 0.0f);
        out_cand.resize(out_H * out_W, 0.0f);

        conv2d_reference(input.data(), H, W, kernel.data(), K, out_ref.data());
        doNotOptimizeAway(out_ref.data());

        for (int w = 0; w < warmup; ++w) {
            std::fill(out_cand.begin(), out_cand.end(), 0.0f);
            clobberMemory<float>();
            if (candidate == "reference") {
                conv2d_reference(input.data(), H, W, kernel.data(), K, out_cand.data());
            } else if (candidate == "row_major") {
                conv2d_row_major(input.data(), H, W, kernel.data(), K, out_cand.data());
            } else if (candidate == "tiled") {
                conv2d_tiled(input.data(), H, W, kernel.data(), K, out_cand.data());
            } else {
                std::cerr << "Unknown conv2d candidate: " << candidate << "\n";
                return 1;
            }
            doNotOptimizeAway(out_cand.data());
        }

        for (int r = 0; r < runs; ++r) {
            std::fill(out_cand.begin(), out_cand.end(), 0.0f);
            clobberMemory<float>();
            auto t0 = BenchmarkClock::now();
            if (candidate == "reference") {
                conv2d_reference(input.data(), H, W, kernel.data(), K, out_cand.data());
            } else if (candidate == "row_major") {
                conv2d_row_major(input.data(), H, W, kernel.data(), K, out_cand.data());
            } else if (candidate == "tiled") {
                conv2d_tiled(input.data(), H, W, kernel.data(), K, out_cand.data());
            }
            doNotOptimizeAway(out_cand.data());
            auto t1 = BenchmarkClock::now();
            run_times_ms.push_back(BenchmarkClock::durationMs(t0, t1));
        }

        auto verif = verify_results_f32(out_cand, out_ref, 1e-4, 1e-4);
        correctness_passed = verif.passed;
        max_abs_err = verif.max_absolute_diff;
        max_rel_err = verif.max_relative_diff;

        double total_flops = 2.0 * static_cast<double>(out_H) * out_W * K * K;
        double min_ms, max_ms, median_ms, mean_ms, stddev_ms, iqr_ms;
        calculate_stats(run_times_ms, min_ms, max_ms, median_ms, mean_ms, stddev_ms, iqr_ms);
        if (median_ms > 0.0) {
            gflops = (total_flops / (median_ms * 1e-3)) / 1e9;
        }

    } else if (workload == "memory") {
        // size = buffer size in KB (e.g. 64 KB, 1024 KB, 16384 KB)
        size_t bytes = static_cast<size_t>(size) * 1024;
        size_t num_elements = bytes / sizeof(float);
        std::vector<float> buffer(num_elements);
        init_vector_f32(buffer, num_elements, 555);

        // Stride: 32 floats = 128 bytes (1 Apple Silicon cache line)
        size_t stride_elements = 32;
        int iters = (size <= 256) ? 100 : ((size <= 4096) ? 20 : 5);

        for (int w = 0; w < warmup; ++w) {
            float s = 0.0f;
            if (candidate == "sequential") {
                s = memory_traverse_sequential(buffer.data(), num_elements, iters);
            } else if (candidate == "strided") {
                s = memory_traverse_strided(buffer.data(), num_elements, stride_elements, iters);
            } else {
                std::cerr << "Unknown memory candidate: " << candidate << "\n";
                return 1;
            }
            doNotOptimizeAway(s);
        }

        for (int r = 0; r < runs; ++r) {
            auto t0 = BenchmarkClock::now();
            float s = 0.0f;
            if (candidate == "sequential") {
                s = memory_traverse_sequential(buffer.data(), num_elements, iters);
            } else if (candidate == "strided") {
                s = memory_traverse_strided(buffer.data(), num_elements, stride_elements, iters);
            }
            doNotOptimizeAway(s);
            auto t1 = BenchmarkClock::now();
            run_times_ms.push_back(BenchmarkClock::durationMs(t0, t1));
        }

        correctness_passed = true;
        // Throughput in GB/s: (bytes * iters) / (median_ms * 1e-3) / 1e9
        double total_bytes = static_cast<double>(bytes) * iters;
        double min_ms, max_ms, median_ms, mean_ms, stddev_ms, iqr_ms;
        calculate_stats(run_times_ms, min_ms, max_ms, median_ms, mean_ms, stddev_ms, iqr_ms);
        if (median_ms > 0.0) {
            gflops = (total_bytes / (median_ms * 1e-3)) / 1e9; // represents GB/s
        }

    } else if (workload == "ml_infer") {
        // size = batch size (e.g. 1, 8, 32, 64)
        int batch_size = size;
        int in_features = 64;
        int hidden1 = 128;
        int hidden2 = 64;
        int out_classes = 10;

        std::vector<float> input, W1, b1, W2, b2, W3, b3, out_ref, out_cand;
        init_matrix_f32(input, batch_size, in_features, 111);
        init_matrix_f32(W1, in_features, hidden1, 222);
        init_vector_f32(b1, hidden1, 333);
        init_matrix_f32(W2, hidden1, hidden2, 444);
        init_vector_f32(b2, hidden2, 555);
        init_matrix_f32(W3, hidden2, out_classes, 666);
        init_vector_f32(b3, out_classes, 777);

        out_ref.resize(batch_size * out_classes, 0.0f);
        out_cand.resize(batch_size * out_classes, 0.0f);

        ml_mlp_forward_reference(
            input.data(), batch_size, in_features,
            W1.data(), b1.data(), hidden1,
            W2.data(), b2.data(), hidden2,
            W3.data(), b3.data(), out_classes,
            out_ref.data()
        );
        doNotOptimizeAway(out_ref.data());

        for (int w = 0; w < warmup; ++w) {
            std::fill(out_cand.begin(), out_cand.end(), 0.0f);
            clobberMemory<float>();
            if (candidate == "reference") {
                ml_mlp_forward_reference(
                    input.data(), batch_size, in_features,
                    W1.data(), b1.data(), hidden1,
                    W2.data(), b2.data(), hidden2,
                    W3.data(), b3.data(), out_classes,
                    out_cand.data()
                );
            } else if (candidate == "vectorized") {
                ml_mlp_forward_vectorized(
                    input.data(), batch_size, in_features,
                    W1.data(), b1.data(), hidden1,
                    W2.data(), b2.data(), hidden2,
                    W3.data(), b3.data(), out_classes,
                    out_cand.data()
                );
            } else {
                std::cerr << "Unknown ml_infer candidate: " << candidate << "\n";
                return 1;
            }
            doNotOptimizeAway(out_cand.data());
        }

        for (int r = 0; r < runs; ++r) {
            std::fill(out_cand.begin(), out_cand.end(), 0.0f);
            clobberMemory<float>();
            auto t0 = BenchmarkClock::now();
            if (candidate == "reference") {
                ml_mlp_forward_reference(
                    input.data(), batch_size, in_features,
                    W1.data(), b1.data(), hidden1,
                    W2.data(), b2.data(), hidden2,
                    W3.data(), b3.data(), out_classes,
                    out_cand.data()
                );
            } else if (candidate == "vectorized") {
                ml_mlp_forward_vectorized(
                    input.data(), batch_size, in_features,
                    W1.data(), b1.data(), hidden1,
                    W2.data(), b2.data(), hidden2,
                    W3.data(), b3.data(), out_classes,
                    out_cand.data()
                );
            }
            doNotOptimizeAway(out_cand.data());
            auto t1 = BenchmarkClock::now();
            run_times_ms.push_back(BenchmarkClock::durationMs(t0, t1));
        }

        auto verif = verify_results_f32(out_cand, out_ref, 1e-4, 1e-4);
        correctness_passed = verif.passed;
        max_abs_err = verif.max_absolute_diff;
        max_rel_err = verif.max_relative_diff;

        // Total FLOPs: 2 * batch_size * (in*h1 + h1*h2 + h2*out)
        double total_flops = 2.0 * batch_size * (
            static_cast<double>(in_features) * hidden1 +
            static_cast<double>(hidden1) * hidden2 +
            static_cast<double>(hidden2) * out_classes
        );
        double min_ms, max_ms, median_ms, mean_ms, stddev_ms, iqr_ms;
        calculate_stats(run_times_ms, min_ms, max_ms, median_ms, mean_ms, stddev_ms, iqr_ms);
        if (median_ms > 0.0) {
            gflops = (total_flops / (median_ms * 1e-3)) / 1e9;
        }

    } else {
        std::cerr << "Unknown workload: " << workload << "\n";
        return 1;
    }

    double min_ms, max_ms, median_ms, mean_ms, stddev_ms, iqr_ms;
    calculate_stats(run_times_ms, min_ms, max_ms, median_ms, mean_ms, stddev_ms, iqr_ms);

    // Emit structured JSON output
    std::cout << "{\n";
    std::cout << "  \"workload\": \"" << workload << "\",\n";
    std::cout << "  \"candidate\": \"" << candidate << "\",\n";
    std::cout << "  \"size\": " << size << ",\n";
    std::cout << "  \"warmup_runs\": " << warmup << ",\n";
    std::cout << "  \"measurement_runs\": " << runs << ",\n";
    std::cout << "  \"correctness_passed\": " << (correctness_passed ? "true" : "false") << ",\n";
    std::cout << "  \"max_abs_error\": " << std::scientific << std::setprecision(6) << max_abs_err << ",\n";
    std::cout << "  \"max_rel_error\": " << std::scientific << std::setprecision(6) << max_rel_err << ",\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  \"min_ms\": " << min_ms << ",\n";
    std::cout << "  \"max_ms\": " << max_ms << ",\n";
    std::cout << "  \"median_ms\": " << median_ms << ",\n";
    std::cout << "  \"mean_ms\": " << mean_ms << ",\n";
    std::cout << "  \"stddev_ms\": " << stddev_ms << ",\n";
    std::cout << "  \"iqr_ms\": " << iqr_ms << ",\n";
    std::cout << "  \"gflops\": " << gflops << ",\n";
    std::cout << "  \"raw_times_ms\": [";
    for (size_t i = 0; i < run_times_ms.size(); ++i) {
        std::cout << run_times_ms[i] << (i + 1 < run_times_ms.size() ? ", " : "");
    }
    std::cout << "]\n";
    std::cout << "}\n";

    return correctness_passed ? 0 : 2;
}
