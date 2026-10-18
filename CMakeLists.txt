cmake_minimum_required(VERSION 3.20)
project(axiom_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization and warning flags
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -O3
    )
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)

# Source files shared across runner and test binaries
set(AXIOM_SOURCES
    cpp/workloads/matmul/matmul_reference.cpp
    cpp/workloads/matmul/matmul_interchanged.cpp
    cpp/workloads/matmul/matmul_blocked.cpp
    cpp/workloads/matmul/matmul_neon.cpp
    cpp/workloads/matmul/matmul_multithreaded.cpp
    cpp/workloads/matmul/matmul_regressed.cpp
    cpp/workloads/matmul/matmul_avx512.cpp
    cpp/workloads/conv2d/conv2d_reference.cpp
    cpp/workloads/conv2d/conv2d_row_major.cpp
    cpp/workloads/conv2d/conv2d_tiled.cpp
    cpp/workloads/memory/memory_locality.cpp
    cpp/workloads/ml_infer/ml_mlp_infer.cpp
)

# Benchmark runner executable
add_executable(axiom_benchmark_runner
    cpp/runner/benchmark_runner.cpp
    ${AXIOM_SOURCES}
)

# Test executable for correctness verification
add_executable(axiom_correctness_tests
    cpp/runner/correctness_tests.cpp
    ${AXIOM_SOURCES}
)
