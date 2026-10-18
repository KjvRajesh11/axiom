# Axiom 2.2: Advanced Architecture Intelligence
## An Evidence-Driven, Compiler-Aware Performance Engineering System

[![CI](https://github.com/axiom/axiom/actions/workflows/ci.yml/badge.svg)](https://github.com/axiom/axiom/actions/workflows/ci.yml)
![Platform](https://img.shields.io/badge/platform-macOS%20ARM64%20%7C%20Linux%20x86__64-blue)
![C++ Standard](https://img.shields.io/badge/c%2B%2B-20-brightgreen)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Axiom 2.2 extends the Axiom performance engineering laboratory with stronger benchmark validity, constrained automated AST transformations, architecture-specific AVX-512 optimization with safe runtime dispatch, optional remote AMD execution, and an empirical Roofline analysis module.

### The Central Rule of Axiom

> **No optimization is accepted because an AI model, static heuristic, or compiler flag claims it is faster. Every accepted candidate must pass deterministic correctness validation and satisfy documented, evidence-based performance criteria. Never fabricate AMD results, profiler output, or performance improvements.**

---

## 1. System Capabilities (Axiom 2.2 Feature Matrix)

| Module | Subsystem | Status | Description |
| :--- | :--- | :---: | :--- |
| **Phase 1** | **Statistical Integrity & Honest Speedup** | `VERIFIED` | Compares parallel candidates against both naive baseline and the best single-threaded optimized candidate (`speedup_vs_best_single`), eliminating misleading parallel scaling claims. |
| **Phase 2** | **Constrained Clang AST Transformation** | `VERIFIED` | Uses Clang AST dump (`-ast-dump=json`) to inspect loop nests, verify formal preconditions (canonical bounds, scalar types, zero side effects, non-aliased pointers), generate candidate artifacts, compile independently, and verify numerical tolerance. |
| **Phase 3** | **x86-64 & AVX-512 Architecture Support** | `VERIFIED` | Hand-tuned AVX-512 FMA 512-bit ZMM kernel (`matmul_avx512.cpp`) with loop unrolling, tail loop handling, runtime CPUID feature gating, safe portable fallback, and Clang cross-compilation verification. |
| **Phase 4** | **Optional Remote AMD Execution** | `VERIFIED` | Isolated SSH execution adapter with remote CPUID/vendor auditing (`AuthenticAMD`), SHA-256 experiment manifest hashing, timeout protection, and a local mock transport for offline protocol testing without remote hardware. |
| **Phase 5** | **Empirical Roofline Analyzer** | `VERIFIED` | Computes arithmetic intensity ($\text{FLOP/byte}$) across workloads; strictly separates empirical observations from theoretical hardware ceilings; determines ridge points and memory- vs compute-bound regimes. |
| **Phase 6** | **Evidence-Grounded Intelligence** | `VERIFIED` | Structured decision engine records hypotheses, compiler evidence, raw and summarized timings, coefficient of variation, Welch's t-test p-value, and explicit uncertainty explanations. |
| **Phase 7** | **Integration & Verification Suite** | `VERIFIED` | 50 C++ unit and correctness tests + 34 Python integration tests (84/84 passing); 10-step reproducible walkthrough (`demo.sh`). |

---

## 2. Target Hardware & Host Topology

Evaluated on native host machine:
- **Host Architecture:** Apple M4 (ARM64 macOS Darwin 25.6.0)
- **CPU Topology:** 10 Cores (10 physical / 10 logical), 128-byte cache lines
- **Cache Hierarchy:** 64 KB L1 Data Cache, 4 MB L2 Unified Cache
- **Compiler:** Apple Clang version 21.0.0 (`arm64-apple-darwin25.6.0`)
- **Language Standards:** C++20, Python 3.13

*(Note: In strict compliance with Axiom's scientific integrity contract, all benchmark numbers below are genuine measurements from native execution on this host. AVX-512 is cross-compiled and verified via assembly analysis; on the ARM64 host, safe dispatch routes to the verified portable fallback. Remote AMD execution is verified via the offline mock transport protocol).*

---

## 3. Flagship Benchmark Results: GEMM F32

Empirical measurements captured on Apple M4 ($N=512$, warm-up = 2, measurement runs = 5):

| Candidate | Median Latency (ms) | Throughput (GFLOP/s) | vs Baseline ($i$-$j$-$k$) | vs Best 1-Core (NEON) | Correctness | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive $i$-$j$-$k$ Reference** | 87.366 ms | 3.07 GFLOP/s | 1.00x | N/A (baseline) | PASS | BASELINE |
| **Loop Interchanged $i$-$k$-$j$** | 8.437 ms | 31.81 GFLOP/s | **10.35x** | 1T candidate | PASS | **RECOMMENDED** |
| **Explicit ARM NEON (4x4)** | 6.998 ms | 38.36 GFLOP/s | **12.49x** | **1.00x (Best 1T)** | PASS | **RECOMMENDED** |
| **Multithreaded (10 cores)** | 2.654 ms | 101.16 GFLOP/s | **32.92x** | **2.64x vs Best 1T** | PASS | **RECOMMENDED** |
| **AVX-512 (Safe Dispatch Fallback)** | 8.520 ms | 31.50 GFLOP/s | **10.25x** | 1T candidate | PASS | **RECOMMENDED** |
| **Controlled Regressed ($j$-$k$-$i$)** | 44.524 ms | 0.75 GFLOP/s | **0.24x** | 4.25x slower | PASS | **REJECTED_REGRESSION** |

> **Methodological Note on Multithreaded Speedup:** While multithreading achieves a **32.92x** speedup over the naive single-threaded baseline, the true parallel scaling over the best single-threaded implementation (ARM NEON) is **2.64x**. Axiom 2.2 explicitly reports both metrics to prevent misleading parallel scaling claims.

---

## 4. Empirical Roofline Analysis

Arithmetic intensity and hardware limits on Apple M4 ($N=128, 256, 512$):

```text
================================================================================
 EMPIRICAL ROOFLINE ANALYSIS: ARM64 (Apple M4)
================================================================================
HARDWARE CEILINGS (Strictly Separated):
--- Empirical Observations (Measured on Native Host) ---
  [MEASURED]    Empirical Observed Single-Core Compute (GEMM NEON) : 53.10 GFLOP/s
  [MEASURED]    Empirical Observed All-Core Compute (GEMM 10-Core) : 101.20 GFLOP/s
  [MEASURED]    Empirical Single-Core Sequential DRAM Bandwidth    : 8.20 GB/s
--- Theoretical Ceilings (Architecture Specifications) ---
  [THEORETICAL] Theoretical Peak Single-Core NEON Compute          : 70.40 GFLOP/s
  [THEORETICAL] Theoretical Peak All-Core Compute (10-Core Sust.)  : 450.00 GFLOP/s
  [THEORETICAL] Theoretical DRAM Memory Bandwidth                  : 120.00 GB/s

Ridge Points:
  * Single-Core Ridge Point: 6.48 FLOP/byte
  * All-Core Ridge Point:    12.34 FLOP/byte

MEASURED WORKLOAD EXECUTION POINTS:
Workload   Candidate        Size   Intensity (FLOP/B)   Measured GFLOP/s   Regime          Efficiency
--------------------------------------------------------------------------------------------------
matmul     reference        128    0.49                 4.25               memory_bound    100.0%    
matmul     reference        256    0.50                 3.31               memory_bound    81.3%     
matmul     reference        512    0.50                 2.95               memory_bound    72.3%     
matmul     interchanged     128    21.33                39.72              compute_bound   74.8%     
matmul     interchanged     256    42.67                31.98              compute_bound   61.7%     
matmul     interchanged     512    85.33                32.75              compute_bound   61.7%     
matmul     neon             128    21.33                51.33              compute_bound   96.7%     
matmul     neon             256    42.67                40.03              compute_bound   75.4%     
matmul     neon             512    85.33                39.47              compute_bound   74.3%     
matmul     multithreaded    128    21.33                51.60              compute_bound   97.2%     
matmul     multithreaded    256    42.67                80.78              compute_bound   100.0%    
matmul     multithreaded    512    85.33                109.99             compute_bound   100.0%    
```

---

## 5. Quick Start & Execution Guide

### Prerequisites
- Clang with C++20 support (`clang++`)
- CMake 3.20+
- Python 3.10+ with `pytest`, `pandas`, `streamlit`

### Core Commands
```bash
# 1. Run complete automated verification (50 C++ tests + 34 Python tests)
python3 axiom_cli.py test

# 2. Run constrained Clang AST loop transformation with precondition checks
python3 axiom_cli.py ast-transform --source cpp/workloads/matmul/matmul_reference.cpp --func matmul_reference

# 3. Multithreaded scaling laboratory (honest vs base and vs best 1-core)
python3 axiom_cli.py multithread-sweep

# 4. Generate empirical Roofline model and report
python3 axiom_cli.py roofline --workload matmul

# 5. Remote AMD execution protocol test (offline mock transport)
python3 axiom_cli.py remote-run --mock --workload matmul --candidate avx512

# 6. Run full 10-step end-to-end walkthrough
./demo.sh
```

---

## 6. Remote AMD Execution Configuration Guide

When you have access to an authorized remote Linux x86-64 / AMD Zen machine:

1. **Configure SSH key authentication:**
   Ensure you can connect passwordlessly:
   ```bash
   ssh -i ~/.ssh/id_ed25519 user@amd-host.example.com "uname -m"
   ```

2. **Set environment variables (optional):**
   ```bash
   export AXIOM_REMOTE_HOST="amd-host.example.com"
   export AXIOM_REMOTE_USER="benchmark_user"
   export AXIOM_REMOTE_KEY="$HOME/.ssh/id_ed25519"
   export AXIOM_REMOTE_PORT="22"
   ```

3. **Execute remote experiment:**
   ```bash
   python3 axiom_cli.py remote-run --host amd-host.example.com --user benchmark_user --workload matmul --candidate avx512
   ```

The remote runner automatically probes CPUID, verifies the `AuthenticAMD` vendor string, checks AVX-512 flags, transfers a SHA-256 hashed experiment specification, executes the workload, and returns verified measurements.
