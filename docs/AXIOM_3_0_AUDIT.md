# Axiom 3.0 Comprehensive Ground Truth Audit
## Systems Performance Engineering & Architecture Verification

**Audit Date:** 2026-10-18  
**Host Architecture:** Apple M4 (ARM64 macOS Darwin 25.6.0)  
**Compiler:** Apple Clang version 21.0.0 (`arm64-apple-darwin25.6.0`)  
**Auditor:** Principal Systems Performance & Compiler Infrastructure Engineer  

---

## 1. Executive Summary & Capability Classification

A comprehensive audit of the Axiom 2.2 codebase was performed across all C++, Python, build, and benchmark subsystems. Each capability has been independently verified through clean builds, test execution, assembly inspection, and mathematical evaluation.

### Verification Matrix

| Capability / Subsystem | Status | Audit Findings & Empirical Evidence |
| :--- | :---: | :--- |
| **C++ Build & Correctness Suite** | `VERIFIED` | `./build/axiom_correctness_tests` executed: **50/50 tests passed** (0 failures). Covers GEMM, Conv2D, MLP inference, memory sweeps, AVX-512 safe dispatch, and non-power-of-two tail bounds ($N=17, 33, 49, 65$). |
| **Python Unit & Integration Suite** | `VERIFIED` | `pytest tests -v` executed: **34/34 tests passed** in 31.75s across all test modules. |
| **End-to-End Walkthrough Script** | `VERIFIED` | `./demo.sh` executed cleanly from step 1 to 10 with exit code 0. |
| **Multithreaded Parallel Scaling Reporting** | `VERIFIED` | Reports both speedup vs naive baseline and speedup vs best single-threaded implementation (`speedup_vs_best_single`). Demonstrates 32.92x vs naive scalar, 2.64x vs best 1-core NEON. |
| **Student-t Statistical Significance** | `FAILED` *(Defect)* | In `axiom/benchmark.py`, `welch_t_test` computes degrees of freedom `df` via Welch-Satterthwaite, but then calculates p-value using standard normal complementary error function `math.erfc(z / sqrt(2))` rather than the Student-t distribution CDF. For small sample sizes ($n=5, 15$), this underestimates p-values and claims false precision. |
| **Benchmark Runner State Isolation** | `PARTIAL` *(Defect)* | In `cpp/runner/benchmark_runner.cpp`, output matrix `C_cand` is not re-zeroed between measurement iterations for all candidates, leaving potential warm-cache residency or dirty-state accumulation. |
| **Clang AST Precondition Engine** | `PARTIAL` *(Defect)* | `axiom/ast_transform.py` validates loop bounds, types, and side effects via JSON AST, but code generation assumes fixed parameter names `A, B, C` and GEMM structure without verifying that inner loop reduction AST matches the canonical contraction formula. |
| **x86-64 AVX-512 Cross-Compilation** | `VERIFIED` | `clang++ -target x86_64-apple-macos -mavx512f -O3 -S cpp/workloads/matmul/matmul_avx512.cpp` produces genuine 512-bit `%zmm` registers and `vfmadd213ps` instructions. |
| **AVX-512 Host Safe Dispatch** | `VERIFIED` | `detect_cpu_features()` inspects CPU flags. On ARM64 host, it returns `has_avx512f=false` and dispatches safely to portable fallback without `SIGILL` traps. |
| **Remote AMD Real Execution** | `BLOCKED` *(No Hardware)* | No authorized Linux x86-64 AMD machine or SSH credentials are provided in the environment. |
| **Remote AMD Execution Workflow** | `PARTIAL` *(Defect)* | `axiom/remote.py` implements SSH transport and mock probe, but lacks complete remote source tarball packaging, remote compilation (`cmake && make`), and granular exit code classification (connection vs compilation vs execution vs correctness failure). |
| **Roofline Operational Intensity** | `VERIFIED` | Arithmetic intensity formulas for GEMM ($2N^3 / 12N^2 = N/6$), Conv2D, and memory streaming are mathematically correct. |
| **Roofline Hardware Ceilings** | `PARTIAL` *(Defect)* | `RooflineAnalyzer` hardcodes empirical ceilings (e.g. 53.1 GFLOP/s, 8.2 GB/s) as defaults instead of dynamically calibrating them from actual host measurements. |
| **Controlled Regression Detection** | `VERIFIED` | `cmd_regression_demo` reliably detects 4.25x slowdown on intentionally unoptimized $j$-$k$-$i$ column traversal with $p < 0.0001$. |

---

## 2. Reproduction Baseline & Commands

The baseline was independently reproduced on Apple M4 macOS Darwin 25.6.0:

```bash
# 1. Clean build
cmake -B build -S . && cmake --build build
# Result: 0 warnings, 0 errors, generated axiom_benchmark_runner & axiom_correctness_tests

# 2. C++ test verification
./build/axiom_correctness_tests
# Result: 50/50 PASS

# 3. Python test verification
python3 -m pytest tests -v
# Result: 34 passed in 31.75s

# 4. End-to-end demo walkthrough
./demo.sh
# Result: Steps 1 through 10 exit code 0
```

---

## 3. Discovered Defects & Gaps

### Defect 1: Inaccurate p-Value in Welch's t-Test (Statistical Rigor)
- **File:** `axiom/benchmark.py`, lines 114–122.
- **Description:** Degrees of freedom `df` is calculated, but the two-tailed p-value is computed using `math.erfc(z / math.sqrt(2.0))`, which is the Gaussian normal distribution tail probability. For small degrees of freedom ($df \in [4, 14]$), the normal distribution has significantly lighter tails, leading to severely underestimated p-values (e.g., claiming $p < 0.01$ when the true Student-t p-value is $> 0.05$).
- **Remediation:** Implement the exact Student-t cumulative distribution function using the regularized incomplete beta function (`math.betainc` in Python 3.12+ or a standard continued fraction expansion).

### Defect 2: Benchmark Timing Loop State Isolation
- **File:** `cpp/runner/benchmark_runner.cpp`, lines 105–125.
- **Description:** In the repeated measurement loop, `C_cand` is allocated once and reused across all $R$ iterations without re-initialization. For candidates that accumulate in-place, this produces dirty accumulators. Furthermore, repeated execution over the exact same buffer warms the cache, measuring hot-cache throughput rather than cold/warm steady-state.
- **Remediation:** Add buffer zeroing before each timed invocation, and support cache-flushing options between runs when testing DRAM-bound regimes.

### Defect 3: AST Transformation Precondition Rigor & Generalization
- **File:** `axiom/ast_transform.py`, lines 220–390.
- **Description:** The AST analyzer verifies that a loop nest has depth 3 and canonical induction variables, but does not verify the exact reduction expression AST structure (e.g. `sum += A[...] * B[...]`). If a 3-loop nest computes something else (e.g. stencil or tensor permutation), the code generator blindly emits the GEMM template.
- **Remediation:** Inspect the assignment AST inside the innermost loop: verify `CompoundAssignOperator` (`+=`) or `BinaryOperator` (`=`), verify that array indices correspond to matrix multiplication contraction $i \times k$ and $k \times j$, and reject any loop that does not conform. Add negative unit tests for non-contraction loop bodies.

### Defect 4: Incomplete Remote AMD Execution Workflow
- **File:** `axiom/remote.py`.
- **Description:** The runner simply calls `axiom_remote_exec`, which does not exist on a fresh remote machine. It lacks the source bundling, remote transfer, remote CMake build, execution of `axiom_benchmark_runner`, and granular error handling.
- **Remediation:** Implement a complete remote execution pipeline:
  1. Package source files into an in-memory tarball with SHA-256 manifest.
  2. Transfer payload via SSH/SCP.
  3. Run remote build script with CMake and GCC/Clang with `-O3 -mavx512f`.
  4. Execute benchmark with timeout.
  5. Parse JSON output and classify errors into `CONNECTION_ERROR`, `REMOTE_COMPILATION_ERROR`, `UNSUPPORTED_INSTRUCTION_ERROR`, `CORRECTNESS_ERROR`, `BENCHMARK_ERROR`.
  6. Document exact remote setup in `docs/AXIOM_3_0_REMOTE_AMD_SETUP.md`.

### Defect 5: Static Empirical Roofline Ceilings
- **File:** `axiom/roofline.py`, lines 136–165.
- **Description:** Empirical compute and bandwidth ceilings are hardcoded to Apple M4 constants rather than measured dynamically from host benchmark sweeps.
- **Remediation:** Allow dynamic calibration where empirical ceilings are populated directly from the maximum measured throughput of local microkernels and sequential memory sweeps.

---

## 4. Prioritized Remediation Plan (Axiom 3.0)

1. **Phase B — Benchmark Integrity:**
   - Implement exact Student-t p-value calculation in `axiom/benchmark.py`.
   - Update `cpp/runner/benchmark_runner.cpp` with per-iteration buffer isolation.
   - Add outlier detection explanation (Tukey's fences / IQR) without silent data deletion.
   - Record full compiler flags, CPU topology, and timestamp metadata in all records.

2. **Phase C — AST Transformation & Architecture Hardening:**
   - Strengthen AST preconditions to verify canonical contraction expression structure.
   - Add negative tests for non-contraction loop bodies, irregular indexing, and side effects.
   - Validate AVX-512 x86-64 cross-compilation with `-target x86_64-linux-gnu` and `-target x86_64-apple-macos`.

3. **Phase D — Roofline & Decision Engine Validation:**
   - Add dynamic calibration of empirical ceilings from live microkernel runs.
   - Add tests for edge cases (zero sizes, missing data).
   - Ensure decision engine records full evidence, hypotheses, and uncertainty.

4. **Phase E — Remote AMD Execution Engine:**
   - Implement source payload packaging, transfer, remote build, and execution pipeline.
   - Provide comprehensive mock and unit test suites covering all remote failure modes.
   - Document remote host configuration in `docs/AXIOM_3_0_REMOTE_AMD_SETUP.md`.

5. **Phase F — Regression Testing & Documentation:**
   - Run complete test suite and generate all required documentation.
