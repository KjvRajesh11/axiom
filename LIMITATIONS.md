# Axiom 2.2 System Limitations, Boundaries & Verification Status

In accordance with Axiom's scientific integrity contract, this document records all environmental boundaries, hardware constraints, and the exact verification status of every subsystem. Axiom never fabricates unmeasured counters or simulates unsupported features.

---

## 1. Feature Verification Status Matrix

| Subsystem / Capability | Verification Status | Verification Evidence |
| :--- | :---: | :--- |
| **Statistical Integrity (Honest 1T vs Baseline)** | `VERIFIED` | Unit tests (`test_benchmark.py`), CLI reports `vs Base` and `vs Best 1T` across all workloads. |
| **Constrained Clang AST Loop Transformation** | `VERIFIED` | 6 unit tests (`test_ast_transform.py`), AST-driven interchange of `matmul_reference.cpp` generating verified candidate artifact with `max_rel_error = 0.0`. |
| **x86-64 & AVX-512 Compilation** | `VERIFIED` | Apple Clang cross-compiles `matmul_avx512.cpp` with `-target x86_64-apple-macos -mavx512f` producing genuine 512-bit ZMM register instructions (`vfmadd...ps`). |
| **AVX-512 Native Execution on Host** | `PARTIAL` *(Safe Dispatch)* | Host is Apple M4 (ARM64); native execution of ZMM instructions is gated by CPU feature detection. Dispatches safely to portable fallback on host, passing 50/50 tests. |
| **Remote AMD Execution Adapter Protocol** | `VERIFIED` | 3 unit tests (`test_remote.py`), mock transport simulates Zen 4 Genoa Linux protocol, verifying SHA-256 manifests, timeouts, and execution without credentials. |
| **Native AMD Hardware Execution** | `MISSING` *(No Hardware Available)* | Local environment is Apple M4 macOS. No native AMD hardware is available. Adapter is ready for execution once an authorized host is provided via `--host`. |
| **Empirical Roofline Analysis Module** | `VERIFIED` | 4 unit tests (`test_roofline.py`), separates empirical and theoretical ceilings, computes arithmetic intensity, produces ASCII and JSON artifacts. |
| **Compiler Intelligence & Assembly Diff** | `VERIFIED` | Parses live Clang `-Rpass=loop-vectorize` remarks, disassembles targets, diffs instruction categories. |
| **Controlled Regression Detection** | `VERIFIED` | Detects and rejects 4.25x performance regression ($p < 0.0001$, `REJECTED_REGRESSION`). |

---

## 2. Host Architecture & Hardware Scope
- **Host Machine:** All measurements reported natively by Axiom in this repository were collected on an **Apple M4 processor running macOS (Darwin 25.6.0, ARM64)**.
- **No Native AMD Hardware:** The local developer environment does not contain AMD x86-64 hardware. In strict compliance with the engineering contract, **zero AMD measurements are fabricated**.
- **AVX-512 Gating:** The AVX-512 matrix multiplication kernel is cross-compiled and tested for assembly correctness, but when executed natively on the Apple M4 host, runtime CPUID feature detection identifies the ARM64 host and routes execution safely to the portable fallback to avoid `SIGILL` (illegal instruction traps).

---

## 3. Scope of AST Transformations
- **Supported Prototype:** Axiom 2.2 implements an automated Clang AST-driven loop interchange transformer for canonical loop nests.
- **Strict Preconditions:** If loops contain non-canonical bounds, function calls (side effects), non-local jumps, or mutable aliased input pointers, the transformation engine strictly rejects the transformation with a detailed explanation.
- **Unrestricted Rewriting Out of Scope:** Axiom deliberately avoids unconstrained code generation or speculative LLM source mutations that cannot be proven correct.

---

## 4. Roofline Model Boundaries
- **Asymptotic Bandwidth vs Latency:** The empirical Roofline model evaluates steady-state data transfer and compute saturation. It does not model memory latency, TLB page table walks, prefetcher warm-up delays, or NUMA interconnect contention.
- **Ceiling Separation:** Theoretical hardware ceilings reflect architectural peak limits from hardware specifications (e.g., 70.4 GFLOP/s single-core NEON, 120 GB/s DRAM bus); empirical ceilings reflect observed throughput on native benchmarks. They are kept strictly separate.

---

## 5. Profiling & Hardware Performance Counters
- **macOS Userland Profiling:** On macOS, hardware performance counter access (PMCs) requires private kernel entitlements. Axiom utilizes native userland sampling (`/usr/bin/sample`) and compiler-emitted diagnostics. It does not fabricate synthetic cache miss numbers.
