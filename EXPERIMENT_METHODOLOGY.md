# Axiom 2.2 Scientific Experiment Methodology
## Advanced Performance Verification, Statistical Rigor & Architecture Modeling

## 1. Principles of Empirical Measurement

Rigorous performance engineering requires establishing controlled experimental conditions and eliminating common measurement artifacts. Axiom 2.2 implements the following experimental safeguards:

### A. Dead-Code Elimination Prevention
When compiling at `-O3`, modern compilers aggressively eliminate code whose side effects do not escape the function scope or whose outputs are never read. Axiom prevents dead-code elimination using two explicit barriers in C++:
```cpp
template <typename T>
inline void doNotOptimizeAway(T const& value) {
#if defined(__clang__) || defined(__GNUC__)
    asm volatile("" : : "r,m"(value) : "memory");
#else
    volatile const T* p = &value;
    (void)p;
#endif
}
```
The inline assembly instruction informs LLVM that memory has been read and modified, forcing the compiler to materialize the array computation in memory.

### B. High-Resolution Monotonic Timing
Axiom uses `std::chrono::steady_clock`, which is guaranteed never to jump or be adjusted by NTP or system clock synchronizations. Timers wrap only the computational kernel, excluding data allocation, random initialization, and verification passes.

### C. Warm-Up Phases
Before timing measurement trials, each candidate executes $W$ unmeasured warm-up iterations (default $W=3$). This ensures:
1. Dynamic voltage and frequency scaling (DVFS) ramps CPU clock frequency from idle state to active benchmark state.
2. Initial instruction cache (I-Cache) and data cache (D-Cache) cold misses do not skew the first measurement run.
3. CPU branch predictors populate branch history tables with steady-state traversal patterns.

---

## 2. Numerical Correctness Criteria

No candidate is considered an optimization if it produces incorrect arithmetic outputs. In single-precision floating point (`float32`), non-associativity of addition means that loop interchange or re-ordered accumulations may yield minute numerical differences due to rounding:
$$(a + b) + c \ne a + (b + c)$$

To distinguish legitimate floating-point rounding from computational bugs, Axiom enforces a dual absolute and relative tolerance threshold:
$$|y_{\text{candidate}}[i] - y_{\text{reference}}[i]| \le \epsilon_{\text{abs}} + \epsilon_{\text{rel}} \cdot |y_{\text{reference}}[i]|$$
where $\epsilon_{\text{abs}} = 10^{-4}$ and $\epsilon_{\text{rel}} = 10^{-4}$.

If a candidate produces even a single element violating this bound, the run is immediately marked as `FAILED` and rejected by the Decision Engine.

---

## 3. Honest Parallel Scaling Methodology

A frequent flaw in published systems benchmarks is computing parallel speedup by dividing multithreaded runtime by a slow, unoptimized naive single-threaded baseline. For example, comparing a 10-core parallel GEMM against an unvectorized $i$-$j$-$k$ scalar loop yields an apparent "33x speedup on 10 cores", which violates Amdahl's Law and misrepresents parallel efficiency.

**Axiom 2.2 Mandate:**
Every parallel evaluation must explicitly measure and report both:
1. **Speedup vs Baseline:** $\text{Speedup}_{\text{base}} = \frac{T_{\text{naive}}}{T_{\text{multithreaded}}}$ (reflects total engineering gains).
2. **Speedup vs Best Single-Threaded:** $\text{Speedup}_{\text{best1T}} = \frac{T_{\text{best\_single}}}{T_{\text{multithreaded}}}$ (reflects genuine multicore parallel scaling).

On Apple M4:
- $N=512$ Naive reference: 87.37 ms (3.07 GFLOP/s).
- $N=512$ Best 1-core NEON: 6.99 ms (38.36 GFLOP/s).
- $N=512$ 10-core parallel: 2.65 ms (101.16 GFLOP/s).
- Speedup vs baseline: $32.92\times$.
- Genuine parallel speedup vs best 1-core: **$2.64\times$**.

---

## 4. Constrained AST Preconditions Methodology

Axiom 2.2 avoids unrestricted, speculative source rewriting in favor of formally constrained AST transformations. Prior to transforming a loop nest via Clang AST JSON extraction, the transformation engine verifies four formal preconditions:
1. **Canonical Loop Bounds:** Loop induction variable initialized to 0, tested strictly with `< upper_bound`, incremented by 1 (`++var`).
2. **Type Safety:** Induction variable is integral (`int`, `size_t`); array elements are scalar primitives (`float`).
3. **Absence of Side Effects:** Recursive traversal of the loop subtree must find zero function calls (`CallExpr`), non-local jumps (`goto`, `break`, `return`), or thrown exceptions.
4. **Pointer Isolation:** Input buffers must be `const`-qualified with isolated destination buffers to prevent invalid aliasing.

---

## 5. Empirical Roofline Modeling Methodology

The Roofline model relates computational throughput $P$ (GFLOP/s) to arithmetic intensity $I$ (FLOP/byte):
$$P = \min(P_{\text{peak}}, B_{\text{peak}} \times I)$$

### A. Separation of Empirical and Theoretical Ceilings
- **Empirical Ceilings:** Derived from native benchmark trials (e.g., peak observed NEON microkernel throughput of 53.1 GFLOP/s, measured sequential DRAM bandwidth of 8.2 GB/s).
- **Theoretical Ceilings:** Derived from manufacturer hardware specifications (e.g., Apple M4 4.4 GHz dual-NEON peak of 70.4 GFLOP/s, 128-bit LPDDR5X-7500 bus bandwidth of 120 GB/s).
- Axiom strictly labels the provenance and assumptions of every ceiling and never blends theoretical ceilings with empirical data points.

### B. Regime Classification
- Ridge point: $I_{\text{ridge}} = \frac{P_{\text{empirical}}}{B_{\text{empirical}}}$.
- If $I < I_{\text{ridge}}$: workload is classified as **`memory_bound`**.
- If $I \ge I_{\text{ridge}}$: workload is classified as **`compute_bound`**.

---

## 6. Remote Hardware Provenance & Attestation Protocol

When remote execution is configured:
1. The remote transport executes a pre-flight probe (`/proc/cpuinfo`, `uname -m`, compiler version) to audit the remote CPU architecture.
2. The runner verifies the `AuthenticAMD` vendor string and presence of `avx512f` flags before dispatching AVX-512 targets.
3. Every experiment payload is versioned and hashed with SHA-256.
4. If testing offline without remote access, the runner uses `MockAMDTransport`, which explicitly stamps all outputs with `is_mock=True` to eliminate any possibility of fabricated hardware results.
