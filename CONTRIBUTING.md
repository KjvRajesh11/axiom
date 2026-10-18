# Contributing to Axiom 2.1

We welcome contributions to Axiom 2.1. To maintain research-grade engineering integrity, all contributions must adhere to the following standards.

---

## 1. Core Contribution Principles
1. **Never Fabricate Data:** Never hardcode artificial benchmark numbers, synthetic cache counters, or fake speedups. All results must be measurable on native hardware.
2. **Deterministic Correctness:** Every new workload or candidate must implement a numerical verification check against a trusted reference baseline.
3. **Preserve Working Systems:** Ensure all existing C++ and Python tests pass before opening a pull request.

---

## 2. Adding a New Workload

To register a new bounded workload in Axiom:
1. **Create the C++ Implementation:**
   - Add reference baseline to `cpp/workloads/<workload_name>/<workload_name>_reference.cpp`.
   - Add candidate transformations to `cpp/workloads/<workload_name>/<workload_name>_<candidate>.cpp`.
   - Declare function signatures in `cpp/include/axiom_utils.hpp`.
2. **Register in C++ Runner & Tests:**
   - Update `cpp/runner/benchmark_runner.cpp` to support dispatching the new workload.
   - Update `cpp/runner/correctness_tests.cpp` with multi-size numerical tolerance assertions.
   - Update `CMakeLists.txt` to include the new source files.
3. **Register in Python Workload Manager:**
   - In `axiom/workload.py`, instantiate a `WorkloadDefinition` and register it in `WorkloadRegistry`.
   - Define supported input sizes, default size, FLOP/bandwidth formula, and numerical tolerance.
4. **Formulate Testable Hypotheses:**
   - In `axiom/hypothesis.py`, add hypotheses detailing expected performance mechanisms, bound regimes, and falsification rules.

---

## 3. Pull Request Validation Checklist

Before submitting a pull request, run the full validation suite locally:
```bash
# 1. Clean build C++
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4

# 2. Run all tests (40 C++ tests + 18 Python tests)
./axiom_cli.py test

# 3. Test CLI workflow end-to-end
./axiom_cli.py diff-asm matmul reference interchanged
./axiom_cli.py run matmul --candidates reference,interchanged,neon,multithreaded --sizes 128,256
./axiom_cli.py regression-demo
./axiom_cli.py memory-sweep
./axiom_cli.py run ml_infer --sizes 8,32
```
All tests must pass with zero compiler warnings under `-Wall -Wextra -Wpedantic`.
