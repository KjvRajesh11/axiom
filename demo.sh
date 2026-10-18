#!/usr/bin/env bash
set -e

echo "================================================================================"
echo "         AXIOM 2.2 ADVANCED ARCHITECTURE INTELLIGENCE WALKTHROUGH               "
echo "================================================================================"

echo ""
echo "[Step 1/10] Probing hardware topology, compiler toolchain, and CPU features..."
python3 axiom_cli.py env

echo ""
echo "[Step 2/10] Inspecting registered workloads and candidate transformations..."
python3 axiom_cli.py workloads

echo ""
echo "[Step 3/10] Running full C++ (50 tests) and Python (34 tests) verification suites..."
python3 axiom_cli.py test

echo ""
echo "[Step 4/10] Demonstrating Constrained Clang AST Loop Interchange with Preconditions..."
python3 axiom_cli.py ast-transform --source cpp/workloads/matmul/matmul_reference.cpp --func matmul_reference --out cpp/workloads/generated/matmul_ast_candidate.cpp

echo ""
echo "[Step 5/10] Benchmarking Multithreaded Parallel Scaling (Honest vs Base & Best 1T)..."
python3 axiom_cli.py multithread-sweep

echo ""
echo "[Step 6/10] Running Empirical Roofline Analysis with Separated Hardware Ceilings..."
python3 axiom_cli.py roofline --workload matmul --sizes 128,256,512

echo ""
echo "[Step 7/10] Demonstrating Architecture-Specific AVX-512 Safe Dispatch..."
python3 axiom_cli.py run matmul --candidates reference,neon,avx512 --sizes 128,256 --warmup 2 --runs 5

echo ""
echo "[Step 8/10] Testing Optional Remote AMD Execution Protocol (Mock Zen 4 Genoa)..."
python3 axiom_cli.py remote-run --mock --workload matmul --candidate avx512 --size 256

echo ""
echo "[Step 9/10] Demonstrating Controlled Performance Regression Detection..."
python3 axiom_cli.py regression-demo

echo ""
echo "[Step 10/10] Executing Memory Locality & Working-Set Cache Transition Sweep..."
python3 axiom_cli.py memory-sweep

echo ""
echo "================================================================================"
echo " Axiom 2.2 demonstration completed successfully."
echo " All empirical artifacts preserved in results/ and artifacts/."
echo " To launch the interactive visual dashboard, run: python3 axiom_cli.py dashboard"
echo "================================================================================"
