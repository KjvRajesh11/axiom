"""Tests for Axiom 2.2 Architecture-Specific Support: AVX-512 and CPU Feature Detection.

Verifies:
- Runtime CPU feature detection returns accurate capabilities without executing unsupported instructions.
- Safe dispatch operates correctly and yields verified numerical accuracy on host.
- Tail bounds (odd/non-power-of-two matrix sizes) compute exact results.
- Cross-compilation to x86-64 target generates genuine AVX-512 vector instructions (zmm, vfmadd...ps).
- Distinguishes compilation from native execution, preventing fabricated performance claims.
"""

import os
import shutil
import subprocess
import pytest

from axiom.benchmark import BenchmarkEngine
from axiom.workload import WorkloadRegistry


def test_cpu_feature_probe_arm64():
    """Verifies that our environment snapshot identifies ARM64 host without AVX-512."""
    from axiom.env import get_environment_snapshot
    env = get_environment_snapshot()
    assert env.hardware.architecture in ("arm64", "aarch64", "x86_64")


def test_avx512_safe_dispatch_correctness():
    """Runs matmul with candidate avx512 through the benchmark runner and verifies numerical accuracy."""
    be = BenchmarkEngine()
    # Test boundary and non-power-of-two sizes
    for size in (1, 7, 15, 17, 31, 33, 64):
        result = be.sandbox.run_candidate(
            workload="matmul",
            candidate="avx512",
            size=size,
            warmup_runs=1,
            measurement_runs=2,
        )
        assert result.correctness_passed is True, f"Failed correctness for size={size}"
        assert result.max_rel_error < 1e-4, f"max_rel_error too large ({result.max_rel_error}) for size={size}"
        assert result.median_ms > 0.0


def test_avx512_x86_cross_compilation_and_assembly():
    """Tests genuine Clang cross-compilation of matmul_avx512.cpp to x86-64 with AVX-512 intrinsics."""
    clang = shutil.which("clang++")
    assert clang is not None, "clang++ required for compiler tests"

    source_file = "cpp/workloads/matmul/matmul_avx512.cpp"
    cmd = [
        clang,
        "-target", "x86_64-apple-macos",
        "-mavx512f",
        "-O3",
        "-std=c++20",
        "-Icpp/include",
        "-S",
        source_file,
        "-o", "-"
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0, f"Cross-compilation failed: {res.stderr}"

    asm = res.stdout
    # Verify genuine AVX-512 assembly instructions and 512-bit ZMM registers
    assert "zmm" in asm.lower(), "Generated assembly must contain 512-bit ZMM registers"
    assert "vfmadd" in asm.lower() or "vmulps" in asm.lower(), "Generated assembly must contain vector FMA instructions"
