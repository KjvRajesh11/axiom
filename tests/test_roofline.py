"""Unit tests for Axiom 2.2 Empirical Roofline Analysis Module.

Tests:
- Operational intensity (FLOP/byte) calculations for GEMM, Conv2D, and Memory.
- Strict separation between empirical and theoretical ceilings.
- Ridge point (knee point) calculation.
- Regime classification (memory-bound vs compute-bound).
- ASCII roofline report generation.
"""

import pytest

from axiom.roofline import (
    ArithmeticIntensityCalculator,
    RooflineAnalyzer,
)


def test_arithmetic_intensity_matmul():
    # GEMM F32 N=64
    # Total FLOPs = 2 * 64^3 = 524,288
    # Compulsory bytes (blocked/vector) = 12 * 64^2 = 49,152
    # I_compulsory = 524,288 / 49,152 = 64 / 6 = 10.666...
    flops, bytes_cnt, ai = ArithmeticIntensityCalculator.calculate_matmul(64, "interchanged")
    assert flops == 2 * (64 ** 3)
    assert bytes_cnt == 12 * (64 ** 2)
    assert pytest.approx(ai, rel=1e-3) == 64.0 / 6.0

    # Naive reference with zero cache reuse on B
    flops_ref, bytes_ref, ai_ref = ArithmeticIntensityCalculator.calculate_matmul(64, "reference")
    assert ai_ref < 1.0  # Naive is severely bandwidth-constrained


def test_arithmetic_intensity_conv2d():
    flops, bytes_cnt, ai = ArithmeticIntensityCalculator.calculate_conv2d(64, 64, 5, "row_major")
    assert flops > 0
    assert bytes_cnt > 0
    assert ai > 0


def test_roofline_ceilings_separation():
    analyzer = RooflineAnalyzer(architecture="arm64", cpu_model="Apple M4")
    emp_ceilings = [c for c in analyzer.ceilings if c.is_empirical]
    theo_ceilings = [c for c in analyzer.ceilings if not c.is_empirical]

    assert len(emp_ceilings) >= 2
    assert len(theo_ceilings) >= 2

    # Every ceiling must document its source and assumptions
    for c in analyzer.ceilings:
        assert c.source_and_assumptions != ""
        assert len(c.source_and_assumptions) > 10


def test_regime_classification_and_report():
    analyzer = RooflineAnalyzer(architecture="arm64", cpu_model="Apple M4")

    # Reference N=256 (naive column stride, AI ~ 0.5) -> must be memory-bound
    p_naive = analyzer.analyze_candidate(
        workload_id="matmul",
        candidate_id="reference",
        size=256,
        median_ms=9.6,
        stddev_ms=0.2,
        measured_gflops=3.5,
    )
    assert p_naive.regime == "memory_bound"

    # NEON N=256 (AI = 256 / 6 = 42.67) -> must be compute-bound
    p_neon = analyzer.analyze_candidate(
        workload_id="matmul",
        candidate_id="neon",
        size=256,
        median_ms=0.83,
        stddev_ms=0.01,
        measured_gflops=40.4,
    )
    assert p_neon.regime == "compute_bound"

    report = analyzer.generate_report([p_naive, p_neon])
    ascii_out = analyzer.generate_ascii_plot(report)

    assert "EMPIRICAL ROOFLINE ANALYSIS" in ascii_out
    assert "memory_bound" in ascii_out
    assert "compute_bound" in ascii_out
    assert "MODEL LIMITATIONS" in ascii_out
