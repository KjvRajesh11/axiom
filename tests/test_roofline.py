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


def test_roofline_zero_and_invalid_inputs():
    analyzer = RooflineAnalyzer(architecture="arm64", cpu_model="Apple M4")

    # Size 0
    p_zero = analyzer.analyze_candidate("matmul", "reference", 0, 0.0, 0.0, 0.0)
    assert p_zero.regime == "invalid_input"
    assert p_zero.arithmetic_intensity == 0.0
    assert p_zero.working_set_bytes == 0
    assert "invalid" in p_zero.notes.lower()

    # Negative size
    p_neg = analyzer.analyze_candidate("matmul", "neon", -10, 1.0, 0.1, 10.0)
    assert p_neg.regime == "invalid_input"

    # Non-positive duration
    p_bad_time = analyzer.analyze_candidate("matmul", "neon", 64, 0.0, 0.0, 10.0)
    assert "invalid measurement duration" in p_bad_time.notes.lower()


def test_roofline_cache_tier_classification():
    analyzer = RooflineAnalyzer(architecture="arm64", cpu_model="Apple M4")

    # N=64: 3 * 64^2 * 4 = 49,152 bytes = 48 KB (L1 cache resident on Apple M4 with 128KB L1)
    p_l1 = analyzer.analyze_candidate("matmul", "neon", 64, 0.05, 0.001, 20.0)
    assert p_l1.cache_tier == "L1_resident"
    assert p_l1.bandwidth_tier == "L1_cache_bandwidth"

    # N=512: 3 * 512^2 * 4 = 3,145,728 bytes = 3 MB (L2/L3 cache resident, 16MB L2)
    p_l2 = analyzer.analyze_candidate("matmul", "neon", 512, 10.0, 0.1, 45.0)
    assert p_l2.cache_tier == "L2_L3_resident"
    assert p_l2.bandwidth_tier == "L2_cache_bandwidth"

    # N=2048: 3 * 2048^2 * 4 = 50,331,648 bytes = 48 MB (Spills into DRAM)
    p_dram = analyzer.analyze_candidate("matmul", "neon", 2048, 500.0, 5.0, 35.0)
    assert p_dram.cache_tier == "DRAM_resident"
    assert p_dram.bandwidth_tier == "DRAM_bandwidth"


def test_roofline_calibration():
    from axiom.benchmark import BenchmarkEngine
    analyzer = RooflineAnalyzer(architecture="arm64", cpu_model="Apple M4")
    be = BenchmarkEngine()
    calibrated = analyzer.calibrate_empirical_ceilings(be)
    assert len(calibrated) > 0
    # Observed single-core compute should be positive and realistic
    for k, v in calibrated.items():
        assert v > 0.0


def test_arithmetic_intensity_memory_sweep():
    # Sequential memory sweep: 64 KB = 65,536 bytes
    flops, bytes_cnt, ai = ArithmeticIntensityCalculator.calculate_memory_sweep(65536, "sequential")
    assert flops == 65536 / 4.0  # 16,384 additions
    assert bytes_cnt == 65536.0
    assert pytest.approx(ai, rel=1e-5) == 0.25

    # Strided memory sweep (128-byte cache-line traffic model)
    s_flops, s_bytes, s_ai = ArithmeticIntensityCalculator.calculate_memory_sweep(65536, "strided")
    assert s_flops == 16384.0
    assert s_bytes == 16384.0 * 128.0
    assert pytest.approx(s_ai, rel=1e-5) == 1.0 / 128.0

    # Zero or invalid sizes
    z_flops, z_bytes, z_ai = ArithmeticIntensityCalculator.calculate_memory_sweep(0, "sequential")
    assert (z_flops, z_bytes, z_ai) == (0.0, 0.0, 0.0)

    neg_flops, neg_bytes, neg_ai = ArithmeticIntensityCalculator.calculate_memory_sweep(-1024, "sequential")
    assert (neg_flops, neg_bytes, neg_ai) == (0.0, 0.0, 0.0)


def test_roofline_memory_workload_analysis():
    analyzer = RooflineAnalyzer(architecture="arm64", cpu_model="Apple M4")

    # Sequential traversal 1024 KB
    p_seq = analyzer.analyze_candidate(
        workload_id="memory",
        candidate_id="sequential",
        size=1024,
        median_ms=2.5,
        stddev_ms=0.05,
        measured_gflops=8.2,  # represents GB/s
    )
    assert p_seq.regime == "memory_bound"
    assert pytest.approx(p_seq.arithmetic_intensity, rel=1e-4) == 0.25
    assert p_seq.working_set_bytes == 1024 * 1024
    assert p_seq.cache_tier == "L2_L3_resident"
    assert "effective bandwidth" in p_seq.notes
    assert "GB/s" in p_seq.notes

    # Large memory sweep (32 MB) spilling into DRAM
    p_dram = analyzer.analyze_candidate(
        workload_id="memory",
        candidate_id="sequential",
        size=32768,
        median_ms=80.0,
        stddev_ms=1.5,
        measured_gflops=6.5,
    )
    assert p_dram.cache_tier == "DRAM_resident"
    assert p_dram.bandwidth_tier == "DRAM_bandwidth"


def test_roofline_unsupported_workload_rejection():
    analyzer = RooflineAnalyzer(architecture="arm64", cpu_model="Apple M4")
    with pytest.raises(ValueError, match="Unsupported workload 'unsupported_test_workload'"):
        analyzer.analyze_candidate("unsupported_test_workload", "reference", 64, 1.0, 0.1, 5.0)
