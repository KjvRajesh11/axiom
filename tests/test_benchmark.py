from axiom.benchmark import (
    BenchmarkEngine,
    welch_t_test,
    calculate_comprehensive_stats,
    StatisticalSummary
)


def test_calculate_comprehensive_stats_normal():
    data = [10.0, 10.5, 9.8, 10.2, 10.1, 9.9, 10.3, 10.0]
    stats = calculate_comprehensive_stats(data)

    assert stats.min_val == 9.8
    assert stats.max_val == 10.5
    assert 9.9 <= stats.median_val <= 10.2
    assert stats.p90_val >= stats.median_val
    assert stats.p95_val >= stats.p90_val
    assert stats.p99_val >= stats.p95_val
    assert stats.ci95_low <= stats.mean_val <= stats.ci95_high
    assert stats.cv > 0.0


def test_calculate_comprehensive_stats_edge_cases():
    # Empty list
    empty_stats = calculate_comprehensive_stats([])
    assert empty_stats.median_val == 0.0
    assert empty_stats.stddev_val == 0.0

    # Single observation
    single_stats = calculate_comprehensive_stats([42.0])
    assert single_stats.median_val == 42.0
    assert single_stats.stddev_val == 0.0
    assert single_stats.ci95_low == 42.0

    # All zeros
    zero_stats = calculate_comprehensive_stats([0.0, 0.0, 0.0])
    assert zero_stats.mean_val == 0.0
    assert zero_stats.cv == 0.0


def test_welch_t_test():
    sample_a = [10.0, 10.2, 10.1, 9.9, 10.0]
    sample_b = [1.0, 1.1, 1.0, 0.9, 1.0]

    t, p = welch_t_test(sample_a, sample_b)
    assert t > 0
    assert p < 0.001

    t2, p2 = welch_t_test(sample_a, sample_a)
    assert abs(t2) < 1e-6
    assert p2 == 1.0


def test_benchmark_engine_run():
    be = BenchmarkEngine()
    suite = be.run_workload(
        workload_id="matmul",
        candidate_ids=["reference", "interchanged", "neon"],
        sizes=[64],
        warmup_runs=1,
        measurement_runs=3
    )
    assert len(suite.comparisons) == 2
    for cmp in suite.comparisons:
        assert cmp.speedup > 1.0
        assert cmp.correctness_passed is True
        assert cmp.candidate_p95_ms > 0.0


def test_student_t_exactness_vs_normal_approximation():
    from axiom.benchmark import student_t_two_tailed_p_value

    # For t=2.776 at df=4, two-tailed p-value is exactly 0.05
    p_val = student_t_two_tailed_p_value(2.7764, df=4.0)
    assert abs(p_val - 0.05) < 0.005

    # At df=4, a t of 2.0 has p around 0.116; standard normal gives ~0.045
    p_t = student_t_two_tailed_p_value(2.0, df=4.0)
    assert p_t > 0.10  # Demonstrates no false precision from Gaussian tail


def test_tukey_outlier_detection_preserves_samples():
    # 7 consistent samples and 1 major context-switch spike (100.0 ms)
    times = [10.0, 10.1, 9.9, 10.0, 10.2, 10.0, 9.8, 100.0]
    stats = calculate_comprehensive_stats(times)

    assert stats.outlier_count == 1
    assert "Detected 1 outlier" in stats.outlier_explanation
    # Verify no samples were deleted
    assert len(times) == 8
    assert stats.max_val == 100.0


def test_sample_stddev_known_values():
    """Validates sample variance denominator N-1 on known mathematical dataset."""
    import math
    # Known dataset: [2, 4, 4, 4, 5, 5, 7, 9], N = 8, mean = 5.0
    # sq_diffs = (2-5)^2 + 3*(4-5)^2 + 2*(5-5)^2 + (7-5)^2 + (9-5)^2 = 9 + 3 + 0 + 4 + 16 = 32
    # Sample variance = 32 / (8 - 1) = 32 / 7
    # Sample stddev = sqrt(32 / 7) ~= 2.138089935
    data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    stats = calculate_comprehensive_stats(data)

    expected_mean = 5.0
    expected_sample_stddev = math.sqrt(32.0 / 7.0)

    assert abs(stats.mean_val - expected_mean) < 1e-6
    assert abs(stats.stddev_val - expected_sample_stddev) < 1e-6

    # 1 sample: stddev must be exactly 0.0
    single = calculate_comprehensive_stats([10.0])
    assert single.stddev_val == 0.0

    # 2 samples: [10.0, 12.0], mean = 11.0, diffs = [-1, 1], sq_diffs = 2, var = 2 / 1 = 2, std = sqrt(2)
    two_samples = calculate_comprehensive_stats([10.0, 12.0])
    assert abs(two_samples.stddev_val - math.sqrt(2.0)) < 1e-6


def test_cpp_and_python_statistical_agreement():
    """Validates that C++ benchmark runner and Python benchmark engine produce identical sample statistics."""
    import json
    import subprocess
    from pathlib import Path

    runner_bin = Path("build/axiom_benchmark_runner")
    assert runner_bin.exists()

    cmd = [
        str(runner_bin),
        "--workload", "matmul",
        "--candidate", "reference",
        "--size", "64",
        "--warmup", "1",
        "--runs", "7",
        "--format", "json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    assert res.returncode == 0

    cpp_data = json.loads(res.stdout.strip())
    raw_times = cpp_data["raw_times_ms"]
    assert len(raw_times) == 7

    py_stats = calculate_comprehensive_stats(raw_times)

    # Assert exact agreement on mean, median, sample stddev (N-1), and IQR
    assert abs(cpp_data["mean_ms"] - py_stats.mean_val) < 1e-4
    assert abs(cpp_data["median_ms"] - py_stats.median_val) < 1e-4
    assert abs(cpp_data["stddev_ms"] - py_stats.stddev_val) < 1e-4
    assert abs(cpp_data["iqr_ms"] - py_stats.iqr_val) < 1e-4
