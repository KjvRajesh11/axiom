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
