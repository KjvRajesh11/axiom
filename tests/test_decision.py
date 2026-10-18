from axiom.benchmark import CandidateComparison
from axiom.decision import DecisionEngine, DecisionCriteria, DecisionVerdict


def test_decision_engine_rules():
    engine = DecisionEngine(DecisionCriteria(min_speedup=1.05, max_acceptable_cv=0.15))

    # Case 1: Fast, correct, statistically significant -> RECOMMENDED
    cmp_good = CandidateComparison(
        workload="matmul",
        size=256,
        candidate_id="interchanged",
        baseline_id="reference",
        baseline_median_ms=10.0,
        candidate_median_ms=1.0,
        speedup=10.0,
        relative_reduction_pct=90.0,
        baseline_cv=0.02,
        candidate_cv=0.02,
        baseline_ci95=(9.8, 10.2),
        candidate_ci95=(0.98, 1.02),
        candidate_p95_ms=1.05,
        candidate_p99_ms=1.08,
        is_statistically_significant=True,
        p_value=0.0001,
        is_noisy=False,
        correctness_passed=True,
        baseline_gflops=3.0,
        candidate_gflops=30.0
    )
    dec_good = engine.evaluate(cmp_good)
    assert dec_good.verdict == DecisionVerdict.RECOMMENDED

    # Case 2: Failed correctness -> REJECTED_CORRECTNESS
    cmp_bad_correctness = CandidateComparison(
        workload="matmul",
        size=256,
        candidate_id="buggy",
        baseline_id="reference",
        baseline_median_ms=10.0,
        candidate_median_ms=0.5,
        speedup=20.0,
        relative_reduction_pct=95.0,
        baseline_cv=0.02,
        candidate_cv=0.02,
        baseline_ci95=(9.8, 10.2),
        candidate_ci95=(0.48, 0.52),
        candidate_p95_ms=0.53,
        candidate_p99_ms=0.55,
        is_statistically_significant=True,
        p_value=0.0001,
        is_noisy=False,
        correctness_passed=False,
        baseline_gflops=3.0,
        candidate_gflops=60.0
    )
    dec_bad = engine.evaluate(cmp_bad_correctness)
    assert dec_bad.verdict == DecisionVerdict.REJECTED_CORRECTNESS

    # Case 3: Slower than baseline -> REJECTED_REGRESSION
    cmp_slow = CandidateComparison(
        workload="matmul",
        size=256,
        candidate_id="slow",
        baseline_id="reference",
        baseline_median_ms=10.0,
        candidate_median_ms=15.0,
        speedup=0.66,
        relative_reduction_pct=-50.0,
        baseline_cv=0.02,
        candidate_cv=0.02,
        baseline_ci95=(9.8, 10.2),
        candidate_ci95=(14.8, 15.2),
        candidate_p95_ms=15.3,
        candidate_p99_ms=15.5,
        is_statistically_significant=True,
        p_value=0.0001,
        is_noisy=False,
        correctness_passed=True,
        baseline_gflops=3.0,
        candidate_gflops=2.0
    )
    dec_slow = engine.evaluate(cmp_slow)
    assert dec_slow.verdict == DecisionVerdict.REJECTED_REGRESSION

    # Case 4: High variance -> INCONCLUSIVE_NOISY
    cmp_noisy = CandidateComparison(
        workload="matmul",
        size=256,
        candidate_id="noisy",
        baseline_id="reference",
        baseline_median_ms=10.0,
        candidate_median_ms=8.0,
        speedup=1.25,
        relative_reduction_pct=20.0,
        baseline_cv=0.25,  # Exceeds 15% threshold
        candidate_cv=0.02,
        baseline_ci95=(9.0, 11.0),
        candidate_ci95=(7.8, 8.2),
        candidate_p95_ms=8.4,
        candidate_p99_ms=8.6,
        is_statistically_significant=False,
        p_value=0.1,
        is_noisy=True,
        correctness_passed=True,
        baseline_gflops=3.0,
        candidate_gflops=3.75
    )
    dec_noisy = engine.evaluate(cmp_noisy)
    assert dec_noisy.verdict == DecisionVerdict.INCONCLUSIVE_NOISY
