"""
Benchmark Engine for Axiom 2.1.
Performs warm-up iterations, repeated measurement trials, statistical aggregation,
percentiles (p50, p90, p95, p99), 95% confidence intervals, and Welch's t-test.
Handles noisy, invalid, or inconclusive runs transparently.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from axiom.sandbox import ValidationSandbox, ExecutionResult
from axiom.workload import WorkloadRegistry, WorkloadDefinition


@dataclass
class StatisticalSummary:
    min_val: float
    max_val: float
    median_val: float
    mean_val: float
    stddev_val: float
    iqr_val: float
    p90_val: float
    p95_val: float
    p99_val: float
    ci95_low: float
    ci95_high: float
    cv: float  # Coefficient of Variation (stddev / mean)
    outlier_count: int = 0
    outlier_bounds: Tuple[float, float] = (0.0, 0.0)
    outlier_explanation: str = ""


def _pure_python_betacf(a: float, b: float, x: float, max_iter: int = 100, eps: float = 1e-12) -> float:
    """Evaluates continued fraction for incomplete beta function."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        del_h = d * c
        h *= del_h
        if abs(del_h - 1.0) < eps:
            break
    return h


def _pure_python_ibeta(a: float, b: float, x: float) -> float:
    """Computes regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _pure_python_betacf(a, b, x)
    else:
        return 1.0 - (math.exp(math.log(1.0 - x) * b + math.log(x) * a - lbeta) / b) * _pure_python_betacf(b, a, 1.0 - x)


def student_t_two_tailed_p_value(t: float, df: float) -> float:
    """Computes exact two-tailed Student-t p-value with df degrees of freedom."""
    if df <= 0.0:
        return 1.0
    try:
        import scipy.stats  # type: ignore
        return float(scipy.stats.t.sf(abs(t), df=df) * 2.0)
    except Exception:
        # Exact numerical fallback using regularized incomplete beta function
        x = df / (df + t * t)
        p = _pure_python_ibeta(0.5 * df, 0.5, x)
        return max(0.0, min(1.0, float(p)))


def calculate_comprehensive_stats(times_ms: List[float]) -> StatisticalSummary:
    """Calculates comprehensive robust descriptive statistics, outlier fences, and confidence intervals."""
    if not times_ms:
        return StatisticalSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    n = len(times_ms)
    sorted_times = sorted(times_ms)

    min_v = sorted_times[0]
    max_v = sorted_times[-1]

    # Median (p50)
    if n % 2 == 0:
        median_v = 0.5 * (sorted_times[n // 2 - 1] + sorted_times[n // 2])
    else:
        median_v = sorted_times[n // 2]

    # Mean
    mean_v = sum(sorted_times) / n

    # Variance and StdDev
    if n > 1:
        sq_diffs = sum((x - mean_v) ** 2 for x in sorted_times)
        stddev_v = math.sqrt(sq_diffs / (n - 1))
    else:
        stddev_v = 0.0

    # IQR = Q3 - Q1
    q1 = sorted_times[n // 4]
    q3 = sorted_times[(3 * n) // 4]
    iqr_v = q3 - q1

    # Tukey's Outlier Fences: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    lower_fence = q1 - 1.5 * iqr_v
    upper_fence = q3 + 1.5 * iqr_v
    outliers = [x for x in times_ms if x < lower_fence or x > upper_fence]
    outlier_count = len(outliers)
    if outlier_count > 0:
        outlier_explanation = (
            f"Detected {outlier_count} outlier sample(s) outside Tukey fence [{lower_fence:.3f}, {upper_fence:.3f}] ms. "
            f"Samples preserved without deletion to maintain scientific validity."
        )
    else:
        outlier_explanation = "Zero outliers detected outside Tukey fences."

    # Percentiles helper
    def get_percentile(p: float) -> float:
        idx = int(p * (n - 1))
        return sorted_times[min(idx, n - 1)]

    p90 = get_percentile(0.90)
    p95 = get_percentile(0.95)
    p99 = get_percentile(0.99)

    # 95% Confidence Interval for the mean: mean +/- 1.96 * (std / sqrt(n))
    margin = (1.96 * stddev_v / math.sqrt(n)) if n > 1 else 0.0
    ci_low = max(0.0, mean_v - margin)
    ci_high = mean_v + margin

    cv = (stddev_v / mean_v) if mean_v > 0 else 0.0

    return StatisticalSummary(
        min_val=min_v,
        max_val=max_v,
        median_val=median_v,
        mean_val=mean_v,
        stddev_val=stddev_v,
        iqr_val=iqr_v,
        p90_val=p90,
        p95_val=p95,
        p99_val=p99,
        ci95_low=ci_low,
        ci95_high=ci_high,
        cv=cv,
        outlier_count=outlier_count,
        outlier_bounds=(lower_fence, upper_fence),
        outlier_explanation=outlier_explanation
    )


def welch_t_test(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Computes Welch's t-statistic and exact Student-t two-tailed p-value."""
    n1 = len(x)
    n2 = len(y)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0

    m1 = sum(x) / n1
    m2 = sum(y) / n2

    v1 = sum((v - m1) ** 2 for v in x) / (n1 - 1)
    v2 = sum((v - m2) ** 2 for v in y) / (n2 - 1)

    denom = math.sqrt((v1 / n1) + (v2 / n2))
    if denom == 0.0:
        return 0.0, 1.0

    t = (m1 - m2) / denom

    # Welch-Satterthwaite degrees of freedom approximation
    df_num = ((v1 / n1) + (v2 / n2)) ** 2
    df_denom = ((v1 / n1) ** 2 / (n1 - 1)) + ((v2 / n2) ** 2 / (n2 - 1))
    df = df_num / df_denom if df_denom > 0 else 1.0

    p_value = student_t_two_tailed_p_value(t, df)
    return t, p_value


@dataclass
class CandidateComparison:
    workload: str
    size: int
    candidate_id: str
    baseline_id: str
    baseline_median_ms: float
    candidate_median_ms: float
    speedup: float  # baseline_ms / candidate_ms
    relative_reduction_pct: float  # (baseline - candidate) / baseline * 100
    baseline_cv: float  # stddev / mean
    candidate_cv: float  # stddev / mean
    baseline_ci95: Tuple[float, float]
    candidate_ci95: Tuple[float, float]
    candidate_p95_ms: float
    candidate_p99_ms: float
    is_statistically_significant: bool
    p_value: Optional[float]
    is_noisy: bool
    correctness_passed: bool
    baseline_gflops: float
    candidate_gflops: float
    notes: str = ""
    speedup_vs_best_single: Optional[float] = None
    best_single_id: Optional[str] = None


@dataclass
class BenchmarkSuiteResult:
    workload_id: str
    sizes: List[int]
    results_by_candidate: Dict[str, Dict[int, ExecutionResult]]
    comparisons: List[CandidateComparison]


class BenchmarkEngine:
    def __init__(self, sandbox: Optional[ValidationSandbox] = None, registry: Optional[WorkloadRegistry] = None):
        self.sandbox = sandbox or ValidationSandbox()
        self.registry = registry or WorkloadRegistry()

    def run_workload(
        self,
        workload_id: str,
        candidate_ids: Optional[List[str]] = None,
        sizes: Optional[List[int]] = None,
        warmup_runs: int = 3,
        measurement_runs: int = 15
    ) -> BenchmarkSuiteResult:
        wl = self.registry.get(workload_id)
        if not wl:
            raise ValueError(f"Unknown workload: {workload_id}")

        all_candidates = [c.candidate_id for c in wl.candidates]
        targets = candidate_ids if candidate_ids else all_candidates
        target_sizes = sizes if sizes else [wl.default_size]

        results_by_candidate: Dict[str, Dict[int, ExecutionResult]] = {c: {} for c in targets}
        comparisons: List[CandidateComparison] = []

        baseline_id = "reference"
        if baseline_id not in all_candidates:
            baseline_id = all_candidates[0] if all_candidates else "reference"

        if baseline_id not in targets:
            targets = [baseline_id] + targets

        for size in target_sizes:
            # Run baseline
            base_res = self.sandbox.run_candidate(
                workload=workload_id,
                candidate=baseline_id,
                size=size,
                warmup_runs=warmup_runs,
                measurement_runs=measurement_runs,
                timeout_seconds=wl.timeout_seconds
            )
            results_by_candidate.setdefault(baseline_id, {})[size] = base_res
            base_stats = calculate_comprehensive_stats(base_res.raw_times_ms)

            # Find best single-threaded candidate for this size
            best_single_id = None
            best_single_ms = float("inf")
            for cid in targets:
                if cid == "multithreaded":
                    continue
                c_res = results_by_candidate.get(cid, {}).get(size)
                if c_res and c_res.correctness_passed and 0 < c_res.median_ms < best_single_ms:
                    best_single_ms = c_res.median_ms
                    best_single_id = cid

            for cand_id in targets:
                if cand_id == baseline_id:
                    continue
                cand_res = self.sandbox.run_candidate(
                    workload=workload_id,
                    candidate=cand_id,
                    size=size,
                    warmup_runs=warmup_runs,
                    measurement_runs=measurement_runs,
                    timeout_seconds=wl.timeout_seconds
                )
                results_by_candidate.setdefault(cand_id, {})[size] = cand_res
                cand_stats = calculate_comprehensive_stats(cand_res.raw_times_ms)

                is_noisy = (base_stats.cv > 0.15) or (cand_stats.cv > 0.15)

                speedup = 0.0
                reduction_pct = 0.0
                if cand_res.median_ms > 0 and base_res.median_ms > 0:
                    speedup = base_res.median_ms / cand_res.median_ms
                    reduction_pct = ((base_res.median_ms - cand_res.median_ms) / base_res.median_ms) * 100.0

                speedup_vs_single = None
                if best_single_ms < float("inf") and cand_res.median_ms > 0:
                    speedup_vs_single = best_single_ms / cand_res.median_ms

                # Statistical significance
                t_stat, p_val = welch_t_test(base_res.raw_times_ms, cand_res.raw_times_ms)
                is_significant = (p_val < 0.01)

                note = ""
                if not cand_res.correctness_passed:
                    note = f"FAILED: Correctness violation (max rel error: {cand_res.max_rel_error:.2e})."
                elif cand_id == "multithreaded" and speedup_vs_single is not None:
                    note = f"Parallel speedup vs best single-threaded ({best_single_id}): {speedup_vs_single:.2f}x (vs naive baseline: {speedup:.2f}x)."
                elif speedup < 0.90 and is_significant:
                    note = f"REGRESSION: Candidate is {1.0/speedup:.2f}x slower than baseline (p < 0.01)."
                elif is_noisy:
                    note = f"WARNING: High measurement variance (Baseline CV={base_stats.cv*100:.1f}%, Candidate CV={cand_stats.cv*100:.1f}%)."
                elif is_significant:
                    note = f"Statistically significant difference (p < {p_val:.4f})."
                else:
                    note = f"Inconclusive statistical separation (p = {p_val:.4f})."

                comparisons.append(CandidateComparison(
                    workload=workload_id,
                    size=size,
                    candidate_id=cand_id,
                    baseline_id=baseline_id,
                    baseline_median_ms=base_res.median_ms,
                    candidate_median_ms=cand_res.median_ms,
                    speedup=speedup,
                    relative_reduction_pct=reduction_pct,
                    baseline_cv=base_stats.cv,
                    candidate_cv=cand_stats.cv,
                    baseline_ci95=(base_stats.ci95_low, base_stats.ci95_high),
                    candidate_ci95=(cand_stats.ci95_low, cand_stats.ci95_high),
                    candidate_p95_ms=cand_stats.p95_val,
                    candidate_p99_ms=cand_stats.p99_val,
                    is_statistically_significant=is_significant,
                    p_value=p_val,
                    is_noisy=is_noisy,
                    correctness_passed=cand_res.correctness_passed,
                    baseline_gflops=base_res.gflops,
                    candidate_gflops=cand_res.gflops,
                    notes=note,
                    speedup_vs_best_single=speedup_vs_single,
                    best_single_id=best_single_id
                ))

        return BenchmarkSuiteResult(
            workload_id=workload_id,
            sizes=target_sizes,
            results_by_candidate=results_by_candidate,
            comparisons=comparisons
        )
