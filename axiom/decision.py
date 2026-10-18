"""
Decision Engine for Axiom 2.0.
Implements evidence-based performance engineering acceptance criteria.
No candidate is accepted based on theoretical claims or LLM assertions.
Strictly requires correctness verification, low measurement noise, and statistical significance.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
from axiom.benchmark import CandidateComparison


class DecisionVerdict(str, Enum):
    RECOMMENDED = "RECOMMENDED"
    REJECTED_CORRECTNESS = "REJECTED_CORRECTNESS"
    REJECTED_REGRESSION = "REJECTED_REGRESSION"
    INCONCLUSIVE_NOISY = "INCONCLUSIVE_NOISY"
    INCONCLUSIVE_NEGLIGIBLE = "INCONCLUSIVE_NEGLIGIBLE"


@dataclass
class DecisionCriteria:
    min_speedup: float = 1.05  # Minimum 5% speedup required
    max_acceptable_cv: float = 0.15  # Maximum 15% coefficient of variation
    max_p_value: float = 0.01  # 99% confidence requirement
    require_correctness: bool = True


@dataclass
class EvaluationDecision:
    workload: str
    size: int
    candidate_id: str
    verdict: DecisionVerdict
    speedup: float
    relative_reduction_pct: float
    is_correct: bool
    variability_ok: bool
    statistically_significant: bool
    rationale: str
    supporting_metrics: Dict[str, Any]
    speedup_vs_best_single: Optional[float] = None
    best_single_id: Optional[str] = None
    uncertainty_explanation: str = ""
    compiler_evidence: Optional[str] = None


class DecisionEngine:
    def __init__(self, criteria: Optional[DecisionCriteria] = None):
        self.criteria = criteria or DecisionCriteria()

    def evaluate(self, comparison: CandidateComparison) -> EvaluationDecision:
        metrics = {
            "baseline_median_ms": comparison.baseline_median_ms,
            "candidate_median_ms": comparison.candidate_median_ms,
            "speedup": comparison.speedup,
            "speedup_vs_best_single": comparison.speedup_vs_best_single,
            "best_single_id": comparison.best_single_id,
            "baseline_cv": comparison.baseline_cv,
            "candidate_cv": comparison.candidate_cv,
            "p_value": comparison.p_value,
            "gflops_gain": comparison.candidate_gflops - comparison.baseline_gflops
        }

        uncertainty = (
            f"Measurement variability: baseline CV={comparison.baseline_cv*100:.2f}%, candidate CV={comparison.candidate_cv*100:.2f}%. "
            f"Welch's t-test p-value={comparison.p_value:.6f} "
            f"({'statistically significant at alpha=0.01' if comparison.is_statistically_significant else 'insufficient statistical separation'})."
        )
        if comparison.speedup_vs_best_single is not None:
            uncertainty += f" Parallel speedup vs best single-threaded ({comparison.best_single_id}) is {comparison.speedup_vs_best_single:.2f}x."

        # 1. Correctness check
        if not comparison.correctness_passed:
            return EvaluationDecision(
                workload=comparison.workload,
                size=comparison.size,
                candidate_id=comparison.candidate_id,
                verdict=DecisionVerdict.REJECTED_CORRECTNESS,
                speedup=comparison.speedup,
                speedup_vs_best_single=comparison.speedup_vs_best_single,
                best_single_id=comparison.best_single_id,
                relative_reduction_pct=comparison.relative_reduction_pct,
                is_correct=False,
                variability_ok=True,
                statistically_significant=comparison.is_statistically_significant,
                rationale="Candidate produced numerical outputs exceeding allowed error tolerance relative to baseline reference.",
                supporting_metrics=metrics,
                uncertainty_explanation=uncertainty
            )

        # 2. Measurement noise check
        variability_ok = (comparison.baseline_cv <= self.criteria.max_acceptable_cv) and (
            comparison.candidate_cv <= self.criteria.max_acceptable_cv
        )
        if not variability_ok:
            return EvaluationDecision(
                workload=comparison.workload,
                size=comparison.size,
                candidate_id=comparison.candidate_id,
                verdict=DecisionVerdict.INCONCLUSIVE_NOISY,
                speedup=comparison.speedup,
                speedup_vs_best_single=comparison.speedup_vs_best_single,
                best_single_id=comparison.best_single_id,
                relative_reduction_pct=comparison.relative_reduction_pct,
                is_correct=True,
                variability_ok=False,
                statistically_significant=comparison.is_statistically_significant,
                rationale=(
                    f"Measurement noise is too high to accept or reject reliably "
                    f"(Baseline CV={comparison.baseline_cv*100:.1f}%, Candidate CV={comparison.candidate_cv*100:.1f}%). "
                    f"Rerun with more measurement trials or on an isolated CPU core."
                ),
                supporting_metrics=metrics,
                uncertainty_explanation=uncertainty
            )

        # 3. Performance regression check
        if comparison.speedup < 0.97 and comparison.is_statistically_significant:
            return EvaluationDecision(
                workload=comparison.workload,
                size=comparison.size,
                candidate_id=comparison.candidate_id,
                verdict=DecisionVerdict.REJECTED_REGRESSION,
                speedup=comparison.speedup,
                speedup_vs_best_single=comparison.speedup_vs_best_single,
                best_single_id=comparison.best_single_id,
                relative_reduction_pct=comparison.relative_reduction_pct,
                is_correct=True,
                variability_ok=True,
                statistically_significant=True,
                rationale=(
                    f"Candidate exhibited statistically significant performance regression: "
                    f"{1.0/comparison.speedup:.2f}x slower than reference baseline."
                ),
                supporting_metrics=metrics,
                uncertainty_explanation=uncertainty
            )

        # 4. Statistically significant speedup check
        if comparison.speedup >= self.criteria.min_speedup and comparison.is_statistically_significant:
            parallel_note = f" ({comparison.speedup_vs_best_single:.2f}x vs best 1-core {comparison.best_single_id})" if comparison.speedup_vs_best_single is not None else ""
            return EvaluationDecision(
                workload=comparison.workload,
                size=comparison.size,
                candidate_id=comparison.candidate_id,
                verdict=DecisionVerdict.RECOMMENDED,
                speedup=comparison.speedup,
                speedup_vs_best_single=comparison.speedup_vs_best_single,
                best_single_id=comparison.best_single_id,
                relative_reduction_pct=comparison.relative_reduction_pct,
                is_correct=True,
                variability_ok=True,
                statistically_significant=True,
                rationale=(
                    f"Candidate passed all correctness checks and delivered a statistically significant "
                    f"{comparison.speedup:.2f}x speedup vs baseline{parallel_note} "
                    f"({comparison.relative_reduction_pct:.1f}% latency reduction, p < 0.01) "
                    f"under controlled variability."
                ),
                supporting_metrics=metrics,
                uncertainty_explanation=uncertainty
            )

        # 5. Marginal / Inconclusive difference
        return EvaluationDecision(
            workload=comparison.workload,
            size=comparison.size,
            candidate_id=comparison.candidate_id,
            verdict=DecisionVerdict.INCONCLUSIVE_NEGLIGIBLE,
            speedup=comparison.speedup,
            speedup_vs_best_single=comparison.speedup_vs_best_single,
            best_single_id=comparison.best_single_id,
            relative_reduction_pct=comparison.relative_reduction_pct,
            is_correct=True,
            variability_ok=True,
            statistically_significant=comparison.is_statistically_significant,
            rationale=(
                f"Candidate did not demonstrate a sufficient or statistically distinguishable speedup "
                f"({comparison.speedup:.2f}x observed vs {self.criteria.min_speedup:.2f}x required threshold)."
            ),
            supporting_metrics=metrics,
            uncertainty_explanation=uncertainty
        )

    def select_best_candidate(self, decisions: List[EvaluationDecision]) -> Optional[EvaluationDecision]:
        """Selects the highest performing recommended candidate, or None if no candidate passes criteria."""
        recommended = [d for d in decisions if d.verdict == DecisionVerdict.RECOMMENDED]
        if not recommended:
            return None
        # Sort by highest speedup
        recommended.sort(key=lambda d: d.speedup, reverse=True)
        return recommended[0]
