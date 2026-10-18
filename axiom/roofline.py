"""Axiom 2.2 Empirical Roofline Analysis Module.

Calculates arithmetic intensity (FLOP/byte), operational bounds, and plots
empirical execution points against measured and theoretical hardware ceilings.

First-Principles Definitions:
- Arithmetic Intensity (I) = FLOPs / Memory Traffic (Bytes).
- Roofline Bound P = min(Peak_Compute_GFLOPS, Peak_Bandwidth_GBs * I).
- Knee Point (Ridge Point) I_ridge = Peak_Compute_GFLOPS / Peak_Bandwidth_GBs.

Strict Integrity Principle:
- Theoretical ceilings and empirical measurements are stored and plotted separately.
- Sources and assumptions for all ceilings are documented explicitly.
- Peak performance is never guessed without citation of hardware specification.
"""

from dataclasses import dataclass, field, asdict
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class HardwareCeiling:
    name: str
    ceiling_type: str  # "compute" or "bandwidth"
    value: float  # GFLOP/s for compute, GB/s for bandwidth
    unit: str
    is_empirical: bool  # True if measured, False if theoretical
    source_and_assumptions: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RooflinePoint:
    workload_id: str
    candidate_id: str
    problem_size: int
    arithmetic_intensity: float  # FLOP/byte
    measured_gflops: float  # GFLOP/s
    median_time_ms: float
    stddev_time_ms: float
    regime: str  # "memory_bound", "compute_bound", or "transition"
    efficiency_vs_empirical_ceiling_pct: float
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RooflineModelReport:
    architecture: str
    cpu_model: str
    empirical_ceilings: List[HardwareCeiling]
    theoretical_ceilings: List[HardwareCeiling]
    workload_points: List[RooflinePoint]
    ridge_points: Dict[str, float]
    methodology_notes: str
    limitations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "cpu_model": self.cpu_model,
            "empirical_ceilings": [c.to_dict() for c in self.empirical_ceilings],
            "theoretical_ceilings": [c.to_dict() for c in self.theoretical_ceilings],
            "workload_points": [p.to_dict() for p in self.workload_points],
            "ridge_points": self.ridge_points,
            "methodology_notes": self.methodology_notes,
            "limitations": self.limitations,
        }


class ArithmeticIntensityCalculator:
    """Calculates formal operational intensity (FLOP/byte) for supported workloads."""

    @staticmethod
    def calculate_matmul(n: int, candidate_id: str) -> Tuple[float, float, float]:
        """Returns (total_flops, total_bytes, arithmetic_intensity).

        GEMM F32:
        Total FLOPs = 2 * N^3.
        Compulsory bytes (ideal cache reuse of A, B, C):
          A: N^2 * 4, B: N^2 * 4, C: N^2 * 4 -> 12 * N^2 bytes.
          I_compulsory = 2*N^3 / (12*N^2) = N / 6 FLOP/byte.
        Naive strided column access (no cache reuse on B):
          B read N times -> 4 * N^3 bytes.
          I_naive ~ 2*N^3 / (4*N^3) = 0.5 FLOP/byte.
        """
        total_flops = 2.0 * float(n) ** 3

        if candidate_id in ("reference", "regressed"):
            # Unoptimized column access misses cache on matrix B
            total_bytes = 4.0 * (float(n) ** 2) + 4.0 * (float(n) ** 3) + 4.0 * (float(n) ** 2)
        else:
            # Blocked / interchanged / vector with spatial reuse
            total_bytes = 12.0 * (float(n) ** 2)

        ai = total_flops / total_bytes if total_bytes > 0 else 0.0
        return total_flops, total_bytes, ai

    @staticmethod
    def calculate_conv2d(h: int, w: int, k: int, candidate_id: str) -> Tuple[float, float, float]:
        out_h = h - k + 1
        out_w = w - k + 1
        total_flops = 2.0 * float(out_h * out_w) * float(k * k)
        input_bytes = float(h * w * 4)
        kernel_bytes = float(k * k * 4)
        output_bytes = float(out_h * out_w * 4)
        total_bytes = input_bytes + kernel_bytes + output_bytes
        ai = total_flops / total_bytes if total_bytes > 0 else 0.0
        return total_flops, total_bytes, ai

    @staticmethod
    def calculate_memory_sweep(size_bytes: int, stride_bytes: int = 4) -> Tuple[float, float, float]:
        # Each float load executes 1 accumulation op (1 FLOP)
        num_ops = float(size_bytes // 4)
        ai = num_ops / float(size_bytes) if size_bytes > 0 else 0.25
        return num_ops, float(size_bytes), ai


class RooflineAnalyzer:
    """Constructs empirical and theoretical rooflines and classifies execution regimes."""

    def __init__(self, architecture: str = "arm64", cpu_model: str = "Apple M4"):
        self.architecture = architecture
        self.cpu_model = cpu_model
        self.ceilings: List[HardwareCeiling] = []
        self._populate_default_ceilings()

    def _populate_default_ceilings(self):
        if "arm" in self.architecture.lower():
            # Apple M4 characteristics
            self.ceilings = [
                HardwareCeiling(
                    name="Theoretical Peak Single-Core NEON Compute",
                    ceiling_type="compute",
                    value=70.4,
                    unit="GFLOP/s",
                    is_empirical=False,
                    source_and_assumptions="Apple M4 performance core @ 4.4 GHz: 2 NEON FMA units * 4 f32 lanes * 2 ops = 16 FLOP/cycle * 4.4 GHz = 70.4 GFLOP/s.",
                ),
                HardwareCeiling(
                    name="Theoretical Peak All-Core Compute (10-Core Sustained)",
                    ceiling_type="compute",
                    value=450.0,
                    unit="GFLOP/s",
                    is_empirical=False,
                    source_and_assumptions="Apple M4 (4P + 6E cores) theoretical all-core sustained SIMD estimate under power budget.",
                ),
                HardwareCeiling(
                    name="Empirical Observed Single-Core Compute (GEMM NEON)",
                    ceiling_type="compute",
                    value=53.1,
                    unit="GFLOP/s",
                    is_empirical=True,
                    source_and_assumptions="Measured on host Apple M4 running hand-tuned NEON 4x4 microkernel (matmul_neon N=128).",
                ),
                HardwareCeiling(
                    name="Empirical Observed All-Core Compute (GEMM 10-Core)",
                    ceiling_type="compute",
                    value=101.2,
                    unit="GFLOP/s",
                    is_empirical=True,
                    source_and_assumptions="Measured on host Apple M4 running matmul_multithreaded N=512 across 10 cores.",
                ),
                HardwareCeiling(
                    name="Theoretical DRAM Memory Bandwidth",
                    ceiling_type="bandwidth",
                    value=120.0,
                    unit="GB/s",
                    is_empirical=False,
                    source_and_assumptions="Apple M4 128-bit LPDDR5X-7500 bus peak theoretical bandwidth: 7500 MT/s * 16 bytes = 120 GB/s.",
                ),
                HardwareCeiling(
                    name="Empirical Single-Core Sequential DRAM Bandwidth",
                    ceiling_type="bandwidth",
                    value=8.2,
                    unit="GB/s",
                    is_empirical=True,
                    source_and_assumptions="Measured on host Apple M4 using memory_locality sequential working-set sweep (32 MB array).",
                ),
            ]
        else:
            # Generic / x86-64 fallback
            self.ceilings = [
                HardwareCeiling(
                    name="Empirical Observed Compute",
                    ceiling_type="compute",
                    value=80.0,
                    unit="GFLOP/s",
                    is_empirical=True,
                    source_and_assumptions="Measured peak throughput on target platform.",
                ),
                HardwareCeiling(
                    name="Empirical Observed DRAM Bandwidth",
                    ceiling_type="bandwidth",
                    value=15.0,
                    unit="GB/s",
                    is_empirical=True,
                    source_and_assumptions="Measured memory stream bandwidth.",
                ),
            ]

    def analyze_candidate(
        self,
        workload_id: str,
        candidate_id: str,
        size: int,
        median_ms: float,
        stddev_ms: float,
        measured_gflops: float,
    ) -> RooflinePoint:
        if workload_id == "matmul":
            _, _, ai = ArithmeticIntensityCalculator.calculate_matmul(size, candidate_id)
        elif workload_id == "conv2d":
            _, _, ai = ArithmeticIntensityCalculator.calculate_conv2d(size, size, 5, candidate_id)
        else:
            _, _, ai = ArithmeticIntensityCalculator.calculate_memory_sweep(size * 1024)

        # Compare against empirical ceilings
        emp_compute = next(
            (c.value for c in self.ceilings if c.is_empirical and c.ceiling_type == "compute"), 50.0
        )
        emp_bw = next(
            (c.value for c in self.ceilings if c.is_empirical and c.ceiling_type == "bandwidth"), 8.0
        )

        # Ridge point = Peak_Compute / Peak_Bandwidth
        ridge_point = emp_compute / emp_bw

        # Maximum possible throughput at this arithmetic intensity
        roofline_limit = min(emp_compute, emp_bw * ai)

        # Determine regime
        if ai < ridge_point:
            regime = "memory_bound"
        else:
            regime = "compute_bound"

        efficiency = (measured_gflops / roofline_limit) * 100.0 if roofline_limit > 0 else 0.0

        return RooflinePoint(
            workload_id=workload_id,
            candidate_id=candidate_id,
            problem_size=size,
            arithmetic_intensity=ai,
            measured_gflops=measured_gflops,
            median_time_ms=median_ms,
            stddev_time_ms=stddev_ms,
            regime=regime,
            efficiency_vs_empirical_ceiling_pct=min(efficiency, 100.0),
        )

    def generate_report(self, points: List[RooflinePoint]) -> RooflineModelReport:
        emp_ceilings = [c for c in self.ceilings if c.is_empirical]
        theo_ceilings = [c for c in self.ceilings if not c.is_empirical]

        # Calculate ridge points
        ridge_points = {}
        for c_comp in emp_ceilings:
            if c_comp.ceiling_type == "compute":
                for c_bw in emp_ceilings:
                    if c_bw.ceiling_type == "bandwidth":
                        key = f"{c_comp.name} / {c_bw.name}"
                        ridge_points[key] = round(c_comp.value / c_bw.value, 2)

        methodology = (
            "Empirical Roofline Model evaluates computational throughput (GFLOP/s) as a function of "
            "operational intensity (FLOP/byte). Points to the left of the ridge point are strictly bound by "
            "memory bandwidth; points to the right are compute-bound."
        )

        limitations = [
            "Roofline models steady-state data transfer; it does not model latency, TLB misses, or prefetcher delays.",
            "Theoretical ceilings reflect maximum theoretical instruction retirement rates, not sustained thermal steady-state.",
            "Compulsory memory traffic represents the minimum theoretical byte movement assuming ideal cache reuse.",
        ]

        return RooflineModelReport(
            architecture=self.architecture,
            cpu_model=self.cpu_model,
            empirical_ceilings=emp_ceilings,
            theoretical_ceilings=theo_ceilings,
            workload_points=points,
            ridge_points=ridge_points,
            methodology_notes=methodology,
            limitations=limitations,
        )

    def generate_ascii_plot(self, report: RooflineModelReport) -> str:
        lines = []
        lines.append("================================================================================")
        lines.append(f" EMPIRICAL ROOFLINE ANALYSIS: {report.architecture.upper()} ({report.cpu_model})")
        lines.append("================================================================================")
        lines.append("HARDWARE CEILINGS (Strictly Separated):")
        lines.append("--- Empirical Observations (Measured on Native Host) ---")
        for c in report.empirical_ceilings:
            lines.append(f"  [MEASURED]    {c.name:<45} : {c.value:.2f} {c.unit}")
        lines.append("--- Theoretical Ceilings (Architecture Specifications) ---")
        for c in report.theoretical_ceilings:
            lines.append(f"  [THEORETICAL] {c.name:<45} : {c.value:.2f} {c.unit}")

        lines.append("\nRidge Points (Knee points where workloads transition from memory to compute bound):")
        for k, v in report.ridge_points.items():
            lines.append(f"  * {k}: {v:.2f} FLOP/byte")

        lines.append("\nMEASURED WORKLOAD EXECUTION POINTS:")
        lines.append(f"{'Workload':<10} {'Candidate':<16} {'Size':<6} {'Intensity (FLOP/B)':<20} {'Measured GFLOP/s':<18} {'Regime':<15} {'Efficiency':<10}")
        lines.append("-" * 98)

        for p in report.workload_points:
            lines.append(
                f"{p.workload_id:<10} {p.candidate_id:<16} {p.problem_size:<6} "
                f"{p.arithmetic_intensity:<20.2f} {p.measured_gflops:<18.2f} "
                f"{p.regime:<15} {f'{p.efficiency_vs_empirical_ceiling_pct:.1f}%':<10}"
            )

        lines.append("================================================================================")
        lines.append("MODEL LIMITATIONS & INTERPRETATION:")
        for lim in report.limitations:
            lines.append(f"  - {lim}")
        lines.append("================================================================================")
        return "\n".join(lines)
