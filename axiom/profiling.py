"""
Profiling and Static Analysis Adapter for Axiom 2.1.
Integrates native profiling tools (Apple Instruments / sample on macOS) and llvm-mca when present.
Never fabricates cache counters, cycle counts, or instruction counts.
Distinguishes real runtime profiles from static estimates, and transparently documents unavailable tools.
"""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class ProfilingArtifact:
    tool_name: str
    available: bool
    status: str
    artifact_path: Optional[str] = None
    summary_data: Dict[str, Any] = field(default_factory=dict)
    text_report: str = ""
    top_functions: List[str] = field(default_factory=list)
    disclaimer: str = ""


class ProfilingAdapter:
    def __init__(self, artifacts_dir: Optional[str] = None):
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else Path("results/profiling")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.has_xctrace = shutil.which("xctrace") is not None
        self.has_sample = shutil.which("sample") is not None
        self.has_llvm_mca = shutil.which("llvm-mca") is not None

    def profile_workload(
        self,
        runner_bin: str,
        workload: str,
        candidate: str,
        size: int,
        duration_sec: int = 1
    ) -> ProfilingArtifact:
        """
        Profiles workload execution using macOS native 'sample' profiler.
        Captures real CPU call-stack traces and identifies hot kernel symbols.
        """
        if not self.has_sample:
            return ProfilingArtifact(
                tool_name="macOS sample",
                available=False,
                status="UNAVAILABLE",
                disclaimer="The macOS 'sample' profiler is not available on this system."
            )

        # Run with sufficient iterations so sampling can capture stacks
        runs = 50 if size <= 256 else 10
        cmd = [runner_bin, workload, candidate, str(size), "2", str(runs)]
        out_file = self.artifacts_dir / f"profile_{workload}_{candidate}_{size}.txt"

        try:
            # Start process
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Run sample
            sample_cmd = ["sample", str(p.pid), str(duration_sec), "1", "-file", str(out_file)]
            s_res = subprocess.run(sample_cmd, capture_output=True, text=True, timeout=duration_sec + 4)
            p.wait(timeout=10)

            if out_file.exists():
                with open(out_file, "r", errors="ignore") as f:
                    content = f.read(8000)

                # Parse hot symbols from sample output
                top_syms = []
                for line in content.splitlines():
                    if "axiom::" in line or "matmul" in line or "conv2d" in line:
                        clean = line.strip().split("(")[0].strip()
                        if clean and clean not in top_syms:
                            top_syms.append(clean)

                return ProfilingArtifact(
                    tool_name="macOS sample",
                    available=True,
                    status="SUCCESS",
                    artifact_path=str(out_file),
                    summary_data={"sample_file": str(out_file), "captured_symbols": len(top_syms)},
                    text_report=content[:2000],
                    top_functions=top_syms[:5],
                    disclaimer="Real call-stack sampling collected via macOS native sample profiler."
                )
            else:
                return ProfilingArtifact(
                    tool_name="macOS sample",
                    available=True,
                    status="COMPLETED_BEFORE_SAMPLE",
                    text_report=s_res.stderr or s_res.stdout,
                    disclaimer="Benchmark execution completed before the sampling interval window closed."
                )
        except Exception as e:
            return ProfilingArtifact(
                tool_name="macOS sample",
                available=True,
                status="FAILED",
                disclaimer=f"Sampling execution failed: {str(e)}"
            )

    def analyze_with_llvm_mca(self, assembly_path_or_content: str, target_cpu: str = "apple-m1") -> ProfilingArtifact:
        """Analyzes assembly with llvm-mca if installed."""
        if not self.has_llvm_mca:
            return ProfilingArtifact(
                tool_name="LLVM Machine Code Analyzer (llvm-mca)",
                available=False,
                status="TOOL_UNAVAILABLE",
                text_report="",
                disclaimer=(
                    "llvm-mca is not installed on PATH. Static pipeline simulation requires the LLVM MCA utility "
                    "with a supported micro-architectural scheduling model. No simulated cycles, dispatch stalls, "
                    "or IPC metrics have been fabricated."
                )
            )

        try:
            mca_bin = shutil.which("llvm-mca")
            proc = subprocess.run(
                [mca_bin, f"-mcpu={target_cpu}"],
                input=assembly_path_or_content,
                text=True,
                capture_output=True,
                timeout=15
            )
            if proc.returncode == 0:
                return ProfilingArtifact(
                    tool_name="LLVM Machine Code Analyzer (llvm-mca)",
                    available=True,
                    status="SUCCESS",
                    text_report=proc.stdout,
                    disclaimer=f"Static pipeline simulation generated by llvm-mca using target '{target_cpu}'."
                )
            else:
                return ProfilingArtifact(
                    tool_name="LLVM Machine Code Analyzer (llvm-mca)",
                    available=True,
                    status="ANALYSIS_ERROR",
                    text_report=proc.stderr,
                    disclaimer="llvm-mca returned a non-zero exit code on the input assembly."
                )
        except Exception as e:
            return ProfilingArtifact(
                tool_name="LLVM Machine Code Analyzer (llvm-mca)",
                available=True,
                status="EXECUTION_ERROR",
                disclaimer=f"Error executing llvm-mca: {str(e)}"
            )

    def static_instruction_breakdown(self, assembly_code: str) -> Dict[str, Any]:
        """Performs honest static instruction breakdown of generated ARM64 assembly."""
        lines = [l.strip() for l in assembly_code.splitlines() if l.strip() and not l.strip().startswith((".", ";", "//"))]

        vector_simd = 0
        scalar_fp = 0
        loads = 0
        stores = 0
        branches = 0
        other = 0

        for line in lines:
            tokens = line.split()
            if not tokens:
                continue
            op = tokens[0].lower()

            if any(op.startswith(p) for p in ["fmla", "fadd", "fmul", "fsub", "fcv", "fneg"]) and (
                any("v" in t for t in tokens[1:]) or op.startswith("ld1") or op.startswith("st1")
            ):
                vector_simd += 1
            elif any(op.startswith(p) for p in ["fmla", "fadd", "fmul", "fsub", "fmov"]):
                scalar_fp += 1
            elif op.startswith("ldr") or op.startswith("ldp") or op.startswith("ld1"):
                loads += 1
            elif op.startswith("str") or op.startswith("stp") or op.startswith("st1"):
                stores += 1
            elif op.startswith("b") or op.startswith("cbz") or op.startswith("cbnz") or op.startswith("ret"):
                branches += 1
            else:
                other += 1

        total_instructions = len(lines)
        return {
            "total_static_instructions": total_instructions,
            "vector_simd_instructions": vector_simd,
            "scalar_fp_instructions": scalar_fp,
            "memory_loads": loads,
            "memory_stores": stores,
            "control_branches": branches,
            "other_instructions": other,
            "vectorization_density_pct": (vector_simd / total_instructions * 100.0) if total_instructions > 0 else 0.0,
            "disclaimer": "Static instruction counts reflect disassembled compiler output, not dynamic execution counts."
        }
