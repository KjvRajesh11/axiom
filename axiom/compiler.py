"""
Compiler Intelligence Engine for Axiom 2.0.
Invokes Clang/LLVM, captures compiler flags, produces generated target assembly,
extracts optimization remarks (successful, missed, analysis), and parses diagnostic evidence.
Never fabricates compiler behavior.
"""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class OptimizationRemark:
    remark_type: str  # 'passed', 'missed', 'analysis'
    pass_name: str
    message: str
    file_path: str
    line_number: Optional[int]


@dataclass
class CompilerDiagnostics:
    source_file: str
    compiler_path: str
    compiler_version: str
    target_triple: str
    flags: List[str]
    success: bool
    remarks_passed: List[OptimizationRemark] = field(default_factory=list)
    remarks_missed: List[OptimizationRemark] = field(default_factory=list)
    remarks_analysis: List[OptimizationRemark] = field(default_factory=list)
    raw_assembly: str = ""
    assembly_stats: Dict[str, int] = field(default_factory=dict)
    summary_explanation: str = ""


class CompilerIntelligence:
    def __init__(self, compiler_path: Optional[str] = None, include_dirs: Optional[List[str]] = None):
        self.compiler_path = compiler_path or shutil.which("clang++") or "/usr/bin/clang++"
        self.include_dirs = include_dirs or []
        self._compiler_version, self._target_triple = self._probe_compiler()

    def _probe_compiler(self) -> tuple[str, str]:
        ver = "Unknown Clang"
        triple = "Unknown Target"
        try:
            res = subprocess.run([self.compiler_path, "-v"], capture_output=True, text=True, timeout=5)
            combined = (res.stdout or "") + "\n" + (res.stderr or "")
            lines = combined.strip().splitlines()
            if lines:
                ver = lines[0]
            for l in lines:
                if l.startswith("Target:"):
                    triple = l.split("Target:", 1)[1].strip()
        except Exception:
            pass
        return ver, triple

    def generate_assembly(self, source_path: str, opt_level: str = "-O3", extra_flags: Optional[List[str]] = None) -> str:
        """Generates target assembly for a C++ source file."""
        args = [
            self.compiler_path,
            "-std=c++20",
            opt_level,
            "-S",
            "-fno-asynchronous-unwind-tables",
            "-fno-verbose-asm"
        ]
        for inc in self.include_dirs:
            args.extend(["-I", inc])
        if extra_flags:
            args.extend(extra_flags)
        args.extend([source_path, "-o", "-"])

        res = subprocess.run(args, capture_output=True, text=True, timeout=15)
        if res.returncode == 0:
            return res.stdout
        return f"; Error generating assembly (code {res.returncode}):\n; {res.stderr}"

    def capture_diagnostics(self, source_path: str, opt_level: str = "-O3") -> CompilerDiagnostics:
        """Captures optimization remarks and generated assembly for a source file."""
        args = [
            self.compiler_path,
            "-std=c++20",
            opt_level,
            "-c",
            "-Rpass=.*",
            "-Rpass-missed=.*",
            "-Rpass-analysis=.*"
        ]
        for inc in self.include_dirs:
            args.extend(["-I", inc])
        args.extend([source_path, "-o", "/dev/null"])

        res = subprocess.run(args, capture_output=True, text=True, timeout=15)
        combined_err = res.stderr or ""

        remarks_passed = []
        remarks_missed = []
        remarks_analysis = []

        # Parse Clang optimization remark format:
        # file.cpp:line:col: remark: message [-Rpass=pass-name]
        pattern = re.compile(r"^(.*?):(\d+):(?:\d+:)?\s*remark:\s*(.*?)\s*\[-R(pass|pass-missed|pass-analysis)=([^\]]+)\]", re.MULTILINE)
        for match in pattern.finditer(combined_err):
            fpath, line_str, msg, rtype, pname = match.groups()
            line = int(line_str) if line_str.isdigit() else None
            remark = OptimizationRemark(
                remark_type=rtype,
                pass_name=pname,
                message=msg.strip(),
                file_path=fpath,
                line_number=line
            )
            if rtype == "pass":
                remarks_passed.append(remark)
            elif rtype == "pass-missed":
                remarks_missed.append(remark)
            elif rtype == "pass-analysis":
                remarks_analysis.append(remark)

        # Generate target assembly
        asm = self.generate_assembly(source_path, opt_level)
        asm_stats = self._analyze_assembly(asm)

        # Build diagnostic explanation
        explanation = self._build_explanation(remarks_passed, remarks_missed, remarks_analysis, asm_stats)

        return CompilerDiagnostics(
            source_file=source_path,
            compiler_path=self.compiler_path,
            compiler_version=self._compiler_version,
            target_triple=self._target_triple,
            flags=args[1:-3],
            success=(res.returncode == 0),
            remarks_passed=remarks_passed,
            remarks_missed=remarks_missed,
            remarks_analysis=remarks_analysis,
            raw_assembly=asm,
            assembly_stats=asm_stats,
            summary_explanation=explanation
        )

    def diff_assembly(self, source_a: str, source_b: str, opt_level: str = "-O3") -> Dict[str, Any]:
        """Compares generated assembly between two source implementations."""
        asm_a = self.generate_assembly(source_a, opt_level)
        asm_b = self.generate_assembly(source_b, opt_level)
        stats_a = self._analyze_assembly(asm_a)
        stats_b = self._analyze_assembly(asm_b)

        diff = {
            "total_lines_diff": stats_b["total_lines"] - stats_a["total_lines"],
            "vector_fmla_diff": stats_b["vector_fmla_count"] - stats_a["vector_fmla_count"],
            "load_diff": stats_b["load_count"] - stats_a["load_count"],
            "store_diff": stats_b["store_count"] - stats_a["store_count"],
            "branch_diff": stats_b["branch_count"] - stats_a["branch_count"],
            "stats_a": stats_a,
            "stats_b": stats_b
        }
        return diff

    def _analyze_assembly(self, asm: str) -> Dict[str, int]:
        stats = {
            "total_lines": len(asm.splitlines()),
            "vector_fmla_count": len(re.findall(r"\bfmla\b", asm, re.IGNORECASE)),
            "vector_fmul_count": len(re.findall(r"\bfmul\b", asm, re.IGNORECASE)),
            "vector_fadd_count": len(re.findall(r"\bfadd\b", asm, re.IGNORECASE)),
            "load_count": len(re.findall(r"\bldr\b|\bld1\b", asm, re.IGNORECASE)),
            "store_count": len(re.findall(r"\bstr\b|\bst1\b", asm, re.IGNORECASE)),
            "branch_count": len(re.findall(r"\bb\b|\bcbnz\b|\bcbz\b|\bb\.", asm, re.IGNORECASE))
        }
        return stats

    def _build_explanation(
        self,
        passed: List[OptimizationRemark],
        missed: List[OptimizationRemark],
        analysis: List[OptimizationRemark],
        asm_stats: Dict[str, int]
    ) -> str:
        parts = []
        vec_passed = [r for r in passed if "vector" in r.pass_name.lower() or "vector" in r.message.lower()]
        vec_missed = [r for r in missed if "vector" in r.pass_name.lower() or "vector" in r.message.lower()]
        licm_passed = [r for r in passed if "licm" in r.pass_name.lower()]

        if vec_passed:
            parts.append(f"Auto-vectorization SUCCEEDED: LLVM loop vectorizer emitted: '{vec_passed[0].message}'.")
        elif vec_missed:
            parts.append(f"Auto-vectorization MISSED: LLVM diagnostic indicates: '{vec_missed[0].message}'.")
        else:
            parts.append("No explicit vectorizer remarks recorded.")

        if licm_passed:
            parts.append(f"Loop-Invariant Code Motion (LICM) hoisted {len(licm_passed)} invariant expressions out of inner loops.")

        v_ops = asm_stats.get("vector_fmla_count", 0) + asm_stats.get("vector_fadd_count", 0) + asm_stats.get("vector_fmul_count", 0)
        parts.append(f"Generated ARM64 assembly contains {v_ops} floating-point arithmetic instructions ({asm_stats.get('vector_fmla_count', 0)} fused multiply-add fmla) and {asm_stats.get('load_count', 0)} loads.")

        return " ".join(parts)
