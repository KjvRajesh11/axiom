"""
Validation Sandbox for Axiom 2.0.
Provides isolated compilation, safe subprocess invocation (no shell injection),
timeout enforcement, sanitizer execution (ASan/UBSan), and strict numerical verification.
Rejects failing, crashing, or timeout-violating candidates immediately.
"""

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


@dataclass
class BuildResult:
    success: bool
    executable_path: Optional[str]
    compiler_output: str
    returncode: int
    sanitized: bool = False


@dataclass
class ExecutionResult:
    success: bool
    workload: str
    candidate: str
    size: int
    correctness_passed: bool
    max_abs_error: float
    max_rel_error: float
    raw_times_ms: List[float]
    min_ms: float
    max_ms: float
    median_ms: float
    mean_ms: float
    stddev_ms: float
    iqr_ms: float
    gflops: float
    error_message: Optional[str] = None
    timeout: bool = False


class ValidationSandbox:
    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = Path(root_dir) if root_dir else Path(__file__).resolve().parent.parent
        self.build_dir = self.root_dir / "build"
        self.runner_binary = self.build_dir / "axiom_benchmark_runner"
        self.tests_binary = self.build_dir / "axiom_correctness_tests"

    def ensure_binaries(self, rebuild: bool = False) -> bool:
        """Ensures the C++ benchmark runner and tests are built."""
        if not rebuild and self.runner_binary.exists() and self.tests_binary.exists():
            return True

        self.build_dir.mkdir(parents=True, exist_ok=True)
        cmake_cmd = ["cmake", "-B", str(self.build_dir), "-DCMAKE_BUILD_TYPE=Release"]
        build_cmd = ["cmake", "--build", str(self.build_dir), "-j", "4"]

        try:
            r1 = subprocess.run(cmake_cmd, cwd=str(self.root_dir), capture_output=True, text=True, timeout=30)
            if r1.returncode != 0:
                return False
            r2 = subprocess.run(build_cmd, cwd=str(self.root_dir), capture_output=True, text=True, timeout=60)
            return r2.returncode == 0
        except Exception:
            return False

    def build_with_sanitizers(self, temp_dir: Optional[str] = None) -> BuildResult:
        """Builds workload with AddressSanitizer and UndefinedBehaviorSanitizer for safety validation."""
        target_dir = Path(temp_dir) if temp_dir else (self.build_dir / "sanitized")
        target_dir.mkdir(parents=True, exist_ok=True)

        cmake_cmd = [
            "cmake",
            "-B", str(target_dir),
            "-DCMAKE_BUILD_TYPE=Debug",
            "-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer"
        ]
        build_cmd = ["cmake", "--build", str(target_dir), "-j", "4"]

        try:
            r1 = subprocess.run(cmake_cmd, cwd=str(self.root_dir), capture_output=True, text=True, timeout=30)
            if r1.returncode != 0:
                return BuildResult(False, None, r1.stderr, r1.returncode, True)
            r2 = subprocess.run(build_cmd, cwd=str(self.root_dir), capture_output=True, text=True, timeout=60)
            exe_path = str(target_dir / "axiom_correctness_tests")
            return BuildResult(r2.returncode == 0, exe_path if r2.returncode == 0 else None, r2.stdout + r2.stderr, r2.returncode, True)
        except Exception as e:
            return BuildResult(False, None, str(e), -1, True)

    def run_candidate(
        self,
        workload: str,
        candidate: str,
        size: int = 256,
        warmup_runs: int = 3,
        measurement_runs: int = 15,
        timeout_seconds: int = 45
    ) -> ExecutionResult:
        """Safely executes candidate benchmark in isolated subprocess with arguments."""
        if not self.ensure_binaries():
            return ExecutionResult(
                success=False,
                workload=workload,
                candidate=candidate,
                size=size,
                correctness_passed=False,
                max_abs_error=0.0,
                max_rel_error=0.0,
                raw_times_ms=[],
                min_ms=0.0,
                max_ms=0.0,
                median_ms=0.0,
                mean_ms=0.0,
                stddev_ms=0.0,
                iqr_ms=0.0,
                gflops=0.0,
                error_message="Failed to locate or build benchmark runner binary."
            )

        cmd = [
            str(self.runner_binary),
            str(workload),
            str(candidate),
            str(size),
            str(warmup_runs),
            str(measurement_runs)
        ]

        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                workload=workload,
                candidate=candidate,
                size=size,
                correctness_passed=False,
                max_abs_error=0.0,
                max_rel_error=0.0,
                raw_times_ms=[],
                min_ms=0.0,
                max_ms=0.0,
                median_ms=0.0,
                mean_ms=0.0,
                stddev_ms=0.0,
                iqr_ms=0.0,
                gflops=0.0,
                error_message=f"Process timed out after {timeout_seconds}s.",
                timeout=True
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                workload=workload,
                candidate=candidate,
                size=size,
                correctness_passed=False,
                max_abs_error=0.0,
                max_rel_error=0.0,
                raw_times_ms=[],
                min_ms=0.0,
                max_ms=0.0,
                median_ms=0.0,
                mean_ms=0.0,
                stddev_ms=0.0,
                iqr_ms=0.0,
                gflops=0.0,
                error_message=f"Process failed to execute: {str(e)}"
            )

        if res.returncode != 0 and res.returncode != 2:
            return ExecutionResult(
                success=False,
                workload=workload,
                candidate=candidate,
                size=size,
                correctness_passed=False,
                max_abs_error=0.0,
                max_rel_error=0.0,
                raw_times_ms=[],
                min_ms=0.0,
                max_ms=0.0,
                median_ms=0.0,
                mean_ms=0.0,
                stddev_ms=0.0,
                iqr_ms=0.0,
                gflops=0.0,
                error_message=f"Process crashed or returned error exit code {res.returncode}:\n{res.stderr}"
            )

        # Parse JSON output from runner
        try:
            data = json.loads(res.stdout)
            passed = data.get("correctness_passed", False)
            return ExecutionResult(
                success=(res.returncode == 0 and passed),
                workload=data.get("workload", workload),
                candidate=data.get("candidate", candidate),
                size=data.get("size", size),
                correctness_passed=passed,
                max_abs_error=data.get("max_abs_error", 0.0),
                max_rel_error=data.get("max_rel_error", 0.0),
                raw_times_ms=data.get("raw_times_ms", []),
                min_ms=data.get("min_ms", 0.0),
                max_ms=data.get("max_ms", 0.0),
                median_ms=data.get("median_ms", 0.0),
                mean_ms=data.get("mean_ms", 0.0),
                stddev_ms=data.get("stddev_ms", 0.0),
                iqr_ms=data.get("iqr_ms", 0.0),
                gflops=data.get("gflops", 0.0),
                error_message=None if passed else "Correctness check failed (numerical tolerance exceeded)."
            )
        except json.JSONDecodeError as err:
            return ExecutionResult(
                success=False,
                workload=workload,
                candidate=candidate,
                size=size,
                correctness_passed=False,
                max_abs_error=0.0,
                max_rel_error=0.0,
                raw_times_ms=[],
                min_ms=0.0,
                max_ms=0.0,
                median_ms=0.0,
                mean_ms=0.0,
                stddev_ms=0.0,
                iqr_ms=0.0,
                gflops=0.0,
                error_message=f"Failed to parse runner JSON output: {err}\nOutput was: {res.stdout}"
            )
