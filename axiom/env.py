"""
Environment Detection and Hardware Topology Probe for Axiom 2.0.
Detects real hardware capabilities, compiler toolchains, and available profilers.
Never fabricates counters, metrics, or machine architectures.
"""

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass
class HardwareTopology:
    cpu_brand: str
    architecture: str
    physical_cores: int
    logical_cores: int
    cache_line_bytes: int
    l1d_cache_bytes: Optional[int]
    l2_cache_bytes: Optional[int]


@dataclass
class ToolchainInfo:
    compiler_path: str
    compiler_version: str
    target_triple: str
    cmake_path: Optional[str]
    cmake_version: Optional[str]
    python_version: str


@dataclass
class ProfilerAvailability:
    has_xctrace: bool
    has_sample: bool
    has_llvm_mca: bool
    has_linux_perf: bool
    details: Dict[str, str]


@dataclass
class EnvironmentSnapshot:
    os_name: str
    os_release: str
    platform_system: str
    hardware: HardwareTopology
    toolchain: ToolchainInfo
    profilers: ProfilerAvailability

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _run_cmd(args: list[str]) -> Optional[str]:
    try:
        res = subprocess.run(args, capture_output=True, text=True, timeout=5)
        out = (res.stdout or "") + "\n" + (res.stderr or "")
        out = out.strip()
        if out:
            return out
    except Exception:
        pass
    return None


def probe_hardware() -> HardwareTopology:
    arch = platform.machine()
    brand = "Unknown CPU"
    cores_physical = os.cpu_count() or 1
    cores_logical = os.cpu_count() or 1
    cache_line = 64
    l1d = None
    l2 = None

    if platform.system() == "Darwin":
        # macOS sysctl queries
        b = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
        if b:
            brand = b
        
        pc = _run_cmd(["sysctl", "-n", "hw.physicalcpu"])
        if pc and pc.isdigit():
            cores_physical = int(pc)
            
        lc = _run_cmd(["sysctl", "-n", "hw.logicalcpu"])
        if lc and lc.isdigit():
            cores_logical = int(lc)

        cl = _run_cmd(["sysctl", "-n", "hw.cachelinesize"])
        if cl and cl.isdigit():
            cache_line = int(cl)

        l1 = _run_cmd(["sysctl", "-n", "hw.l1dcachesize"])
        if l1 and l1.isdigit():
            l1d = int(l1)

        l2_raw = _run_cmd(["sysctl", "-n", "hw.l2cachesize"])
        if l2_raw and l2_raw.isdigit():
            l2 = int(l2_raw)

    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        brand = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

    return HardwareTopology(
        cpu_brand=brand,
        architecture=arch,
        physical_cores=cores_physical,
        logical_cores=cores_logical,
        cache_line_bytes=cache_line,
        l1d_cache_bytes=l1d,
        l2_cache_bytes=l2
    )


def probe_toolchain() -> ToolchainInfo:
    clang_path = shutil.which("clang++") or shutil.which("clang") or "/usr/bin/clang++"
    clang_ver = "Unknown Clang"
    target_triple = "Unknown Target"

    out = _run_cmd([clang_path, "-v"])
    if out:
        lines = out.splitlines()
        if lines:
            clang_ver = lines[0]
        for l in lines:
            if l.startswith("Target:"):
                target_triple = l.split("Target:", 1)[1].strip()

    cmake_path = shutil.which("cmake")
    cmake_ver = None
    if cmake_path:
        out_cm = _run_cmd([cmake_path, "--version"])
        if out_cm:
            cmake_ver = out_cm.splitlines()[0]

    return ToolchainInfo(
        compiler_path=clang_path,
        compiler_version=clang_ver,
        target_triple=target_triple,
        cmake_path=cmake_path,
        cmake_version=cmake_ver,
        python_version=sys.version.split()[0]
    )


def probe_profilers() -> ProfilerAvailability:
    has_xctrace = shutil.which("xctrace") is not None
    has_sample = shutil.which("sample") is not None
    has_llvm_mca = shutil.which("llvm-mca") is not None
    has_perf = shutil.which("perf") is not None

    details = {}
    if has_xctrace:
        details["xctrace"] = shutil.which("xctrace") or "available"
    else:
        details["xctrace"] = "Unavailable on this system"

    if has_sample:
        details["sample"] = shutil.which("sample") or "available"
    else:
        details["sample"] = "Unavailable on this system"

    if has_llvm_mca:
        details["llvm-mca"] = shutil.which("llvm-mca") or "available"
    else:
        details["llvm-mca"] = "Unavailable on PATH (requires llvm suite)"

    if has_perf:
        details["perf"] = shutil.which("perf") or "available"
    else:
        details["perf"] = "Unavailable (Linux perf tool not found on macOS)"

    return ProfilerAvailability(
        has_xctrace=has_xctrace,
        has_sample=has_sample,
        has_llvm_mca=has_llvm_mca,
        has_linux_perf=has_perf,
        details=details
    )


def get_environment_snapshot() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        os_name=platform.system(),
        os_release=platform.release(),
        platform_system=platform.platform(),
        hardware=probe_hardware(),
        toolchain=probe_toolchain(),
        profilers=probe_profilers()
    )


if __name__ == "__main__":
    import json
    snap = get_environment_snapshot()
    print(json.dumps(snap.to_dict(), indent=2))
