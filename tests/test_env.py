import platform
from axiom.env import get_environment_snapshot, probe_hardware, probe_toolchain, probe_profilers


def test_environment_snapshot():
    snap = get_environment_snapshot()
    assert snap.os_name == platform.system()
    assert snap.hardware.architecture == platform.machine()
    assert snap.hardware.physical_cores >= 1
    assert snap.hardware.logical_cores >= 1
    assert snap.hardware.cache_line_bytes in (64, 128)


def test_toolchain_probe():
    tc = probe_toolchain()
    assert tc.compiler_path is not None
    assert "clang" in tc.compiler_path.lower()
    assert tc.python_version is not None
    assert tc.cmake_path is not None


def test_profilers_probe():
    prof = probe_profilers()
    # On macOS, xctrace and sample should be present
    if platform.system() == "Darwin":
        assert prof.has_xctrace is True
        assert prof.has_sample is True
    # llvm-mca is handled honestly
    assert isinstance(prof.has_llvm_mca, bool)
