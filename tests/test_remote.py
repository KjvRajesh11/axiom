"""Unit and protocol tests for Axiom 2.2 Remote AMD Execution Adapter.

Verifies:
- Mock AMD transport protocol executes end-to-end and reports is_mock=True.
- Never fabricates hardware performance results.
- Unreachable hosts raise clean ConnectionErrors without crashing.
- Experiment specifications are hashed with SHA-256 for integrity auditing.
- SSH command construction avoids exposing secrets and respects port/timeout settings.
"""

import pytest

from axiom.remote import (
    MockAMDTransport,
    RemoteAmdRunner,
    RemoteHostConfig,
    SSHRemoteTransport,
)


def test_mock_amd_transport_protocol():
    transport = MockAMDTransport()
    probe = transport.probe_host()

    assert probe.is_reachable is True
    assert probe.architecture == "x86_64"
    assert probe.is_authentic_amd is True
    assert probe.has_avx512 is True
    assert "EPYC" in probe.cpu_model

    runner = RemoteAmdRunner(transport)
    result = runner.run_remote_experiment(
        workload_id="matmul",
        candidate_id="avx512",
        size=256,
    )

    # Must explicitly state that this is a mock test
    assert result.is_mock is True
    assert result.exit_code == 0
    assert result.passed_correctness is True
    assert result.gflops > 0.0
    assert len(result.spec_sha256) == 64  # Valid SHA-256 hash


def test_ssh_command_building_and_credential_safety():
    cfg = RemoteHostConfig(
        host="amd-node.compute.internal",
        user="benchmark_user",
        port=2222,
        ssh_key_path="/path/to/private_key.id_ed25519",
        timeout_seconds=30,
    )
    ssh = SSHRemoteTransport(cfg)
    cmd = ssh._build_ssh_cmd("uname -m")

    assert cmd[0] == "ssh"
    assert "-p" in cmd and "2222" in cmd
    assert "-i" in cmd and "/path/to/private_key.id_ed25519" in cmd
    assert "benchmark_user@amd-node.compute.internal" in cmd
    assert "uname -m" in cmd
    # Ensure no passwords or raw credentials appear in command tokens
    assert not any("password" in token.lower() for token in cmd)


def test_unreachable_host_error_handling():
    cfg = RemoteHostConfig(
        host="192.0.2.1",  # Test-net unreachable IP
        user="testuser",
        timeout_seconds=1,
    )
    ssh = SSHRemoteTransport(cfg)
    runner = RemoteAmdRunner(ssh)

    # Probe must fail gracefully
    probe = ssh.probe_host()
    assert probe.is_reachable is False

    with pytest.raises(ConnectionError):
        runner.run_remote_experiment(workload_id="matmul", candidate_id="avx512")


def test_mock_amd_unsupported_instruction_rejection():
    # Transport reporting host CPU without AVX-512
    transport = MockAMDTransport(simulate_no_avx512=True)
    runner = RemoteAmdRunner(transport)
    result = runner.run_remote_experiment(workload_id="matmul", candidate_id="avx512")

    assert result.status.value == "UNSUPPORTED_INSTRUCTION_FAILURE"
    assert result.passed_correctness is False
    assert "AVX-512" in result.error_message


def test_mock_amd_remote_build_failure():
    # Transport simulating compiler/linker error during remote cmake build
    transport = MockAMDTransport(simulate_build_failure=True)
    runner = RemoteAmdRunner(transport)
    result = runner.run_remote_experiment(workload_id="matmul", candidate_id="avx512")

    assert result.status.value == "REMOTE_COMPILATION_FAILURE"
    assert result.passed_correctness is False
    assert "compilation failed" in result.error_message.lower()
    assert len(result.compiler_output) > 0


def test_mock_amd_remote_correctness_failure():
    # Transport simulating failure in remote correctness tests
    transport = MockAMDTransport(simulate_correctness_failure=True)
    runner = RemoteAmdRunner(transport)
    result = runner.run_remote_experiment(workload_id="matmul", candidate_id="avx512")

    assert result.status.value == "CORRECTNESS_FAILURE"
    assert result.passed_correctness is False
    assert "correctness" in result.error_message.lower()
    assert len(result.correctness_output) > 0


def test_mock_amd_remote_benchmark_failure():
    # Transport simulating crash/segfault in benchmark runner
    transport = MockAMDTransport(simulate_benchmark_failure=True)
    runner = RemoteAmdRunner(transport)
    result = runner.run_remote_experiment(workload_id="matmul", candidate_id="avx512")

    assert result.status.value == "BENCHMARK_FAILURE"
    assert result.passed_correctness is False


def test_payload_packing_integrity():
    from axiom.remote import pack_remote_payload
    b64_str, sha256_hex = pack_remote_payload()
    assert len(b64_str) > 0
    assert len(sha256_hex) == 64


def test_remote_runner_command_contract_against_real_binary():
    """Validates that the exact command constructed by RemoteAmdRunner executes cleanly on the real binary."""
    import json
    import subprocess
    from pathlib import Path

    runner_bin = Path("build/axiom_benchmark_runner")
    assert runner_bin.exists(), "Benchmark runner binary must be built before running this test"

    cmd = [
        str(runner_bin),
        "--workload", "matmul",
        "--candidate", "reference",
        "--size", "64",
        "--warmup", "1",
        "--runs", "3",
        "--format", "json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    assert res.returncode == 0, f"Benchmark runner failed: {res.stderr}"

    data = json.loads(res.stdout.strip())
    assert data["workload"] == "matmul"
    assert data["candidate"] == "reference"
    assert data["size"] == 64
    assert data["correctness_passed"] is True
    assert data["passed_correctness"] is True
    assert data["gflops"] > 0.0
    assert len(data["raw_times_ms"]) == 3


def test_real_binary_invalid_argument_rejection():
    """Validates that real binary rejects bad options, invalid bounds, and unknown identifiers with exit code 1."""
    import subprocess
    from pathlib import Path

    runner_bin = Path("build/axiom_benchmark_runner")
    assert runner_bin.exists()

    # Unknown option
    r = subprocess.run([str(runner_bin), "--invalid-option"], capture_output=True, text=True)
    assert r.returncode == 1
    assert "Unknown argument" in r.stderr

    # Missing option argument
    r = subprocess.run([str(runner_bin), "--workload"], capture_output=True, text=True)
    assert r.returncode == 1
    assert "requires an argument" in r.stderr

    # Unknown workload
    r = subprocess.run([str(runner_bin), "--workload", "bad_workload"], capture_output=True, text=True)
    assert r.returncode == 1
    assert "Unknown workload" in r.stderr

    # Unknown candidate
    r = subprocess.run([str(runner_bin), "--workload", "matmul", "--candidate", "bad_cand"], capture_output=True, text=True)
    assert r.returncode == 1
    assert "Unknown candidate" in r.stderr

    # Invalid size
    r = subprocess.run([str(runner_bin), "--size", "-5"], capture_output=True, text=True)
    assert r.returncode == 1
    assert "Invalid size" in r.stderr

    # Positional invalid size
    r = subprocess.run([str(runner_bin), "matmul", "reference", "-5"], capture_output=True, text=True)
    assert r.returncode == 1
    assert "Invalid size" in r.stderr


def test_mock_transport_command_argument_validation():
    """Validates that MockAMDTransport validates command arguments rather than blindly returning mock data."""
    transport = MockAMDTransport()

    # Valid command
    c_ok = "./build/axiom_benchmark_runner --workload matmul --candidate avx512 --size 256 --warmup 2 --runs 5 --format json"
    code, out, err = transport.execute_command(c_ok)
    assert code == 0
    assert "SIMULATED_PROTOCOL_MOCK_AMD_ZEN4_NOT_NATIVE" in out

    # Unknown workload in command
    c_bad_wl = "./build/axiom_benchmark_runner --workload fake_wl --candidate avx512 --size 256 --warmup 2 --runs 5 --format json"
    code, out, err = transport.execute_command(c_bad_wl)
    assert code == 1
    assert "Unknown workload" in err

    # Unknown option in command
    c_bad_opt = "./build/axiom_benchmark_runner --bad-flag"
    code, out, err = transport.execute_command(c_bad_opt)
    assert code == 1
    assert "Unknown argument" in err
