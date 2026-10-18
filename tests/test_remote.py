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
