from axiom.sandbox import ValidationSandbox


def test_sandbox_build_and_run():
    sb = ValidationSandbox()
    assert sb.ensure_binaries() is True

    # Run small matmul reference
    res = sb.run_candidate(
        workload="matmul",
        candidate="reference",
        size=64,
        warmup_runs=1,
        measurement_runs=3,
        timeout_seconds=10
    )
    assert res.success is True
    assert res.correctness_passed is True
    assert res.median_ms > 0
    assert len(res.raw_times_ms) == 3


def test_sandbox_invalid_candidate_rejection():
    sb = ValidationSandbox()
    res = sb.run_candidate(
        workload="matmul",
        candidate="nonexistent_candidate",
        size=64,
        warmup_runs=1,
        measurement_runs=1,
        timeout_seconds=5
    )
    assert res.success is False
