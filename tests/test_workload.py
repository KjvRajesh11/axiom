from pathlib import Path
from axiom.workload import WorkloadRegistry, compute_file_hash


def test_workload_registry_full():
    reg = WorkloadRegistry()
    workloads = reg.list_workloads()
    assert len(workloads) == 4

    # 1. matmul
    matmul = reg.get("matmul")
    assert matmul is not None
    cand_ids = [c.candidate_id for c in matmul.candidates]
    assert "reference" in cand_ids
    assert "interchanged" in cand_ids
    assert "blocked" in cand_ids
    assert "neon" in cand_ids
    assert "multithreaded" in cand_ids
    assert "regressed" in cand_ids

    # 2. conv2d
    conv2d = reg.get("conv2d")
    assert conv2d is not None
    assert len(conv2d.candidates) == 3

    # 3. ml_infer
    ml_infer = reg.get("ml_infer")
    assert ml_infer is not None
    assert len(ml_infer.candidates) == 2

    # 4. memory
    memory = reg.get("memory")
    assert memory is not None
    assert len(memory.candidates) == 2


def test_file_hashing():
    reg = WorkloadRegistry()
    for wl in reg.list_workloads():
        assert Path(wl.reference_file).exists()
        h = compute_file_hash(wl.reference_file)
        assert len(h) == 64
        assert wl.reference_hash == h
