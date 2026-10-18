from axiom.compiler import CompilerIntelligence
from axiom.workload import WorkloadRegistry


def test_compiler_assembly_generation():
    ci = CompilerIntelligence(include_dirs=["cpp/include"])
    reg = WorkloadRegistry()
    wl = reg.get("matmul")
    assert wl is not None

    asm = ci.generate_assembly(wl.candidates[1].source_file)
    assert len(asm) > 0
    assert "matmul_interchanged" in asm or "section" in asm


def test_compiler_diagnostics_capture():
    ci = CompilerIntelligence(include_dirs=["cpp/include"])
    reg = WorkloadRegistry()
    wl = reg.get("matmul")
    assert wl is not None

    # Reference should show missed vectorization
    diag_ref = ci.capture_diagnostics(wl.candidates[0].source_file)
    assert diag_ref.success is True
    assert "the cost-model indicates that vectorization is not beneficial" in [
        r.message for r in diag_ref.remarks_missed
    ]

    # Interchanged candidate should show successful vectorization
    diag_opt = ci.capture_diagnostics(wl.candidates[1].source_file)
    assert diag_opt.success is True
    assert any("vectorized loop" in r.message for r in diag_opt.remarks_passed)
    assert diag_opt.assembly_stats["vector_fmla_count"] > 0
