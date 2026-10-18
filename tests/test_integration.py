import json
from pathlib import Path
from axiom.workload import WorkloadRegistry
from axiom.benchmark import BenchmarkEngine
from axiom.decision import DecisionEngine
from axiom.compiler import CompilerIntelligence
from axiom.store import ExperimentStore
from axiom.report import ReportGenerator
from axiom.env import get_environment_snapshot


def test_full_pipeline_integration(tmp_path):
    reg = WorkloadRegistry()
    be = BenchmarkEngine(registry=reg)

    # 1. Run benchmark for matmul including NEON and multithreaded
    suite = be.run_workload(
        workload_id="matmul",
        candidate_ids=["reference", "interchanged", "neon", "multithreaded"],
        sizes=[64, 128],
        warmup_runs=1,
        measurement_runs=3
    )
    assert len(suite.comparisons) == 6  # 3 candidates * 2 sizes

    # 2. Evaluate decisions
    dec_engine = DecisionEngine()
    decisions = {}
    for cmp in suite.comparisons:
        dec = dec_engine.evaluate(cmp)
        dec_key = f"{cmp.candidate_id}_{cmp.size}"
        decisions[dec_key] = {
            "verdict": dec.verdict.value,
            "speedup": dec.speedup,
            "rationale": dec.rationale,
            "statistically_significant": dec.statistically_significant
        }
        assert dec.is_correct is True

    # 3. Store record
    store = ExperimentStore(base_dir=str(tmp_path / "results"))
    env_snap = get_environment_snapshot()

    candidate_results_dict = {}
    for cid, sdict in suite.results_by_candidate.items():
        candidate_results_dict[cid] = {}
        for sz, er in sdict.items():
            candidate_results_dict[cid][sz] = {
                "median_ms": er.median_ms,
                "min_ms": er.min_ms,
                "max_ms": er.max_ms,
                "stddev_ms": er.stddev_ms,
                "iqr_ms": er.iqr_ms,
                "gflops": er.gflops,
                "correctness_passed": er.correctness_passed,
                "max_abs_error": er.max_abs_error,
                "max_rel_error": er.max_rel_error,
                "raw_times_ms": er.raw_times_ms
            }

    comparisons_list = [
        {
            "size": c.size,
            "candidate_id": c.candidate_id,
            "baseline_median_ms": c.baseline_median_ms,
            "candidate_median_ms": c.candidate_median_ms,
            "speedup": c.speedup,
            "correctness_passed": c.correctness_passed,
            "candidate_gflops": c.candidate_gflops,
            "candidate_p95_ms": c.candidate_p95_ms,
            "candidate_p99_ms": c.candidate_p99_ms,
            "p_value": c.p_value,
            "notes": c.notes
        }
        for c in suite.comparisons
    ]

    record = {
        "workload_id": "matmul",
        "warmup_runs": 1,
        "measurement_runs": 3,
        "environment": env_snap.to_dict(),
        "candidate_results": candidate_results_dict,
        "comparisons": comparisons_list,
        "decisions": decisions
    }

    json_path = store.save_run(record)
    assert Path(json_path).exists()
    assert store.summary_csv.exists()

    loaded = store.load_run(record["experiment_id"])
    assert loaded is not None

    # 4. Generate report
    rep_gen = ReportGenerator()
    md = rep_gen.generate_markdown_report(loaded, include_ai_synthesis=True)
    assert "# Axiom 2.1 Performance Engineering Report" in md
    assert "## 1. Research Question and Hypotheses" in md
    assert "## 2. Experimental Setup" in md
    assert "## 6. Correctness Evidence" in md
    assert "## 7. Raw and Summarized Performance Data" in md
    assert "## 10. Engineering Conclusion and Next Experiment" in md


def test_ml_infer_and_memory_workloads():
    reg = WorkloadRegistry()
    be = BenchmarkEngine(registry=reg)

    # ML Inference run
    ml_suite = be.run_workload(
        workload_id="ml_infer",
        candidate_ids=["reference", "vectorized"],
        sizes=[8],
        warmup_runs=1,
        measurement_runs=3
    )
    assert len(ml_suite.comparisons) == 1
    assert ml_suite.comparisons[0].correctness_passed is True

    # Memory working-set run
    mem_suite = be.run_workload(
        workload_id="memory",
        candidate_ids=["sequential", "strided"],
        sizes=[64],
        warmup_runs=1,
        measurement_runs=3
    )
    assert len(mem_suite.comparisons) == 1
    assert mem_suite.comparisons[0].correctness_passed is True
