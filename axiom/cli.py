"""
Command Line Interface for Axiom 2.1.
Provides unified access to environment probing, workload inspection,
compiler diagnostics, assembly diffing, benchmark execution, correctness validation,
controlled regression demonstration, multithread scaling, and report generation.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from axiom.env import get_environment_snapshot
from axiom.workload import WorkloadRegistry
from axiom.compiler import CompilerIntelligence
from axiom.hypothesis import HypothesisEngine
from axiom.sandbox import ValidationSandbox
from axiom.benchmark import BenchmarkEngine
from axiom.decision import DecisionEngine, DecisionCriteria, DecisionVerdict
from axiom.profiling import ProfilingAdapter
from axiom.store import ExperimentStore
from axiom.report import ReportGenerator
from axiom.llm import get_llm_provider


def cmd_env(args):
    snap = get_environment_snapshot()
    hw = snap.hardware
    tc = snap.toolchain
    prof = snap.profilers

    print("================================================================================")
    print("                      AXIOM 2.1 ENVIRONMENT & HARDWARE PROBE                    ")
    print("================================================================================")
    print(f"Host OS:             {snap.os_name} {snap.os_release} ({snap.platform_system})")
    print(f"CPU Brand:           {hw.cpu_brand} ({hw.architecture})")
    print(f"CPU Cores:           {hw.physical_cores} Physical / {hw.logical_cores} Logical")
    print(f"Cache Line Size:     {hw.cache_line_bytes} bytes")
    l1_kb = (hw.l1d_cache_bytes // 1024) if hw.l1d_cache_bytes else "Unknown"
    l2_mb = (hw.l2_cache_bytes // (1024 * 1024)) if hw.l2_cache_bytes else "Unknown"
    print(f"L1 Data Cache:       {l1_kb} KB")
    print(f"L2 Cache:            {l2_mb} MB")
    print("--------------------------------------------------------------------------------")
    print(f"Compiler:            {tc.compiler_path}")
    print(f"Compiler Version:    {tc.compiler_version}")
    print(f"Target Triple:       {tc.target_triple}")
    print(f"CMake:               {tc.cmake_path} ({tc.cmake_version or 'N/A'})")
    print(f"Python:              {tc.python_version}")
    print("--------------------------------------------------------------------------------")
    print("Profiling Tools:")
    print(f"  xctrace:           {'AVAILABLE' if prof.has_xctrace else 'UNAVAILABLE'}")
    print(f"  sample:            {'AVAILABLE' if prof.has_sample else 'UNAVAILABLE'}")
    print(f"  llvm-mca:          {'AVAILABLE' if prof.has_llvm_mca else 'UNAVAILABLE (requires LLVM suite)'}")
    print(f"  perf:              {'AVAILABLE' if prof.has_linux_perf else 'UNAVAILABLE (Linux only)'}")
    print("================================================================================")


def cmd_workloads(args):
    reg = WorkloadRegistry()
    print("================================================================================")
    print("                        REGISTERED AXIOM 2.1 WORKLOADS                          ")
    print("================================================================================")
    for wl in reg.list_workloads():
        print(f"\nWorkload ID:    {wl.workload_id}")
        print(f"Name:           {wl.name}")
        print(f"Domain:         {wl.domain}")
        print(f"Description:    {wl.description}")
        print(f"Default Size:   {wl.default_size} (Supported: {wl.supported_sizes})")
        print(f"FLOP/Bandwidth: {wl.flop_formula} ({wl.unit_label})")
        print("Candidates:")
        for c in wl.candidates:
            print(f"  - [{c.candidate_id}] {c.name}: {c.description}")
    print("================================================================================")


def cmd_diagnose(args):
    reg = WorkloadRegistry()
    wl = reg.get(args.workload)
    if not wl:
        print(f"Error: Workload '{args.workload}' not found.", file=sys.stderr)
        return 1

    ci = CompilerIntelligence(include_dirs=["cpp/include"])
    print(f"Analyzing Compiler Diagnostics for Workload '{args.workload}'...\n")

    for cand in wl.candidates:
        if args.candidate and cand.candidate_id != args.candidate:
            continue
        print("--------------------------------------------------------------------------------")
        print(f"Candidate: [{cand.candidate_id}] - {cand.source_file}")
        diag = ci.capture_diagnostics(cand.source_file)
        print(f"Explanation:    {diag.summary_explanation}")
        print(f"Assembly Stats: {diag.assembly_stats}")
        if diag.remarks_passed:
            print(f"Successful Optimization Remarks ({len(diag.remarks_passed)}):")
            for r in diag.remarks_passed[:3]:
                print(f"  [+PASS] {r.pass_name}: {r.message} (line {r.line_number})")
        if diag.remarks_missed:
            print(f"Missed Optimization Remarks ({len(diag.remarks_missed)}):")
            for r in diag.remarks_missed[:3]:
                print(f"  [-MISS] {r.pass_name}: {r.message} (line {r.line_number})")
    print("--------------------------------------------------------------------------------")


def cmd_diff_asm(args):
    reg = WorkloadRegistry()
    wl = reg.get(args.workload)
    if not wl:
        print(f"Error: Workload '{args.workload}' not found.", file=sys.stderr)
        return 1

    cands = {c.candidate_id: c for c in wl.candidates}
    if args.cand1 not in cands or args.cand2 not in cands:
        print(f"Error: Candidates must be from {list(cands.keys())}", file=sys.stderr)
        return 1

    ci = CompilerIntelligence(include_dirs=["cpp/include"])
    diff = ci.diff_assembly(cands[args.cand1].source_file, cands[args.cand2].source_file)

    print("================================================================================")
    print(f" ASSEMBLY DIFFERENTIAL ANALYSIS: [{args.cand1}] vs [{args.cand2}]")
    print("================================================================================")
    print(f"Metric                       [{args.cand1}]        [{args.cand2}]        Delta")
    print("--------------------------------------------------------------------------------")
    s1 = diff["stats_a"]
    s2 = diff["stats_b"]
    for k in ["total_lines", "vector_fmla_count", "load_count", "store_count", "branch_count"]:
        v1 = s1.get(k, 0)
        v2 = s2.get(k, 0)
        delta = v2 - v1
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        print(f"{k:<28} {v1:<18} {v2:<18} {delta_str}")
    print("================================================================================")


def cmd_run(args):
    reg = WorkloadRegistry()
    wl = reg.get(args.workload)
    if not wl:
        print(f"Error: Workload '{args.workload}' not found.", file=sys.stderr)
        return 1

    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else [wl.default_size]
    candidates = args.candidates.split(",") if args.candidates else None

    print("================================================================================")
    print(f" RUNNING AXIOM 2.1 EXPERIMENT: {wl.name}")
    print(f" Sizes: {sizes} | Warmup: {args.warmup} | Runs: {args.runs}")
    print("================================================================================")

    # 1. Capture environment
    env_snap = get_environment_snapshot()

    # 2. Capture compiler diagnostics
    ci = CompilerIntelligence(include_dirs=["cpp/include"])
    diagnostics = {}
    for cand in wl.candidates:
        d = ci.capture_diagnostics(cand.source_file)
        diagnostics[cand.candidate_id] = {
            "summary_explanation": d.summary_explanation,
            "assembly_stats": d.assembly_stats
        }

    # 3. Generate formal hypotheses
    hyp_engine = HypothesisEngine()
    hypotheses = []
    for cand in wl.candidates:
        if cand.candidate_id != "reference":
            hyps = hyp_engine.generate_hypotheses_for_workload(wl.workload_id, cand.candidate_id, size=sizes[0])
            hypotheses.extend(hyps)

    # 4. Benchmark execution
    be = BenchmarkEngine(registry=reg)
    suite_res = be.run_workload(
        workload_id=args.workload,
        candidate_ids=candidates,
        sizes=sizes,
        warmup_runs=args.warmup,
        measurement_runs=args.runs
    )

    # 5. Evaluate decisions and hypotheses
    dec_engine = DecisionEngine(DecisionCriteria(min_speedup=args.min_speedup))
    decisions = {}
    for cmp in suite_res.comparisons:
        dec = dec_engine.evaluate(cmp)
        dec_key = f"{cmp.candidate_id}_{cmp.size}"
        decisions[dec_key] = {
            "verdict": dec.verdict.value,
            "speedup": dec.speedup,
            "rationale": dec.rationale,
            "statistically_significant": dec.statistically_significant
        }
        for h in hypotheses:
            if h.candidate_id == cmp.candidate_id:
                h.evaluate(cmp.speedup, cmp.p_value)

    # Display results table
    print("\nRESULTS SUMMARY:")
    print(f"{'Size':<6} {'Candidate':<16} {'Median (ms)':<12} {'StdDev':<9} {wl.unit_label:<10} {'vs Base':<9} {'vs Best 1T':<14} {'Correct':<8} {'Verdict':<15}")
    print("-" * 102)

    for size in sizes:
        base_res = suite_res.results_by_candidate.get("reference", {}).get(size)
        if base_res:
            print(f"{size:<6} {'reference':<16} {base_res.median_ms:<12.3f} {base_res.stddev_ms:<9.3f} {base_res.gflops:<10.2f} {'1.00x':<9} {'N/A (base)':<14} {'PASS':<8} {'BASELINE':<15}")

        for cmp in suite_res.comparisons:
            if cmp.size == size:
                cand_exec = suite_res.results_by_candidate.get(cmp.candidate_id, {}).get(size)
                cand_sd = cand_exec.stddev_ms if cand_exec else 0.0
                v = decisions.get(f"{cmp.candidate_id}_{size}", {}).get("verdict", "N/A")
                c_pass = "PASS" if cmp.correctness_passed else "FAIL"
                vs_single = f"{cmp.speedup_vs_best_single:.2f}x ({cmp.best_single_id})" if cmp.speedup_vs_best_single is not None else "1T candidate"
                print(f"{size:<6} {cmp.candidate_id:<16} {cmp.candidate_median_ms:<12.3f} {cand_sd:<9.3f} {cmp.candidate_gflops:<10.2f} {f'{cmp.speedup:.2f}x':<9} {vs_single:<14} {c_pass:<8} {v:<15}")

    # Build experiment record
    candidate_results_dict = {}
    for cid, sdict in suite_res.results_by_candidate.items():
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
            "speedup_vs_best_single": c.speedup_vs_best_single,
            "best_single_id": c.best_single_id,
            "correctness_passed": c.correctness_passed,
            "candidate_gflops": c.candidate_gflops,
            "candidate_p95_ms": c.candidate_p95_ms,
            "candidate_p99_ms": c.candidate_p99_ms,
            "p_value": c.p_value,
            "notes": c.notes
        }
        for c in suite_res.comparisons
    ]

    record = {
        "workload_id": args.workload,
        "warmup_runs": args.warmup,
        "measurement_runs": args.runs,
        "environment": env_snap.to_dict(),
        "compiler_diagnostics": diagnostics,
        "hypotheses": [
            {
                "hypothesis_id": h.hypothesis_id,
                "candidate_id": h.candidate_id,
                "category": h.category,
                "bound_regime": h.bound_regime.value,
                "statement": h.statement,
                "uncertainty": h.uncertainty.value,
                "supporting_evidence": h.supporting_evidence,
                "confirmation_criteria": h.confirmation_criteria,
                "status": h.status.value,
                "evaluation_notes": h.evaluation_notes
            }
            for h in hypotheses
        ],
        "candidate_results": candidate_results_dict,
        "comparisons": comparisons_list,
        "decisions": decisions
    }

    store = ExperimentStore()
    json_path = store.save_run(record)
    print(f"\n[+] Saved Experiment Record to: {json_path}")
    print(f"[+] Appended summary row to:   {store.summary_csv}")

    rep_gen = ReportGenerator()
    md_content = rep_gen.generate_markdown_report(record, include_ai_synthesis=True)
    report_path = Path("results") / f"report_{record.get('experiment_id')}.md"
    with open(report_path, "w") as f:
        f.write(md_content)
    print(f"[+] Generated Markdown Report:  {report_path}")

    return 0


def cmd_regression_demo(args):
    """Executes controlled regression demonstration (Module F)."""
    print("================================================================================")
    print("         AXIOM 2.1 CONTROLLED PERFORMANCE REGRESSION DEMONSTRATION             ")
    print("================================================================================")
    print("Scenario: Evaluating an intentionally inefficient column-strided candidate.")
    print("Goal: Prove Decision Engine flags and rejects performance regression with p < 0.01.")
    print("--------------------------------------------------------------------------------")

    be = BenchmarkEngine()
    suite = be.run_workload(
        workload_id="matmul",
        candidate_ids=["reference", "regressed"],
        sizes=[256],
        warmup_runs=2,
        measurement_runs=10
    )

    dec_engine = DecisionEngine()
    cmp = suite.comparisons[0]
    decision = dec_engine.evaluate(cmp)

    print(f"Workload:           GEMM F32 (N=256)")
    print(f"Reference Baseline: {cmp.baseline_median_ms:.3f} ms ({cmp.baseline_gflops:.2f} GFLOP/s)")
    print(f"Regressed Candidate:{cmp.candidate_median_ms:.3f} ms ({cmp.candidate_gflops:.2f} GFLOP/s)")
    print(f"Observed Speedup:   {cmp.speedup:.2f}x ({1.0/cmp.speedup:.2f}x SLOWER)")
    print(f"Statistical p-val:  {cmp.p_value:.6f} (Significant: {cmp.is_statistically_significant})")
    print(f"Numerical Check:    {'PASSED' if cmp.correctness_passed else 'FAILED'}")
    print("--------------------------------------------------------------------------------")
    print(f"DECISION ENGINE VERDICT: [{decision.verdict.value}]")
    print(f"Rationale: {decision.rationale}")
    print("================================================================================")
    return 0


def cmd_multithread_sweep(args):
    """Executes multithreaded scaling benchmark across problem sizes (Module B2)."""
    print("================================================================================")
    print("           AXIOM 2.2 MULTITHREADED PARALLEL SCALING LABORATORY                  ")
    print("================================================================================")
    sizes = [128, 256, 512]
    be = BenchmarkEngine()
    suite = be.run_workload(
        workload_id="matmul",
        candidate_ids=["reference", "interchanged", "neon", "multithreaded"],
        sizes=sizes,
        warmup_runs=2,
        measurement_runs=5
    )

    print(f"{'Size':<6} {'Candidate':<16} {'Median (ms)':<12} {'GFLOP/s':<10} {'vs Base':<10} {'vs Best 1T':<16}")
    print("-" * 74)
    for sz in sizes:
        base_res = suite.results_by_candidate.get("reference", {}).get(sz)
        inter_res = suite.results_by_candidate.get("interchanged", {}).get(sz)
        neon_res = suite.results_by_candidate.get("neon", {}).get(sz)
        mt_res = suite.results_by_candidate.get("multithreaded", {}).get(sz)

        # find best single-threaded
        best_single_cand = "interchanged"
        best_single_res = inter_res
        if neon_res and (best_single_res is None or neon_res.median_ms < best_single_res.median_ms):
            best_single_cand = "neon"
            best_single_res = neon_res

        if base_res:
            print(f"{sz:<6} {'reference':<16} {base_res.median_ms:<12.3f} {base_res.gflops:<10.2f} {'1.00x':<10} {'N/A (base)':<16}")
        if inter_res and base_res:
            sp_inter = base_res.median_ms / inter_res.median_ms
            print(f"{sz:<6} {'interchanged':<16} {inter_res.median_ms:<12.3f} {inter_res.gflops:<10.2f} {f'{sp_inter:.2f}x':<10} {'1T candidate':<16}")
        if neon_res and base_res:
            sp_neon = base_res.median_ms / neon_res.median_ms
            print(f"{sz:<6} {'neon':<16} {neon_res.median_ms:<12.3f} {neon_res.gflops:<10.2f} {f'{sp_neon:.2f}x':<10} {'1T candidate (best)':<16}")
        if mt_res and base_res and best_single_res:
            sp_mt = base_res.median_ms / mt_res.median_ms
            mt_vs_single = best_single_res.median_ms / mt_res.median_ms
            print(f"{sz:<6} {'multithreaded':<16} {mt_res.median_ms:<12.3f} {mt_res.gflops:<10.2f} {f'{sp_mt:.2f}x':<10} {f'{mt_vs_single:.2f}x ({best_single_cand})':<16}")
        print("-" * 74)


def cmd_memory_sweep(args):
    """Executes working-set size and access pattern memory sweep (Module B3)."""
    print("================================================================================")
    print("             AXIOM 2.1 MEMORY WORKING-SET & LOCALITY SWEEP                      ")
    print("================================================================================")
    sizes = [16, 64, 256, 1024, 4096, 16384]
    be = BenchmarkEngine()
    suite = be.run_workload(
        workload_id="memory",
        candidate_ids=["sequential", "strided"],
        sizes=sizes,
        warmup_runs=2,
        measurement_runs=5
    )

    print(f"{'Buffer Size':<15} {'Working Set':<15} {'Sequential (GB/s)':<20} {'Strided 128B (GB/s)':<20} {'Ratio':<10}")
    print("-" * 80)
    for sz in sizes:
        seq_res = suite.results_by_candidate.get("sequential", {}).get(sz)
        str_res = suite.results_by_candidate.get("strided", {}).get(sz)

        tier = "L1 Cache" if sz <= 64 else ("L2 Cache" if sz <= 4096 else "DRAM")
        sz_str = f"{sz} KB" if sz < 1024 else f"{sz // 1024} MB"

        if seq_res and str_res:
            ratio = seq_res.gflops / str_res.gflops if str_res.gflops > 0 else 0.0
            print(f"{sz_str:<15} {tier:<15} {seq_res.gflops:<20.2f} {str_res.gflops:<20.2f} {f'{ratio:.2f}x':<10}")
    print("================================================================================")


def cmd_report(args):
    store = ExperimentStore()
    record = store.load_run(args.experiment_id)
    if not record:
        print(f"Error: Experiment run '{args.experiment_id}' not found.", file=sys.stderr)
        return 1

    rep_gen = ReportGenerator()
    md = rep_gen.generate_markdown_report(record, include_ai_synthesis=True)
    if args.output:
        with open(args.output, "w") as f:
            f.write(md)
        print(f"Report written to: {args.output}")
    else:
        print(md)


def cmd_test(args):
    print("=== Running Axiom 2.1 Test Suite ===")
    print("1. Running C++ correctness verification suite...")
    res_cpp = subprocess.run(["./build/axiom_correctness_tests"])
    if res_cpp.returncode != 0:
        print("FAIL: C++ test suite failed.", file=sys.stderr)
        return res_cpp.returncode

    print("\n2. Running Python unit and integration tests with pytest...")
    res_py = subprocess.run(["pytest", "-v", "tests"])
    return res_py.returncode


def cmd_ast_transform(args):
    """Executes Clang AST parsing, precondition checks, candidate generation, compilation, and numerical verification."""
    from axiom.ast_transform import ConstrainedLoopTransformer

    print("================================================================================")
    print("      AXIOM 2.2 CONSTRAINED CLANG AST TRANSFORMATION ENGINE                     ")
    print("================================================================================")
    print(f"Source file:   {args.source}")
    print(f"Target func:   {args.func}")
    print(f"Candidate ID:  {args.candidate_id}")
    print(f"Artifact out:  {args.out}")
    print("--------------------------------------------------------------------------------")

    transformer = ConstrainedLoopTransformer()
    res = transformer.transform_loop_interchange(
        source_file=args.source,
        func_name=args.func,
        output_candidate_path=args.out,
        candidate_id=args.candidate_id,
    )

    print(f"Compiler Toolchain:  {res.compiler_version}")
    print("\nPRECONDITION EVALUATION REPORT:")
    for chk in res.preconditions.checks:
        status_str = "[PASS]" if chk.passed else "[FAIL]"
        print(f"  {status_str} {chk.name}: {chk.details}")

    print("\nVERIFICATION STATUS:")
    print(f"  Preconditions Met:    {'YES' if res.preconditions.satisfied else 'NO'}")
    print(f"  Independent Build:    {'SUCCESS' if res.compilation_passed else 'FAILED'}")
    print(f"  Numerical Tolerance:  {'PASSED (max_rel=' + str(res.max_relative_error) + ')' if res.correctness_passed else 'FAILED/SKIPPED'}")
    print(f"  Overall Status:       {'VERIFIED' if res.success else 'REJECTED'}")

    if not res.success:
        print(f"\nRejection Reason: {res.rejection_reason}")
        if res.error_message:
            print(f"Details: {res.error_message}")
        return 1

    print(f"\nCandidate artifact generated successfully: {res.generated_source_file}")
    print("================================================================================")
    return 0


def cmd_remote_run(args):
    """Executes workload benchmark on an authorized remote AMD machine or via mock protocol test."""
    from axiom.remote import SSHRemoteTransport, MockAMDTransport, RemoteAmdRunner, RemoteHostConfig

    print("================================================================================")
    print("          AXIOM 2.2 OPTIONAL REMOTE AMD EXECUTION ADAPTER                       ")
    print("================================================================================")

    if args.mock:
        print("[MODE: LOCAL MOCK TRANSPORT FOR PROTOCOL VERIFICATION]")
        print("Note: Simulating AMD EPYC Zen 4 Linux machine. No remote credentials required.")
        transport = MockAMDTransport()
    else:
        host = args.host or os.environ.get("AXIOM_REMOTE_HOST")
        user = args.user or os.environ.get("AXIOM_REMOTE_USER")
        if not host or not user:
            print("Error: Remote host and user must be specified via --host/--user or environment variables AXIOM_REMOTE_HOST / AXIOM_REMOTE_USER.", file=sys.stderr)
            print("Tip: Use '--mock' to test the remote execution protocol offline without hardware.", file=sys.stderr)
            return 1
        cfg = RemoteHostConfig(host=host, user=user, port=args.port, ssh_key_path=args.key)
        transport = SSHRemoteTransport(cfg)

    runner = RemoteAmdRunner(transport)
    try:
        probe = transport.probe_host()
        print(f"Target Host Reachable: {probe.is_reachable}")
        print(f"Remote Architecture:   {probe.architecture}")
        print(f"CPU Vendor / Model:    {probe.cpu_vendor} | {probe.cpu_model}")
        print(f"Authentic AMD Zen:     {'YES' if probe.is_authentic_amd else 'NO'}")
        print(f"AVX-512 Support:       {'YES' if probe.has_avx512 else 'NO'}")
        print(f"Remote Compiler:       {probe.compiler_version}")
        print("--------------------------------------------------------------------------------")

        if not probe.is_reachable:
            print(f"Error: Remote host is unreachable: {probe.error_message}", file=sys.stderr)
            return 1

        print(f"Dispatching experiment: Workload=[{args.workload}] Candidate=[{args.candidate}] Size=[{args.size}]...")
        res = runner.run_remote_experiment(
            workload_id=args.workload,
            candidate_id=args.candidate,
            size=args.size,
            warmup=args.warmup,
            runs=args.runs
        )

        print("\nREMOTE EXECUTION RESULTS:")
        print(f"  Run Identifier:       {res.run_id}")
        print(f"  Experiment SHA-256:   {res.spec_sha256}")
        print(f"  Execution Mode:       {'[MOCK SIMULATION - PROTOCOL TEST ONLY]' if res.is_mock else '[GENUINE HARDWARE EXECUTION]'}")
        print(f"  Remote Exit Code:     {res.exit_code}")
        print(f"  Correctness Status:   {'PASSED' if res.passed_correctness else 'FAILED'}")
        print(f"  Median Time:          {res.median_ms:.3f} ms")
        print(f"  Throughput:           {res.gflops:.2f} GFLOP/s")
        print("================================================================================")
        return 0
    except Exception as err:
        print(f"Remote execution failed: {err}", file=sys.stderr)
        return 1


def cmd_roofline(args):
    """Executes empirical Roofline analysis for a workload across problem sizes."""
    from axiom.roofline import RooflineAnalyzer
    from axiom.benchmark import BenchmarkEngine
    from axiom.env import get_environment_snapshot

    print("================================================================================")
    print("           AXIOM 2.2 EMPIRICAL ROOFLINE ANALYSIS LABORATORY                     ")
    print("================================================================================")

    env = get_environment_snapshot()
    analyzer = RooflineAnalyzer(
        architecture=env.hardware.architecture,
        cpu_model=env.hardware.cpu_brand
    )

    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else [128, 256, 512]
    candidate_ids = args.candidates.split(",") if args.candidates else ["reference", "interchanged", "neon", "multithreaded"]

    print(f"Workload:     {args.workload}")
    print(f"Candidates:   {candidate_ids}")
    print(f"Sizes:        {sizes}")
    print("Collecting empirical execution measurements...\n")

    be = BenchmarkEngine()
    suite = be.run_workload(
        workload_id=args.workload,
        candidate_ids=candidate_ids,
        sizes=sizes,
        warmup_runs=args.warmup,
        measurement_runs=args.runs
    )

    points = []
    for cid, sdict in suite.results_by_candidate.items():
        for sz, er in sdict.items():
            if er.correctness_passed:
                pt = analyzer.analyze_candidate(
                    workload_id=args.workload,
                    candidate_id=cid,
                    size=sz,
                    median_ms=er.median_ms,
                    stddev_ms=er.stddev_ms,
                    measured_gflops=er.gflops
                )
                points.append(pt)

    report = analyzer.generate_report(points)
    ascii_chart = analyzer.generate_ascii_plot(report)
    print(ascii_chart)

    # Save artifact
    out_dir = Path("artifacts/roofline")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"roofline_{args.workload}.json"
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\n[ARTIFACT] Empirical Roofline report saved to: {report_path}")
    print("================================================================================")
    return 0


def cmd_dashboard(args):
    print("Launching Axiom 2.2 Interactive Streamlit Dashboard on port 8501...")
    subprocess.run(["streamlit", "run", "dashboard/app.py", "--server.port", "8501", "--server.headless", "true"])


def main():
    parser = argparse.ArgumentParser(description="Axiom 2.2: Compiler-Aware Performance Optimization System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # env
    subparsers.add_parser("env", help="Probe and display hardware, compiler, and profiler environment")

    # workloads
    subparsers.add_parser("workloads", help="List registered workloads and candidates")

    # diagnose
    diag_p = subparsers.add_parser("diagnose", help="Inspect compiler diagnostics and generated assembly")
    diag_p.add_argument("workload", help="Workload ID (e.g. matmul, conv2d)")
    diag_p.add_argument("--candidate", help="Candidate ID filter (optional)")

    # diff-asm
    diff_p = subparsers.add_parser("diff-asm", help="Compare assembly instructions between two candidates")
    diff_p.add_argument("workload", help="Workload ID (e.g. matmul, conv2d)")
    diff_p.add_argument("cand1", help="Baseline candidate ID (e.g. reference)")
    diff_p.add_argument("cand2", help="Optimized candidate ID (e.g. interchanged, neon)")

    # run
    run_p = subparsers.add_parser("run", help="Run benchmark experiment, validate correctness, and record results")
    run_p.add_argument("workload", help="Workload ID (e.g. matmul, conv2d, ml_infer, memory)")
    run_p.add_argument("--sizes", help="Comma-separated input sizes")
    run_p.add_argument("--candidates", help="Comma-separated candidate IDs (optional)")
    run_p.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    run_p.add_argument("--runs", type=int, default=15, help="Measurement iterations")
    run_p.add_argument("--min-speedup", type=float, default=1.05, help="Minimum speedup threshold for recommendation")

    # regression-demo
    subparsers.add_parser("regression-demo", help="Demonstrate controlled performance regression detection")

    # multithread-sweep
    subparsers.add_parser("multithread-sweep", help="Benchmark multithreaded scaling across problem sizes")

    # memory-sweep
    subparsers.add_parser("memory-sweep", help="Benchmark memory working-set and cache locality")

    # ast-transform
    ast_p = subparsers.add_parser("ast-transform", help="Run Clang AST loop interchange with precondition verification")
    ast_p.add_argument("--source", default="cpp/workloads/matmul/matmul_reference.cpp", help="Source C++ file to analyze")
    ast_p.add_argument("--func", default="matmul_reference", help="Function name in source file")
    ast_p.add_argument("--candidate-id", default="matmul_ast_interchanged", help="Candidate identifier")
    ast_p.add_argument("--out", default="cpp/workloads/generated/matmul_ast_candidate.cpp", help="Generated candidate artifact path")

    # remote-run
    rem_p = subparsers.add_parser("remote-run", help="Execute benchmark on remote AMD x86-64 machine or local mock")
    rem_p.add_argument("--workload", default="matmul", help="Workload ID")
    rem_p.add_argument("--candidate", default="avx512", help="Candidate ID")
    rem_p.add_argument("--size", type=int, default=256, help="Problem dimension")
    rem_p.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    rem_p.add_argument("--runs", type=int, default=5, help="Measurement iterations")
    rem_p.add_argument("--host", help="Remote SSH hostname / IP")
    rem_p.add_argument("--user", help="Remote SSH username")
    rem_p.add_argument("--port", type=int, default=22, help="Remote SSH port")
    rem_p.add_argument("--key", help="Path to private SSH key file")
    rem_p.add_argument("--mock", action="store_true", help="Use local mock transport for protocol testing without remote hardware")

    # roofline
    roof_p = subparsers.add_parser("roofline", help="Run empirical Roofline analysis against hardware ceilings")
    roof_p.add_argument("--workload", default="matmul", help="Workload ID (default: matmul)")
    roof_p.add_argument("--sizes", default="128,256,512", help="Comma-separated input sizes")
    roof_p.add_argument("--candidates", default="reference,interchanged,neon,multithreaded", help="Candidates to measure")
    roof_p.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    roof_p.add_argument("--runs", type=int, default=5, help="Measurement iterations")

    # report
    rep_p = subparsers.add_parser("report", help="Generate Markdown report from existing experiment run")
    rep_p.add_argument("experiment_id", help="Experiment ID or timestamp")
    rep_p.add_argument("--output", "-o", help="Output file path (default: stdout)")

    # test
    subparsers.add_parser("test", help="Run full C++ and Python test suites")

    # dashboard
    subparsers.add_parser("dashboard", help="Launch interactive Streamlit dashboard")

    args = parser.parse_args()

    dispatch = {
        "env": cmd_env,
        "workloads": cmd_workloads,
        "diagnose": cmd_diagnose,
        "diff-asm": cmd_diff_asm,
        "run": cmd_run,
        "regression-demo": cmd_regression_demo,
        "multithread-sweep": cmd_multithread_sweep,
        "memory-sweep": cmd_memory_sweep,
        "ast-transform": cmd_ast_transform,
        "remote-run": cmd_remote_run,
        "roofline": cmd_roofline,
        "report": cmd_report,
        "test": cmd_test,
        "dashboard": cmd_dashboard
    }

    fn = dispatch.get(args.command)
    if fn:
        sys.exit(fn(args) or 0)


if __name__ == "__main__":
    main()
