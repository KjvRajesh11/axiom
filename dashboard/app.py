"""
Axiom 2.1 Interactive Performance Engineering Dashboard.
Visualizes compiler diagnostics, benchmark statistics, multi-core scaling,
ARM NEON vs auto-vectorization, memory locality sweeps, and regression detection.
"""

import json
import os
import pandas as pd
import streamlit as st
from pathlib import Path

from axiom.env import get_environment_snapshot
from axiom.workload import WorkloadRegistry
from axiom.compiler import CompilerIntelligence
from axiom.store import ExperimentStore
from axiom.report import ReportGenerator

st.set_page_config(
    page_title="Axiom 2.1 | Advanced Systems Performance Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚡ Axiom 2.1: Systems Performance Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Compiler-aware hardware co-design, multithread scaling, explicit SIMD, and empirical regression detection.</div>', unsafe_allow_html=True)

# ----------------- SIDEBAR: HARDWARE & TOOLCHAIN -----------------
env_snap = get_environment_snapshot()
hw = env_snap.hardware
tc = env_snap.toolchain
prof = env_snap.profilers

st.sidebar.title("System Hardware & Toolchain")
st.sidebar.markdown(f"**CPU:** `{hw.cpu_brand}` ({hw.architecture})")
st.sidebar.markdown(f"**Cores:** {hw.physical_cores} Physical / {hw.logical_cores} Logical")
st.sidebar.markdown(f"**Cache Line:** {hw.cache_line_bytes} Bytes")
l1_kb = (hw.l1d_cache_bytes // 1024) if hw.l1d_cache_bytes else "Unknown"
l2_mb = (hw.l2_cache_bytes // (1024 * 1024)) if hw.l2_cache_bytes else "Unknown"
st.sidebar.markdown(f"**L1 Data Cache:** {l1_kb} KB")
st.sidebar.markdown(f"**L2 Cache:** {l2_mb} MB")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Compiler:** `{tc.compiler_version.splitlines()[0] if tc.compiler_version else 'Clang'}`")
st.sidebar.markdown(f"**Target Triple:** `{tc.target_triple}`")
st.sidebar.markdown(f"**CMake:** `{tc.cmake_version or 'N/A'}`")
st.sidebar.markdown("---")
st.sidebar.markdown("**Profiling Capabilities:**")
st.sidebar.markdown(f"- `xctrace`: {'✅' if prof.has_xctrace else '❌'}")
st.sidebar.markdown(f"- `sample`: {'✅' if prof.has_sample else '❌'}")
st.sidebar.markdown(f"- `llvm-mca`: {'✅' if prof.has_llvm_mca else '❌ (Tool absent on PATH)'}")

# ----------------- STORE & EXPERIMENT SELECTION -----------------
store = ExperimentStore()
saved_runs = store.list_runs()

if not saved_runs:
    st.warning("No experiment records found in results/runs. Run `./axiom_cli.py run matmul` from the CLI to generate data.")
    st.stop()

run_options = {f"{r['experiment_id']} ({r['workload_id']} @ {r['timestamp']})": r["experiment_id"] for r in saved_runs}
selected_label = st.selectbox("Select Experiment Run Archive:", list(run_options.keys()))
selected_exp_id = run_options[selected_label]
record = store.load_run(selected_exp_id)

if not record:
    st.error(f"Failed to load record for {selected_exp_id}")
    st.stop()

# ----------------- MAIN TABS -----------------
tab_perf, tab_diag, tab_hyps, tab_scaling, tab_report = st.tabs([
    "📈 Performance & Speedups",
    "🔍 Compiler Diagnostics & Diff",
    "🧪 Hypotheses & Evidence",
    "⚡ Multi-Core & Co-Design",
    "📄 Engineering Report"
])

# TAB 1: PERFORMANCE
with tab_perf:
    st.subheader("Performance Comparison Across Input Sizes")

    comparisons = record.get("comparisons", [])
    if comparisons:
        df_comp = pd.DataFrame(comparisons)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Workload", record.get("workload_id", "N/A").upper())
        max_speedup = df_comp["speedup"].max()
        best_cand = df_comp.loc[df_comp["speedup"].idxmax()]["candidate_id"]
        col2.metric("Peak Speedup vs Baseline", f"{max_speedup:.2f}x")
        col3.metric("Top Recommended Candidate", best_cand)

        # Tabular Data with research percentiles
        st.markdown("### Measurement Data & Research Statistics")
        table_cols = ["size", "candidate_id", "baseline_median_ms", "candidate_median_ms", "candidate_p95_ms", "candidate_p99_ms", "speedup", "candidate_gflops", "correctness_passed"]
        display_df = df_comp[[c for c in table_cols if c in df_comp.columns]].copy()
        display_df.columns = ["Size", "Candidate", "Base (ms)", "Cand Median (ms)", "p95 (ms)", "p99 (ms)", "Speedup", "Throughput", "Correctness"]
        st.dataframe(display_df.style.format({
            "Base (ms)": "{:.3f}",
            "Cand Median (ms)": "{:.3f}",
            "p95 (ms)": "{:.3f}",
            "p99 (ms)": "{:.3f}",
            "Speedup": "{:.2f}x",
            "Throughput": "{:.2f}"
        }), use_container_width=True)

        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            st.markdown("#### Speedup vs Reference Baseline (Higher is Better)")
            chart_data = df_comp.pivot(index="size", columns="candidate_id", values="speedup")
            st.bar_chart(chart_data)

        with c_chart2:
            st.markdown("#### Execution Latency (ms) (Lower is Better)")
            chart_data_lat = df_comp.pivot(index="size", columns="candidate_id", values="candidate_median_ms")
            chart_data_lat["reference"] = df_comp.groupby("size")["baseline_median_ms"].first()
            st.line_chart(chart_data_lat)

# TAB 2: COMPILER DIAGNOSTICS & ASSEMBLY DIFF
with tab_diag:
    st.subheader("Compiler Intelligence & Generated ARM64 Assembly")
    diagnostics = record.get("compiler_diagnostics", {})

    if diagnostics:
        for cid, diag in diagnostics.items():
            with st.expander(f"Candidate: [{cid}]", expanded=(cid == "interchanged" or cid == "neon")):
                st.markdown(f"**Diagnostic Summary:** {diag.get('summary_explanation')}")
                stats = diag.get("assembly_stats", {})
                if stats:
                    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
                    s_col1.metric("Total Lines", stats.get("total_lines", 0))
                    s_col2.metric("Vector FMLA", stats.get("vector_fmla_count", 0))
                    s_col3.metric("Memory Loads", stats.get("load_count", 0))
                    s_col4.metric("Branches", stats.get("branch_count", 0))

# TAB 3: HYPOTHESES & EVIDENCE
with tab_hyps:
    st.subheader("Testable Hypotheses & Evidence Evaluation")
    hypotheses = record.get("hypotheses", [])
    if hypotheses:
        for h in hypotheses:
            status = h.get("status", "PROPOSED")
            badge_color = "green" if status == "CONFIRMED" else ("red" if status == "WEAKENED" else "orange")
            regime = h.get("bound_regime", "UNKNOWN")
            st.markdown(f"#### :{badge_color}[[{status}]] `{regime}` {h.get('hypothesis_id')}: {h.get('category')}")
            st.markdown(f"**Statement:** {h.get('statement')}")
            st.markdown(f"**Uncertainty:** `{h.get('uncertainty')}`")
            st.markdown(f"**Confirmation Rule:** {h.get('confirmation_criteria')}")
            st.markdown(f"**Outcome:** *{h.get('evaluation_notes')}*")
            st.markdown("---")

# TAB 4: MULTI-CORE & CO-DESIGN
with tab_scaling:
    st.subheader("Hardware-Software Co-Design Laboratories")
    st.markdown("""
    Explore specialized hardware execution experiments from the CLI:
    - **Multithreaded Parallel Scaling:** `./axiom_cli.py multithread-sweep`
    - **Memory Locality & Working-Set Sweep:** `./axiom_cli.py memory-sweep`
    - **Explicit ARM NEON vs Compiler Vectorization:** `./axiom_cli.py run matmul --candidates reference,interchanged,neon,multithreaded`
    - **Controlled Regression Demonstration:** `./axiom_cli.py regression-demo`
    """)

# TAB 5: ENGINEERING REPORT
with tab_report:
    st.subheader("Automated 10-Section Engineering Research Report")
    rep_gen = ReportGenerator()
    md_report = rep_gen.generate_markdown_report(record, include_ai_synthesis=True)
    st.markdown(md_report)
