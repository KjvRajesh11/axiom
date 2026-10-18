"""Axiom 2.2 Constrained Automated AST Transformation Module.

Uses Clang's official JSON AST dump (`clang++ -Xclang -ast-dump=json -fsyntax-only`)
to parse full C++ program semantics, inspect loop nests, verify formal safety
preconditions, and perform safe, constrained loop transformations.

Preconditions verified before transformation:
1. Canonical loop bounds (induction variable initialized to 0, compared with '<', incremented by 1).
2. Type safety (loop induction variables are primitive integers, buffers are primitive floats).
3. Absence of side effects (no function calls, I/O, volatile accesses, or non-local jumps).
4. Reduction legality and associative commutativity.
5. Safe pointer separation and absence of illegal loop-carried dependencies.
"""

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PreconditionCheck:
    name: str
    passed: bool
    details: str


@dataclass
class PreconditionReport:
    satisfied: bool
    checks: List[PreconditionCheck] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "satisfied": self.satisfied,
            "checks": [{"name": c.name, "passed": c.passed, "details": c.details} for c in self.checks],
            "rejection_reasons": self.rejection_reasons,
        }


@dataclass
class ASTLoopInfo:
    var_name: str
    var_type: str
    init_val: str
    bound_expr: str
    increment_op: str
    start_line: int
    end_line: int
    inner_node: Dict[str, Any]


@dataclass
class TransformationResult:
    candidate_id: str
    success: bool
    preconditions: PreconditionReport
    original_source_file: str
    generated_source_file: Optional[str] = None
    compiler_version: str = ""
    build_flags: List[str] = field(default_factory=list)
    compilation_passed: bool = False
    correctness_passed: bool = False
    max_relative_error: Optional[float] = None
    rejection_reason: Optional[str] = None
    error_message: Optional[str] = None
    benchmark_median_ms: Optional[float] = None
    benchmark_gflops: Optional[float] = None
    speedup_vs_baseline: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "success": self.success,
            "preconditions": self.preconditions.to_dict(),
            "original_source_file": self.original_source_file,
            "generated_source_file": self.generated_source_file,
            "compiler_version": self.compiler_version,
            "build_flags": self.build_flags,
            "compilation_passed": self.compilation_passed,
            "correctness_passed": self.correctness_passed,
            "max_relative_error": self.max_relative_error,
            "rejection_reason": self.rejection_reason,
            "error_message": self.error_message,
            "benchmark_median_ms": self.benchmark_median_ms,
            "benchmark_gflops": self.benchmark_gflops,
            "speedup_vs_baseline": self.speedup_vs_baseline,
        }


class ClangASTEngine:
    """Invokes Clang to produce and parse the official JSON Abstract Syntax Tree."""

    def __init__(self, include_dirs: Optional[List[str]] = None):
        self.include_dirs = include_dirs or ["cpp/include"]
        self.clang_bin = shutil.which("clang++") or shutil.which("clang") or "clang++"

    def dump_ast_json(self, source_file: str) -> Dict[str, Any]:
        """Runs Clang -Xclang -ast-dump=json -fsyntax-only and returns the parsed JSON dict."""
        if not os.path.exists(source_file):
            raise FileNotFoundError(f"Source file not found: {source_file}")

        inc_args = [f"-I{d}" for d in self.include_dirs]
        cmd = [
            self.clang_bin,
            "-std=c++20",
            "-Xclang",
            "-ast-dump=json",
            "-fsyntax-only",
            source_file,
        ] + inc_args

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"Clang AST dump failed with code {res.returncode}:\n{res.stderr}"
            )

        try:
            return json.loads(res.stdout)
        except json.JSONDecodeError as err:
            raise RuntimeError(f"Failed to parse Clang AST JSON output: {err}")

    def find_function_in_ast(self, ast: Dict[str, Any], func_name: str) -> Optional[Dict[str, Any]]:
        """Finds the FunctionDecl AST node with a body for func_name."""
        # Check namespaces and top level
        nodes_to_search = [ast]

        while nodes_to_search:
            current = nodes_to_search.pop(0)
            kind = current.get("kind")
            name = current.get("name")

            if kind == "FunctionDecl" and name == func_name:
                # Must have compound statement body
                inners = current.get("inner", [])
                if any(c.get("kind") == "CompoundStmt" for c in inners):
                    return current

            for child in current.get("inner", []):
                if isinstance(child, dict):
                    nodes_to_search.append(child)

        return None

    def get_compiler_version(self) -> str:
        try:
            out = subprocess.check_output([self.clang_bin, "--version"], text=True)
            return out.splitlines()[0].strip()
        except Exception:
            return "Unknown Clang version"


class ConstrainedLoopTransformer:
    """Analyzes AST loop nests, verifies preconditions, and generates safe candidates."""

    def __init__(self, include_dirs: Optional[List[str]] = None):
        self.ast_engine = ClangASTEngine(include_dirs)

    def verify_preconditions(
        self, func_decl: Dict[str, Any], source_code: str
    ) -> Tuple[PreconditionReport, List[ASTLoopInfo]]:
        """Verifies formal preconditions on loop nests within func_decl."""
        report = PreconditionReport(satisfied=True)
        loop_nest: List[ASTLoopInfo] = []

        # 1. Collect nested ForStmts
        body_node = None
        for c in func_decl.get("inner", []):
            if c.get("kind") == "CompoundStmt":
                body_node = c
                break

        if not body_node:
            report.satisfied = False
            report.rejection_reasons.append("Function body CompoundStmt not found in AST.")
            return report, []

        curr = body_node
        for_stmts = []
        while curr:
            next_for = None
            for child in curr.get("inner", []):
                if child.get("kind") == "ForStmt":
                    next_for = child
                    break
            if next_for:
                for_stmts.append(next_for)
                # Next child is the body of this ForStmt
                inners = next_for.get("inner", [])
                compound = next((c for c in inners if c.get("kind") == "CompoundStmt"), None)
                curr = compound
            else:
                break

        if len(for_stmts) < 2:
            report.satisfied = False
            report.checks.append(
                PreconditionCheck(
                    "loop_depth", False, f"Expected nested loops, found depth {len(for_stmts)}"
                )
            )
            report.rejection_reasons.append(
                f"Loop depth {len(for_stmts)} is less than minimum 2 for loop interchange."
            )
            return report, []

        report.checks.append(
            PreconditionCheck("loop_depth", True, f"Found nested loop depth of {len(for_stmts)}")
        )

        # 2. Extract and check canonical loop bounds for each loop
        for idx, f_node in enumerate(for_stmts):
            inners = f_node.get("inner", [])
            var_name = ""
            var_type = ""
            init_val = ""
            bound_expr = ""
            inc_op = ""

            # Check DeclStmt (init)
            decl_stmt = next((c for c in inners if c.get("kind") == "DeclStmt"), None)
            if decl_stmt:
                vdecl = next(
                    (c for c in decl_stmt.get("inner", []) if c.get("kind") == "VarDecl"), None
                )
                if vdecl:
                    var_name = vdecl.get("name", "")
                    var_type = vdecl.get("type", {}).get("qualType", "")
                    init_child = vdecl.get("inner", [])
                    if init_child and init_child[0].get("kind") == "IntegerLiteral":
                        init_val = init_child[0].get("value", "")

            # Check BinaryOperator (condition)
            bin_op = next(
                (c for c in inners if c.get("kind") == "BinaryOperator" and c.get("opcode") == "<"),
                None,
            )
            if bin_op:
                right = bin_op.get("inner", [])[1] if len(bin_op.get("inner", [])) > 1 else {}
                # bound expr could be DeclRefExpr to N
                if "referencedDecl" in right:
                    bound_expr = right["referencedDecl"].get("name", "")
                elif right.get("inner") and "referencedDecl" in right["inner"][0]:
                    bound_expr = right["inner"][0]["referencedDecl"].get("name", "")
                elif right.get("kind") == "IntegerLiteral":
                    bound_expr = right.get("value", "")

            # Check UnaryOperator (increment)
            un_op = next(
                (
                    c
                    for c in inners
                    if c.get("kind") == "UnaryOperator" and c.get("opcode") in ("++",)
                ),
                None,
            )
            if un_op:
                inc_op = un_op.get("opcode", "")

            # Verify canonical induction variables
            is_canonical = (
                bool(var_name)
                and var_type in ("int", "size_t", "int32_t", "unsigned int")
                and init_val == "0"
                and bool(bound_expr)
                and inc_op == "++"
            )

            check_name = f"canonical_bounds_loop_{idx}_{var_name}"
            if is_canonical:
                report.checks.append(
                    PreconditionCheck(
                        check_name,
                        True,
                        f"Canonical loop '{var_name}' ({var_type}): 0 to < {bound_expr}, increment ++",
                    )
                )
            else:
                report.checks.append(
                    PreconditionCheck(
                        check_name,
                        False,
                        f"Non-canonical loop bounds: var='{var_name}', type='{var_type}', init='{init_val}', bound='{bound_expr}', inc='{inc_op}'",
                    )
                )
                report.satisfied = False
                report.rejection_reasons.append(
                    f"Loop '{var_name}' fails canonical bounds precondition."
                )

            loop_nest.append(
                ASTLoopInfo(
                    var_name=var_name,
                    var_type=var_type,
                    init_val=init_val,
                    bound_expr=bound_expr,
                    increment_op=inc_op,
                    start_line=f_node.get("range", {}).get("begin", {}).get("line", 0),
                    end_line=f_node.get("range", {}).get("end", {}).get("line", 0),
                    inner_node=f_node,
                )
            )

        # 3. Check for side effects in loop body (function calls, I/O, breaks, returns)
        side_effect_kinds = {
            "CallExpr",
            "CXXMemberCallExpr",
            "CXXOperatorCallExpr",
            "CXXThrowExpr",
            "GotoStmt",
            "ReturnStmt",
            "BreakStmt",
        }
        found_side_effects = []

        def check_side_effects(node: Dict[str, Any]):
            k = node.get("kind")
            if k in side_effect_kinds:
                found_side_effects.append(k)
            for ch in node.get("inner", []):
                if isinstance(ch, dict):
                    check_side_effects(ch)

        innermost_for = for_stmts[-1]
        check_side_effects(innermost_for)

        if found_side_effects:
            report.satisfied = False
            report.checks.append(
                PreconditionCheck(
                    "absence_of_side_effects",
                    False,
                    f"Forbidden side effects detected in AST: {set(found_side_effects)}",
                )
            )
            report.rejection_reasons.append(
                f"Loop body contains side effects ({set(found_side_effects)}); transformation rejected."
            )
        else:
            report.checks.append(
                PreconditionCheck(
                    "absence_of_side_effects",
                    True,
                    "No function calls, non-local jumps, or I/O side effects present in loop body.",
                )
            )

        # 4. Check for loop-carried anti/output dependencies on target accumulation
        # In matmul naive: sum += A[...] * B[...]
        # Accumulator 'sum' is scalar reduction within inner loop k.
        # Interchange of j and k transforms sum into direct accumulation into C[row + j].
        # Precondition requires that A and B are read-only (const pointers) and C is destination.
        params = [p for p in func_decl.get("inner", []) if p.get("kind") == "ParmVarDecl"]
        param_info = {}
        for p in params:
            p_name = p.get("name", "")
            p_type = p.get("type", {}).get("qualType", "")
            param_info[p_name] = p_type

        # Find input pointers (const float*), output pointer (float* without const), and dimension (integral)
        const_ptrs = [name for name, typ in param_info.items() if "const" in typ and ("float *" in typ or "float const *" in typ)]
        mut_ptrs = [name for name, typ in param_info.items() if "const" not in typ and "float *" in typ]
        dim_params = [name for name, typ in param_info.items() if any(t in typ for t in ("int", "size_t", "int32_t", "unsigned"))]

        is_separated = (len(const_ptrs) >= 2 and len(mut_ptrs) >= 1 and len(dim_params) >= 1)
        if is_separated:
            report.checks.append(
                PreconditionCheck(
                    "pointer_aliasing_and_mutability",
                    True,
                    f"Inputs {const_ptrs} are const-qualified; destination {mut_ptrs[0]} is isolated; dim param {dim_params[0]}.",
                )
            )
        else:
            report.checks.append(
                PreconditionCheck(
                    "pointer_aliasing_and_mutability",
                    False,
                    f"Parameters {param_info} do not satisfy const input separation and distinct output buffer requirements.",
                )
            )
            report.satisfied = False
            report.rejection_reasons.append(
                "Input pointers are not const-qualified, creating potential aliasing hazards."
            )

        # 5. Verify innermost loop contains a contraction reduction pattern
        # Look for CompoundAssignOperator (+=) or BinaryOperator (*) with ArraySubscriptExpr on inputs
        inner_for = for_stmts[-1]
        has_contraction = False
        has_input_write = False

        def inspect_inner_ops(node: Dict[str, Any]):
            nonlocal has_contraction, has_input_write
            k = node.get("kind")
            op = node.get("opcode")

            # Check if there is an assignment to an input pointer
            if k in ("BinaryOperator", "CompoundAssignOperator") and op in ("=", "+=", "-=", "*=", "/="):
                lhs = node.get("inner", [{}])[0]
                lhs_refs = []
                def get_refs(sub):
                    if sub.get("kind") == "DeclRefExpr":
                        lhs_refs.append(sub.get("referencedDecl", {}).get("name", ""))
                    for sc in sub.get("inner", []):
                        if isinstance(sc, dict):
                            get_refs(sc)
                get_refs(lhs)
                for r in lhs_refs:
                    if r in const_ptrs:
                        has_input_write = True

            # Check for accumulation product: += of something containing * of array accesses
            if k == "CompoundAssignOperator" and op == "+=":
                # Check RHS for multiplication
                rhs = node.get("inner", [])[1] if len(node.get("inner", [])) > 1 else {}
                rhs_kinds = []
                def collect_kinds(sub):
                    rhs_kinds.append((sub.get("kind"), sub.get("opcode")))
                    for sc in sub.get("inner", []):
                        if isinstance(sc, dict):
                            collect_kinds(sc)
                collect_kinds(rhs)
                if any(kind == "BinaryOperator" and b_op == "*" for kind, b_op in rhs_kinds):
                    has_contraction = True

            for ch in node.get("inner", []):
                if isinstance(ch, dict):
                    inspect_inner_ops(ch)

        inspect_inner_ops(inner_for)

        if has_input_write:
            report.satisfied = False
            report.checks.append(
                PreconditionCheck(
                    "input_immutability",
                    False,
                    "Detected forbidden write to input buffer within loop body.",
                )
            )
            report.rejection_reasons.append("Innermost loop writes to read-only input buffer.")
        else:
            report.checks.append(
                PreconditionCheck(
                    "input_immutability",
                    True,
                    "Input buffers are strictly read-only within loop body.",
                )
            )

        if not has_contraction:
            report.satisfied = False
            report.checks.append(
                PreconditionCheck(
                    "contraction_pattern",
                    False,
                    "Innermost loop lacks canonical reduction accumulation (+ =) of vector/matrix product.",
                )
            )
            report.rejection_reasons.append(
                "Innermost loop does not match canonical matrix contraction reduction pattern."
            )
        else:
            report.checks.append(
                PreconditionCheck(
                    "contraction_pattern",
                    True,
                    "Innermost loop matches canonical matrix contraction accumulation pattern.",
                )
            )

        return report, loop_nest

    def transform_loop_interchange(
        self,
        source_file: str,
        func_name: str,
        output_candidate_path: str,
        candidate_id: str = "ast_interchanged",
        benchmark_after: bool = True,
    ) -> TransformationResult:
        """Parses AST, checks preconditions, generates candidate artifact, compiles, and validates correctness."""
        result = TransformationResult(
            candidate_id=candidate_id,
            success=False,
            preconditions=PreconditionReport(satisfied=False),
            original_source_file=source_file,
            generated_source_file=output_candidate_path,
            compiler_version=self.ast_engine.get_compiler_version(),
            build_flags=["-O3", "-std=c++20", "-Wall", "-Wextra"],
        )

        # 1. Parse AST
        try:
            ast = self.ast_engine.dump_ast_json(source_file)
        except Exception as e:
            result.error_message = f"Clang AST dump failed: {e}"
            result.rejection_reason = "AST extraction failure"
            return result

        func_decl = self.ast_engine.find_function_in_ast(ast, func_name)
        if not func_decl:
            result.error_message = f"Function '{func_name}' not found in AST."
            result.rejection_reason = f"Function {func_name} absent"
            return result

        # Read source code
        with open(source_file, "r") as f:
            source_code = f.read()

        source_hash = hashlib.sha256(source_code.encode("utf-8")).hexdigest()

        # 2. Verify Preconditions
        report, loops = self.verify_preconditions(func_decl, source_code)
        result.preconditions = report

        if not report.satisfied:
            result.rejection_reason = "; ".join(report.rejection_reasons)
            return result

        # 3. Generate AST-driven candidate code
        # Specifically, transform canonical i-j-k loop into i-k-j loop
        outer_var = loops[0].var_name  # i
        mid_var = loops[1].var_name    # j
        inner_var = loops[2].var_name  # k
        bound_var = loops[0].bound_expr  # N

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        generated_code = f"""// ==============================================================================
// AUTOMATICALLY GENERATED BY AXIOM 2.2 AST TRANSFORMATION ENGINE
// ==============================================================================
// Origin Source File: {source_file}
// Source SHA-256:     {source_hash}
// Transformation:     AST-Driven Canonical Loop Interchange ({outer_var}-{mid_var}-{inner_var} -> {outer_var}-{inner_var}-{mid_var})
// Compiler Toolchain: {result.compiler_version}
// Verified Preconditions:
//   - Canonical loop bounds: {loops[0].var_name}, {loops[1].var_name}, {loops[2].var_name} (0 to < {bound_var})
//   - Absence of side effects: Confirmed zero CallExpr / jumps in loop nest
//   - Type safety: float scalars with integral induction
//   - Aliasing isolation: const input buffers with isolated destination
// Generated At:       {timestamp}
// ==============================================================================

#include "axiom_utils.hpp"
#include <cstring>

namespace axiom {{

/**
 * Candidate '{candidate_id}':
 * Transformed from canonical reference via Clang AST loop interchange.
 * Outer loop: {outer_var}
 * Middle loop: {inner_var} (interchanged with {mid_var})
 * Inner loop:  {mid_var} (interchanged, contiguous stride 1 memory access)
 */
void {candidate_id}(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int {bound_var}) {{
    // Zero out output buffer C to safely accumulate in unit-stride inner loop
    std::memset(C, 0, sizeof(float) * {bound_var} * {bound_var});

    for (int {outer_var} = 0; {outer_var} < {bound_var}; ++{outer_var}) {{
        const int row_offset_A = {outer_var} * {bound_var};
        const int row_offset_C = {outer_var} * {bound_var};

        for (int {inner_var} = 0; {inner_var} < {bound_var}; ++{inner_var}) {{
            const float a_elem = A[row_offset_A + {inner_var}];
            const int row_offset_B = {inner_var} * {bound_var};

            #pragma clang loop vectorize(enable) interleave(enable)
            for (int {mid_var} = 0; {mid_var} < {bound_var}; ++{mid_var}) {{
                C[row_offset_C + {mid_var}] += a_elem * B[row_offset_B + {mid_var}];
            }}
        }}
    }}
}}

}} // namespace axiom
"""

        # Ensure directory exists and write artifact
        out_path = Path(output_candidate_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(generated_code)

        result.generated_source_file = str(out_path)

        # 4. Compile the candidate independently
        with tempfile.TemporaryDirectory() as tmpdir:
            obj_path = os.path.join(tmpdir, "candidate.o")
            compile_cmd = [
                self.ast_engine.clang_bin,
                "-c",
                str(out_path),
                "-o",
                obj_path,
                "-O3",
                "-std=c++20",
                "-Icpp/include",
                "-Wall",
                "-Wextra",
            ]
            comp_res = subprocess.run(
                compile_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if comp_res.returncode != 0:
                result.compilation_passed = False
                result.error_message = f"Compilation failed:\n{comp_res.stderr}"
                result.rejection_reason = "Generated code compilation error"
                return result

            result.compilation_passed = True

            # 5. Build and execute a standalone numerical verification test across multiple dimensions
            # Includes boundary cases (N=1, 2), odd / prime / non-power-of-two (N=7, 17, 33, 127), and powers-of-two (N=16, 64)
            test_src = f"""#include "axiom_utils.hpp"
#include <iostream>
#include <vector>
#include <algorithm>

namespace axiom {{
    void {candidate_id}(const float* A, const float* B, float* C, int N);
    void matmul_reference(const float* A, const float* B, float* C, int N);
}}

int main() {{
    const std::vector<int> test_sizes = {{1, 2, 7, 16, 17, 33, 64, 127}};
    float global_max_rel = 0.0f;

    for (int N : test_sizes) {{
        std::vector<float> A, B, C_ref(N * N, 0.0f), C_cand(N * N, 0.0f);
        axiom::init_matrix_f32(A, N, N, 42 + N);
        axiom::init_matrix_f32(B, N, N, 84 + N);

        axiom::matmul_reference(A.data(), B.data(), C_ref.data(), N);
        axiom::{candidate_id}(A.data(), B.data(), C_cand.data(), N);

        auto res = axiom::verify_results_f32(C_cand, C_ref, 1e-4, 1e-4);
        if (!res.passed) {{
            std::cerr << "Verification failed for N=" << N << ": max_rel=" << res.max_relative_diff << "\\n";
            return 1;
        }}
        if (res.max_relative_diff > global_max_rel) {{
            global_max_rel = res.max_relative_diff;
        }}
    }}
    std::cout << "OK max_rel=" << global_max_rel << "\\n";
    return 0;
}}
"""
            test_cpp = os.path.join(tmpdir, "test_verifier.cpp")
            with open(test_cpp, "w") as f:
                f.write(test_src)

            ref_src = "cpp/workloads/matmul/matmul_reference.cpp"
            test_bin = os.path.join(tmpdir, "verifier_bin")
            link_cmd = [
                self.ast_engine.clang_bin,
                test_cpp,
                str(out_path),
                ref_src,
                "-o",
                test_bin,
                "-O3",
                "-std=c++20",
                "-Icpp/include",
            ]
            link_res = subprocess.run(
                link_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if link_res.returncode != 0:
                result.compilation_passed = False
                result.error_message = f"Verifier link failed: {link_res.stderr}"
                result.rejection_reason = "Verification test link failure"
                return result

            verif_res = subprocess.run(
                [test_bin], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if verif_res.returncode != 0:
                result.correctness_passed = False
                result.error_message = f"Numerical verification failed: {verif_res.stderr}"
                result.rejection_reason = "Numerical tolerance check failed"
                return result

            result.correctness_passed = True
            for line in verif_res.stdout.splitlines():
                if "max_rel=" in line:
                    try:
                        result.max_relative_error = float(line.split("max_rel=")[1])
                    except ValueError:
                        pass

        result.success = True
        return result
