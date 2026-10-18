"""
Workload Manager and Registry for Axiom 2.1.
Maintains trusted reference implementations, candidates, workload parameters,
source file hashing, and execution bounds.
"""

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class CandidateInfo:
    candidate_id: str
    name: str
    description: str
    source_file: str
    transformation_type: str
    source_hash: Optional[str] = None


@dataclass
class WorkloadDefinition:
    workload_id: str
    name: str
    domain: str
    description: str
    reference_file: str
    candidates: List[CandidateInfo]
    supported_sizes: List[int]
    default_size: int
    flop_formula: str
    unit_label: str = "GFLOP/s"
    tolerance_abs: float = 1e-4
    tolerance_rel: float = 1e-4
    timeout_seconds: int = 45
    reference_hash: Optional[str] = None


def compute_file_hash(filepath: str) -> str:
    """Computes SHA-256 hash of a source file."""
    p = Path(filepath)
    if not p.exists():
        return "file_not_found"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class WorkloadRegistry:
    def __init__(self, root_dir: Optional[str] = None):
        if root_dir is None:
            self.root_dir = Path(__file__).resolve().parent.parent
        else:
            self.root_dir = Path(root_dir)
        self._workloads: Dict[str, WorkloadDefinition] = {}
        self._register_default_workloads()

    def _register_default_workloads(self):
        # 1. Flagship Experiment: Matrix Multiplication (GEMM F32)
        matmul_ref = str(self.root_dir / "cpp" / "workloads" / "matmul" / "matmul_reference.cpp")
        matmul_interchanged = str(self.root_dir / "cpp" / "workloads" / "matmul" / "matmul_interchanged.cpp")
        matmul_blocked = str(self.root_dir / "cpp" / "workloads" / "matmul" / "matmul_blocked.cpp")
        matmul_neon = str(self.root_dir / "cpp" / "workloads" / "matmul" / "matmul_neon.cpp")
        matmul_mt = str(self.root_dir / "cpp" / "workloads" / "matmul" / "matmul_multithreaded.cpp")
        matmul_regressed = str(self.root_dir / "cpp" / "workloads" / "matmul" / "matmul_regressed.cpp")

        matmul = WorkloadDefinition(
            workload_id="matmul",
            name="General Matrix Multiplication (GEMM F32)",
            domain="Linear Algebra / BLAS",
            description="Square matrix multiplication C = A * B of size N x N with row-major layout.",
            reference_file=matmul_ref,
            candidates=[
                CandidateInfo(
                    candidate_id="reference",
                    name="Naive i-j-k Reference",
                    description="Standard 3-nested loop with strided column access on matrix B.",
                    source_file=matmul_ref,
                    transformation_type="baseline",
                    source_hash=compute_file_hash(matmul_ref)
                ),
                CandidateInfo(
                    candidate_id="interchanged",
                    name="Loop Interchanged i-k-j",
                    description="Loop order interchanged so inner loop runs over j, making accesses contiguous and vectorizable.",
                    source_file=matmul_interchanged,
                    transformation_type="loop_interchange",
                    source_hash=compute_file_hash(matmul_interchanged)
                ),
                CandidateInfo(
                    candidate_id="blocked",
                    name="Cache Blocked 64x64 i-k-j",
                    description="Loop tiling with 64x64 sub-blocks designed to reside within L1 Data Cache.",
                    source_file=matmul_blocked,
                    transformation_type="cache_blocking",
                    source_hash=compute_file_hash(matmul_blocked)
                ),
                CandidateInfo(
                    candidate_id="neon",
                    name="Explicit ARM NEON 4x4 Microkernel",
                    description="Hand-coded ARM NEON intrinsics using 4x4 register blocking and fused multiply-accumulate (vfmaq_f32).",
                    source_file=matmul_neon,
                    transformation_type="explicit_simd",
                    source_hash=compute_file_hash(matmul_neon)
                ),
                CandidateInfo(
                    candidate_id="multithreaded",
                    name="Multithreaded C++20 jthread",
                    description="Parallel row-block partitioning across available CPU cores using C++20 concurrency.",
                    source_file=matmul_mt,
                    transformation_type="multithreaded",
                    source_hash=compute_file_hash(matmul_mt)
                ),
                CandidateInfo(
                    candidate_id="regressed",
                    name="Controlled Inefficient Column-Stride",
                    description="Intentionally unoptimized j-k-i column-strided traversal to demonstrate regression detection.",
                    source_file=matmul_regressed,
                    transformation_type="controlled_regression",
                    source_hash=compute_file_hash(matmul_regressed)
                ),
                CandidateInfo(
                    candidate_id="avx512",
                    name="x86-64 AVX-512 FMA Vector Kernel (Safe Dispatched)",
                    description="AVX-512 512-bit ZMM intrinsics with runtime CPU feature gating and portable fallback.",
                    source_file=str(self.root_dir / "cpp" / "workloads" / "matmul" / "matmul_avx512.cpp"),
                    transformation_type="architecture_simd_avx512",
                    source_hash=compute_file_hash(str(self.root_dir / "cpp" / "workloads" / "matmul" / "matmul_avx512.cpp"))
                )
            ],
            supported_sizes=[64, 128, 256, 512, 1024],
            default_size=256,
            flop_formula="2 * N^3",
            unit_label="GFLOP/s",
            tolerance_abs=1e-4,
            tolerance_rel=1e-4,
            timeout_seconds=60,
            reference_hash=compute_file_hash(matmul_ref)
        )
        self.register(matmul)

        # 2. Workload: 2D Spatial Convolution (CONV2D F32)
        conv_ref = str(self.root_dir / "cpp" / "workloads" / "conv2d" / "conv2d_reference.cpp")
        conv_row = str(self.root_dir / "cpp" / "workloads" / "conv2d" / "conv2d_row_major.cpp")
        conv_tiled = str(self.root_dir / "cpp" / "workloads" / "conv2d" / "conv2d_tiled.cpp")

        conv2d = WorkloadDefinition(
            workload_id="conv2d",
            name="2D Spatial Convolution (CONV2D F32)",
            domain="Computer Vision / Signal Processing",
            description="2D Convolution of H x W input with 5x5 kernel using valid padding.",
            reference_file=conv_ref,
            candidates=[
                CandidateInfo(
                    candidate_id="reference",
                    name="Naive Spatial 2D Convolution",
                    description="Direct 4-nested loop iterating output pixels then kernel filter.",
                    source_file=conv_ref,
                    transformation_type="baseline",
                    source_hash=compute_file_hash(conv_ref)
                ),
                CandidateInfo(
                    candidate_id="row_major",
                    name="Kernel-Inverted Streaming",
                    description="Inverted loop ordering holding kernel weights in registers while streaming rows.",
                    source_file=conv_row,
                    transformation_type="loop_inversion_streaming",
                    source_hash=compute_file_hash(conv_row)
                ),
                CandidateInfo(
                    candidate_id="tiled",
                    name="Tiled Output Columns",
                    description="Column tiling in blocks of 64 floats to maximize cache line reuse.",
                    source_file=conv_tiled,
                    transformation_type="cache_blocking",
                    source_hash=compute_file_hash(conv_tiled)
                )
            ],
            supported_sizes=[64, 128, 256, 512],
            default_size=256,
            flop_formula="2 * (H - K + 1) * (W - K + 1) * K^2",
            unit_label="GFLOP/s",
            tolerance_abs=1e-4,
            tolerance_rel=1e-4,
            timeout_seconds=45,
            reference_hash=compute_file_hash(conv_ref)
        )
        self.register(conv2d)

        # 3. Workload: CPU ML Inference (MLP Feed-Forward F32)
        ml_ref = str(self.root_dir / "cpp" / "workloads" / "ml_infer" / "ml_mlp_infer.cpp")
        ml_infer = WorkloadDefinition(
            workload_id="ml_infer",
            name="CPU Neural Network MLP Inference (F32)",
            domain="Machine Learning / AI Infrastructure",
            description="3-layer feed-forward neural network inference (64 -> 128 -> 64 -> 10) with ReLU activations across batch sizes.",
            reference_file=ml_ref,
            candidates=[
                CandidateInfo(
                    candidate_id="reference",
                    name="Scalar Forward Pass Reference",
                    description="Direct nested batch and layer GEMV passes with scalar activations.",
                    source_file=ml_ref,
                    transformation_type="baseline",
                    source_hash=compute_file_hash(ml_ref)
                ),
                CandidateInfo(
                    candidate_id="vectorized",
                    name="Row-Streaming Vectorized Forward Pass",
                    description="Hoisted row-streaming dot-products with auto-vectorized ReLU activations.",
                    source_file=ml_ref,
                    transformation_type="vectorized_inference",
                    source_hash=compute_file_hash(ml_ref)
                )
            ],
            supported_sizes=[1, 8, 32, 64, 128],
            default_size=32,
            flop_formula="2 * Batch * (64*128 + 128*64 + 64*10)",
            unit_label="GFLOP/s",
            tolerance_abs=1e-4,
            tolerance_rel=1e-4,
            timeout_seconds=30,
            reference_hash=compute_file_hash(ml_ref)
        )
        self.register(ml_infer)

        # 4. Workload: Memory Locality & Working-Set Traversal
        mem_ref = str(self.root_dir / "cpp" / "workloads" / "memory" / "memory_locality.cpp")
        memory = WorkloadDefinition(
            workload_id="memory",
            name="Memory Working-Set & Locality Sweep",
            domain="Hardware-Software Co-Design",
            description="Evaluates memory bandwidth and cache line residency (L1, L2, DRAM) using sequential vs 128-byte strided traversals.",
            reference_file=mem_ref,
            candidates=[
                CandidateInfo(
                    candidate_id="sequential",
                    name="Sequential Unit-Stride Traversal",
                    description="Contiguous array scan utilizing hardware prefetcher and full 128-byte cache lines.",
                    source_file=mem_ref,
                    transformation_type="baseline",
                    source_hash=compute_file_hash(mem_ref)
                ),
                CandidateInfo(
                    candidate_id="strided",
                    name="Strided 128-byte Cache-Line Traversal",
                    description="Accesses 1 float per 128-byte cache line, testing capacity and prefetcher limits.",
                    source_file=mem_ref,
                    transformation_type="strided_access",
                    source_hash=compute_file_hash(mem_ref)
                )
            ],
            supported_sizes=[16, 64, 256, 1024, 4096, 16384],
            default_size=1024,
            flop_formula="Buffer_Bytes * Iterations",
            unit_label="GB/s",
            tolerance_abs=1e-3,
            tolerance_rel=1e-3,
            timeout_seconds=30,
            reference_hash=compute_file_hash(mem_ref)
        )
        self.register(memory)

    def register(self, workload: WorkloadDefinition):
        self._workloads[workload.workload_id] = workload

    def get(self, workload_id: str) -> Optional[WorkloadDefinition]:
        return self._workloads.get(workload_id)

    def list_workloads(self) -> List[WorkloadDefinition]:
        return list(self._workloads.values())
