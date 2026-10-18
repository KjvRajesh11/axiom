"""
Experiment and Results Store for Axiom 2.0.
Saves comprehensive JSON experiment archives preserving raw observations,
environment snapshots, compiler diagnostics, and decision verdicts.
Maintains tabular CSV summary for rapid analysis.
"""

import csv
import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Any


class ExperimentStore:
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path("results")
        self.runs_dir = self.base_dir / "runs"
        self.summary_csv = self.base_dir / "summary.csv"

        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_headers()

    def _ensure_csv_headers(self):
        if not self.summary_csv.exists():
            headers = [
                "experiment_id",
                "timestamp",
                "workload",
                "size",
                "candidate",
                "median_ms",
                "min_ms",
                "max_ms",
                "stddev_ms",
                "iqr_ms",
                "gflops",
                "correctness_passed",
                "speedup_vs_reference",
                "verdict",
                "cpu_brand",
                "compiler"
            ]
            with open(self.summary_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def save_run(self, experiment_record: Dict[str, Any]) -> str:
        """Saves experiment JSON document and appends rows to summary CSV."""
        exp_id = experiment_record.get("experiment_id") or f"exp_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        experiment_record["experiment_id"] = exp_id
        if "timestamp" not in experiment_record:
            experiment_record["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Save JSON
        json_path = self.runs_dir / f"{exp_id}.json"
        with open(json_path, "w") as f:
            json.dump(experiment_record, f, indent=2)

        # Append CSV rows for each candidate comparison
        cpu_brand = experiment_record.get("environment", {}).get("hardware", {}).get("cpu_brand", "Unknown CPU")
        compiler = experiment_record.get("environment", {}).get("toolchain", {}).get("compiler_version", "Clang")
        workload = experiment_record.get("workload_id", "unknown")
        timestamp = experiment_record["timestamp"]

        with open(self.summary_csv, "a", newline="") as f:
            writer = csv.writer(f)
            # Find candidate summaries
            candidates = experiment_record.get("candidate_results", {})
            decisions = experiment_record.get("decisions", {})

            for cand_id, size_dict in candidates.items():
                for size_str, exec_res in size_dict.items():
                    size = int(size_str)
                    dec_key = f"{cand_id}_{size}"
                    dec = decisions.get(dec_key, {})
                    speedup = dec.get("speedup", 1.0 if cand_id == "reference" else 0.0)
                    verdict = dec.get("verdict", "BASELINE" if cand_id == "reference" else "UNKNOWN")

                    writer.writerow([
                        exp_id,
                        timestamp,
                        workload,
                        size,
                        cand_id,
                        f"{exec_res.get('median_ms', 0.0):.4f}",
                        f"{exec_res.get('min_ms', 0.0):.4f}",
                        f"{exec_res.get('max_ms', 0.0):.4f}",
                        f"{exec_res.get('stddev_ms', 0.0):.4f}",
                        f"{exec_res.get('iqr_ms', 0.0):.4f}",
                        f"{exec_res.get('gflops', 0.0):.2f}",
                        exec_res.get("correctness_passed", False),
                        f"{speedup:.2f}",
                        verdict,
                        cpu_brand,
                        compiler
                    ])

        return str(json_path)

    def list_runs(self) -> List[Dict[str, Any]]:
        """Lists metadata for all saved runs."""
        runs = []
        for p in sorted(self.runs_dir.glob("*.json"), reverse=True):
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                runs.append({
                    "experiment_id": data.get("experiment_id", p.stem),
                    "timestamp": data.get("timestamp", "unknown"),
                    "workload_id": data.get("workload_id", "unknown"),
                    "path": str(p)
                })
            except Exception:
                pass
        return runs

    def load_run(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Loads a run record by ID."""
        json_path = self.runs_dir / f"{experiment_id}.json"
        if not json_path.exists():
            # Try searching by stem
            for p in self.runs_dir.glob("*.json"):
                if experiment_id in p.name:
                    json_path = p
                    break
        if not json_path.exists():
            return None
        with open(json_path, "r") as f:
            return json.load(f)
