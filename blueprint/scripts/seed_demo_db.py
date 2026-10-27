# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed a v1-schema demo SQLite DB so the webui has data to render.

Mirrors the schema in src/attribution_blueprint/connectors/outputs/sqlite/__init__.py
on main as of 2026-05-24 (adds terminal_state + sub_domain columns).

Run:
    uv run python scripts/seed_demo_db.py --out demo.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA = [
    "CREATE TABLE IF NOT EXISTS schema_version (version TEXT PRIMARY KEY, description TEXT, applied_at TEXT NOT NULL)",
    """CREATE TABLE IF NOT EXISTS execution_context (
        id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL,
        attempt_id TEXT,
        scheduler_type TEXT NOT NULL,
        start_time TEXT NOT NULL,
        end_time TEXT NOT NULL,
        nodes_json TEXT NOT NULL,
        status TEXT,
        workload_framework TEXT,
        user_name TEXT,
        raw_scheduler_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        UNIQUE(job_id, attempt_id, scheduler_type)
    )""",
    """CREATE TABLE IF NOT EXISTS attribution_run (
        id TEXT PRIMARY KEY,
        execution_context_id TEXT NOT NULL REFERENCES execution_context(id),
        schema_version TEXT NOT NULL,
        pipeline_version TEXT,
        service_api_version TEXT,
        ruleset_version TEXT,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        outcome TEXT NOT NULL,
        terminal_state TEXT,
        sub_domain TEXT,
        raw_result_json TEXT NOT NULL,
        responsible_nodes_json TEXT NOT NULL DEFAULT '[]'
    )""",
    "CREATE INDEX IF NOT EXISTS idx_execution_context_job_id ON execution_context(job_id)",
    "CREATE INDEX IF NOT EXISTS idx_run_outcome ON attribution_run(outcome)",
]


_COMPOUND_FIELDS: dict[str, dict] = {
    "Fail InfraHW":                  {"terminal_states": ["Fail"], "failure_domains": ["InfraHW"],             "cancellation_actors": [], "uncertainty_type": None},
    "Fail InfraSW":                  {"terminal_states": ["Fail"], "failure_domains": ["InfraSW"],             "cancellation_actors": [], "uncertainty_type": None},
    "Fail User":                     {"terminal_states": ["Fail"], "failure_domains": ["User"],                "cancellation_actors": [], "uncertainty_type": None},
    "Fail MultiCategory":            {"terminal_states": ["Fail"], "failure_domains": ["User","InfraHW","InfraSW"], "cancellation_actors": [], "uncertainty_type": "Aleatoric"},
    "Fail Undetermined":             {"terminal_states": ["Fail"], "failure_domains": ["User","InfraHW","InfraSW"], "cancellation_actors": [], "uncertainty_type": "Epistemic"},
    "Pass":                          {"terminal_states": ["Pass"], "failure_domains": [],                      "cancellation_actors": [], "uncertainty_type": None},
    "Artificial":                    {"terminal_states": ["ExpectedFail"], "failure_domains": [],              "cancellation_actors": [], "uncertainty_type": None},
    "Cancelled by User":             {"terminal_states": ["Cancelled"], "failure_domains": [],                 "cancellation_actors": ["User"], "uncertainty_type": None},
    "Cancelled by cluster operators":{"terminal_states": ["Cancelled"], "failure_domains": [],                 "cancellation_actors": ["Operator"], "uncertainty_type": None},
    "Incomplete Info":               {"terminal_states": ["Pass","Fail","ExpectedFail","Cancelled"], "failure_domains": [], "cancellation_actors": [], "uncertainty_type": "Aleatoric"},
}


def _verdict(symptom, source, node, line, rule, outcome, detail):
    compound = _COMPOUND_FIELDS.get(outcome, {"terminal_states": ["Fail"], "failure_domains": ["User","InfraHW","InfraSW"], "cancellation_actors": [], "uncertainty_type": "Epistemic"})
    return {
        "verdict": {
            "root_symptom": {
                "symptom": symptom,
                "source": {
                    "observation_id": f"obs-{uuid.uuid4().hex[:8]}",
                    "source": source,
                    "pos": 42,
                    "line": line,
                    "rule": rule,
                    "time": "2026-05-22T18:30:00+00:00",
                },
            },
            "attribution_outcome": outcome,
            "detail": detail,
            **compound,
        },
        "symptoms": {
            symptom: [
                {
                    "type": "NodeSource",
                    "node": node,
                    "source": {
                        "observation_id": f"obs-{uuid.uuid4().hex[:8]}",
                        "source": source,
                        "pos": 42,
                        "line": line,
                        "rule": rule,
                        "time": "2026-05-22T18:30:00+00:00",
                    },
                }
            ]
        },
        "observation_context": [],
    }


SCENARIOS = [
    {
        "job_id": "1001",
        "scheduler": "slurm",
        "attempt_id": "1",
        "nodes": ["example-slurm-gpu-007"],
        "user": "training-team",
        "framework": "PyTorch",
        "outcome": "Fail InfraHW",
        "terminal_state": "Fail",
        "sub_domain": "InfraHW",
        "responsible_nodes": ["example-slurm-gpu-007"],
        "verdict": _verdict(
            "GpuFell",
            "SYSLOG",
            "example-slurm-gpu-007",
            "NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.",
            "gpu_fell_off_bus",
            "Fail InfraHW",
            "GPU fell off the bus on example-slurm-gpu-007 mid-run (XID 79).",
        ),
    },
    {
        "job_id": "1002",
        "scheduler": "slurm",
        "attempt_id": "1",
        "nodes": ["example-slurm-gpu-012", "example-slurm-gpu-013"],
        "user": "training-team",
        "framework": "PyTorch",
        "outcome": "Fail InfraHW",
        "terminal_state": "Fail",
        "sub_domain": "InfraHW",
        "responsible_nodes": ["example-slurm-gpu-012"],
        "verdict": _verdict(
            "EccDoubleBit",
            "SYSLOG",
            "example-slurm-gpu-012",
            "NVRM: Xid (PCI:0000:81:00): 48, GPU 5 GPU has fallen off the bus due to: Double-bit ECC error.",
            "ecc_double_bit",
            "Fail InfraHW",
            "Uncontained double-bit ECC error on example-slurm-gpu-012 (XID 48).",
        ),
    },
    {
        "job_id": "1003",
        "scheduler": "slurm",
        "attempt_id": "1",
        "nodes": ["example-slurm-gpu-022"],
        "user": "data-team",
        "framework": "JAX",
        "outcome": "Fail InfraSW",
        "terminal_state": "Fail",
        "sub_domain": "InfraSW",
        "responsible_nodes": [],
        "verdict": _verdict(
            "DriverHang",
            "APP_LOG",
            "example-slurm-gpu-022",
            "CUDA error: an illegal memory access was encountered, NVRM watchdog timeout fired.",
            "cuda_illegal_access",
            "Fail InfraSW",
            "CUDA driver hung; pre-flight health check passed. Software-class.",
        ),
    },
    {
        "job_id": "1004",
        "scheduler": "slurm",
        "attempt_id": "1",
        "nodes": ["example-slurm-gpu-031"],
        "user": "inference-team",
        "framework": "Triton",
        "outcome": "Fail User",
        "terminal_state": "Fail",
        "sub_domain": "User",
        "responsible_nodes": [],
        "verdict": _verdict(
            "OOM",
            "APP_LOG",
            "example-slurm-gpu-031",
            "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.94 GiB.",
            "torch_oom",
            "Fail User",
            "OOM at allocation. User config exceeds available memory.",
        ),
    },
    {
        "job_id": "1005",
        "scheduler": "slurm",
        "attempt_id": "1",
        "nodes": ["example-slurm-gpu-044", "example-slurm-gpu-045"],
        "user": "training-team",
        "framework": "PyTorch",
        "outcome": "Pass",
        "terminal_state": "Pass",
        "sub_domain": None,
        "responsible_nodes": [],
        "verdict": _verdict(
            "Clean",
            "SCHEDULER",
            "example-slurm-gpu-044",
            "Job completed with exit code 0.",
            "clean_exit",
            "Pass",
            "Clean completion, no failure signals.",
        ),
    },
    {
        "job_id": "1006",
        "scheduler": "slurm",
        "attempt_id": "1",
        "nodes": ["example-slurm-gpu-051"],
        "user": "research-team",
        "framework": "PyTorch",
        "outcome": "Fail MultiCategory",
        "terminal_state": "Fail",
        "sub_domain": "Multi",
        "responsible_nodes": ["example-slurm-gpu-051"],
        "verdict": _verdict(
            "MixedSignals",
            "SYSLOG",
            "example-slurm-gpu-051",
            "Multiple concurrent symptoms: NVRM Xid 31 (illegal address) and OOM observed within 30s window.",
            "multi_category",
            "Fail MultiCategory",
            "Two distinct symptom categories within one job attempt.",
        ),
    },
    # XID 145 + NCCL 12 cascade — the worked example in the README.
    # Multi-symptom: NVLink RLW on the root node, NCCL error 12 on two peers,
    # scheduler cancel. catalog_id wires the tenant_sre + cloud_operator
    # persona blocks from failures.yml.
    {
        "job_id": "1007",
        "scheduler": "slurm",
        "attempt_id": "1",
        "nodes": ["example-slurm-gpu-018", "example-slurm-gpu-019", "example-slurm-gpu-020"],
        "user": "training-team",
        "framework": "PyTorch",
        "outcome": "Fail InfraHW",
        "terminal_state": "Fail",
        "sub_domain": "InfraHW",
        "responsible_nodes": ["example-slurm-gpu-018"],
        "verdict": {
            "verdict": {
                "root_symptom": {
                    "symptom": "NvlinkRlw",
                    "catalog_id": "xid-145",
                    "source": {
                        "observation_id": "obs-nvl145-001",
                        "source": "SYSLOG",
                        "pos": 117,
                        "line": "NVRM: Xid (PCI:0000:81:00): 145, pid=28311, NVLINK_RLW_ERROR RLW_SRC_TRACK ErrorStatus=0x00000004",
                        "rule": "nvlink_rlw_cascade",
                        "time": "2026-05-22T18:17:42+00:00",
                    },
                },
                "attribution_outcome": "Fail InfraHW",
                "detail": "NVLink RLW cascade on example-slurm-gpu-018: XID 145 NVLINK_RLW_ERROR triggered NCCL collective failure across 3 nodes (XID 145 -> NCCL error 12 -> scheduler FAILED).",
                "terminal_states": ["Fail"],
                "failure_domains": ["InfraHW"],
                "cancellation_actors": [],
                "uncertainty_type": None,
            },
            "symptoms": {
                "NvlinkRlw": [
                    {
                        "type": "NodeSource",
                        "node": "example-slurm-gpu-018",
                        "source": {
                            "observation_id": "obs-nvl145-001",
                            "source": "SYSLOG",
                            "pos": 117,
                            "line": "NVRM: Xid (PCI:0000:81:00): 145, pid=28311, NVLINK_RLW_ERROR RLW_SRC_TRACK ErrorStatus=0x00000004",
                            "rule": "nvlink_rlw_cascade",
                            "time": "2026-05-22T18:17:42+00:00",
                        },
                    },
                ],
                "NcclCollectiveFail": [
                    {
                        "type": "NodeSource",
                        "node": "example-slurm-gpu-019",
                        "source": {
                            "observation_id": "obs-nccl12-001",
                            "source": "APP_LOG",
                            "pos": 2041,
                            "line": "example-slurm-gpu-019:28312:28312 [0] NCCL WARN Net : Connection closed by remote peer example-slurm-gpu-018:28311; error 12: unhandled cuda error, nccl version 2.26.2",
                            "rule": "nccl_error_12",
                            "time": "2026-05-22T18:17:43+00:00",
                        },
                    },
                    {
                        "type": "NodeSource",
                        "node": "example-slurm-gpu-020",
                        "source": {
                            "observation_id": "obs-nccl12-002",
                            "source": "APP_LOG",
                            "pos": 2042,
                            "line": "example-slurm-gpu-020:28313:28313 [0] NCCL WARN Net : Connection closed by remote peer example-slurm-gpu-018:28311; error 12: unhandled cuda error, nccl version 2.26.2",
                            "rule": "nccl_error_12",
                            "time": "2026-05-22T18:17:43+00:00",
                        },
                    },
                ],
                "SchedulerJobFailed": [
                    {
                        "type": "NodeSource",
                        "node": "example-slurm-gpu-018",
                        "source": {
                            "observation_id": "obs-sched-001",
                            "source": "SCHEDULER",
                            "pos": 1,
                            "line": "slurmstepd: error: *** STEP 1007.0 ON example-slurm-gpu-018 CANCELLED DUE TO SIGNAL ***",
                            "rule": "slurm_step_cancelled",
                            "time": "2026-05-22T18:17:44+00:00",
                        },
                    },
                ],
            },
            "observation_context": [],
        },
    },
    {
        "job_id": "team-alpha/job-llama-7b-pretrain-xyz77",
        "scheduler": "kueue",
        "attempt_id": "uid-bbbb-1002",
        "nodes": ["example-kueue-gpu-014", "example-kueue-gpu-015", "example-kueue-gpu-016"],
        "user": "bob",
        "framework": "PyTorch",
        "outcome": "Fail InfraHW",
        "terminal_state": "Fail",
        "sub_domain": "InfraHW",
        "responsible_nodes": ["example-kueue-gpu-014"],
        "verdict": _verdict(
            "GpuFell",
            "SYSLOG",
            "example-kueue-gpu-014",
            "NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.",
            "gpu_fell_off_bus",
            "Fail InfraHW",
            "GPU fell off the bus on example-kueue-gpu-014 during distributed pretraining.",
        ),
    },
    {
        "job_id": "team-alpha/job-llama-7b-pretrain-xyz77",
        "scheduler": "kueue",
        "attempt_id": "uid-cccc-1003",
        "nodes": ["example-kueue-gpu-022", "example-kueue-gpu-023", "example-kueue-gpu-024"],
        "user": "bob",
        "framework": "PyTorch",
        "outcome": "Fail InfraSW",
        "terminal_state": "Fail",
        "sub_domain": "InfraSW",
        "responsible_nodes": [],
        "verdict": _verdict(
            "NcclTimeout",
            "APP_LOG",
            "example-kueue-gpu-022",
            "NCCL WARN [Net] Connection closed by remote peer; aborting collective allreduce.",
            "nccl_collective_timeout",
            "Fail InfraSW",
            "NCCL collective timeout after Kueue requeued the workload onto different nodes.",
        ),
    },
    {
        "job_id": "team-beta/job-rag-eval-suite-2026-05",
        "scheduler": "kueue",
        "attempt_id": "uid-dddd-2001",
        "nodes": ["example-kueue-gpu-031"],
        "user": "alice",
        "framework": "JAX",
        "outcome": "Fail User",
        "terminal_state": "Fail",
        "sub_domain": "User",
        "responsible_nodes": [],
        "verdict": _verdict(
            "OOM",
            "APP_LOG",
            "example-kueue-gpu-031",
            "jax.errors.XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1431655765 bytes.",
            "jax_oom",
            "Fail User",
            "JAX OOM at batch size 256. Reduce batch or use gradient checkpointing.",
        ),
    },
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="demo.db")
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        out.unlink()
    con = sqlite3.connect(out)
    cur = con.cursor()
    for stmt in SCHEMA:
        cur.execute(stmt)
    now = datetime.now(timezone.utc)
    cur.execute(
        "INSERT INTO schema_version(version, description, applied_at) VALUES (?, ?, ?)",
        ("1", "demo seed v1", now.isoformat()),
    )
    for i, s in enumerate(SCENARIOS):
        ctx_id = f"ctx-{uuid.uuid4().hex[:12]}"
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        start = now - timedelta(hours=2 + i * 3)
        end = start + timedelta(hours=1, minutes=20 + i * 5)
        cur.execute(
            "INSERT INTO execution_context(id, job_id, attempt_id, scheduler_type, start_time, end_time, nodes_json, status, workload_framework, user_name, raw_scheduler_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                ctx_id,
                s["job_id"],
                s["attempt_id"],
                s["scheduler"],
                start.isoformat(),
                end.isoformat(),
                json.dumps(s["nodes"]),
                "FAILED" if s["outcome"] != "Pass" else "COMPLETED",
                s["framework"],
                s["user"],
                json.dumps({"jobid": s["job_id"], "scheduler": s["scheduler"]}),
                now.isoformat(),
            ),
        )
        cur.execute(
            "INSERT INTO attribution_run(id, execution_context_id, schema_version, pipeline_version, service_api_version, ruleset_version, started_at, completed_at, outcome, terminal_state, sub_domain, raw_result_json, responsible_nodes_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                ctx_id,
                "1",
                "0.1.0",
                "v2",
                "external-2026.05.20",
                end.isoformat(),
                (end + timedelta(seconds=8)).isoformat(),
                s["outcome"],
                s["terminal_state"],
                s["sub_domain"],
                json.dumps(s["verdict"]),
                json.dumps(s["responsible_nodes"]),
            ),
        )
    con.commit()
    con.close()
    print(f"wrote {len(SCENARIOS)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
