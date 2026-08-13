# Kueue Sample Output

This document shows representative Attribution Blueprint results for a healthy Kueue workload and an unhealthy Kueue execution context with XID-149.4.
The identifiers and node names use placeholders, and the timestamps are illustrative.
The field names and taxonomy values match the CLI, enriched result, and SQLite output contracts.

Use these examples with the [Kueue guide](./kueue.md) and its
[commented configuration](../../examples/kueue/README.md).

## Important CLI Distinction

The CLI reports whether execution contexts completed the attribution pipeline:

```text
Attribution complete: 1/1 execution contexts succeeded.
```

This line does not mean the workload was healthy. It means the blueprint
obtained a verdict and at least one output connector wrote it. Read `outcome`,
`root_symptom`, `terminal_state`, and `sub_domain` to determine workload health.

## Healthy Workload

This example assumes the pod log contains a recognized completion marker such
as `[rank 0] Main process ended cleanly`. `FINISHED_SUCCEEDED` is preserved as
the Kueue scheduler state, but the current service rules do not translate that
state to the `COMPLETED` success fallback.

### CLI summary

```text
Attribution complete: 1/1 execution contexts succeeded.
- team-a/job-training-healthy-a1b2c attempt=64b4b82e-7f5d-4fd0-8ed4-1a2b3c4d5e6f terminal_state=pass sub_domain=none root_symptom=the process completes symptoms=the process completes
```

### Persisted summary

| Field | Representative value | Meaning |
|---|---|---|
| `execution_context.scheduler_type` | `kueue` | The context came from the Kueue adapter. |
| `execution_context.status` | `FINISHED_SUCCEEDED` | State derived from the successful Kueue terminal condition. |
| `execution_context.raw_scheduler_json.state` | `FINISHED_SUCCEEDED` | Kueue-derived state preserved in the raw context. |
| `attribution_run.outcome` | `Pass` | The Attribution Service classified the workload as healthy. |
| `attribution_run.terminal_state` | `pass` | Blueprint-normalized terminal taxonomy. |
| `attribution_run.sub_domain` | `none` | No failure domain applies. |
| `attribution_run.responsible_nodes_json` | `[]` | No node is identified as responsible for a failure. |

### Enriched result excerpt

```json
{
  "verdict": {
    "root_symptom": {
      "symptom": "the process completes",
      "source": {
        "observation_id": 2681252656797335323,
        "source": "applog",
        "pos": 1,
        "line": "[rank 0] Main process ended cleanly",
        "rule": "the process completes",
        "time": null
      },
      "source_channel": "applog"
    },
    "attribution_outcome": "Pass",
    "detail": "Not Applicable"
  },
  "symptoms": {
    "the process completes": [
      {
        "type": "ResourcelessSource",
        "source": {
          "observation_id": 2681252656797335323,
          "source": "applog",
          "pos": 1,
          "line": "[rank 0] Main process ended cleanly",
          "rule": "the process completes",
          "time": null
        },
        "source_channel": "applog"
      }
    ]
  },
  "model_type": "AttributionResponseV2",
  "terminal_state": "pass",
  "sub_domain": "none",
  "connector_reports": [
    {
      "connector": "KubectlLogsConnector",
      "source": "applog",
      "status": "ok",
      "reason": "",
      "observation_count": 1
    },
    {
      "connector": "LokiConnector",
      "source": "syslog",
      "status": "ok",
      "reason": "",
      "observation_count": 1
    }
  ],
  "tenant_sre_action": "",
  "cloud_operator_sre_action": ""
}
```

The `applog` source shows that the completion verdict came from collected pod
output, not from Kueue's `FINISHED_SUCCEEDED` state. The persisted
`scheduler_type` remains `kueue`.

Observation counts vary with pod count and Loki stream coverage. For example,
a four-pod JobSet normally produces four APPLOG observations.

## Unhealthy Execution Context: XID-149.4

XID-149.4 is the registered symptom
`XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0`. It represents an NVLink NETIR
port-down event whose reason is alignment loss. The raw kernel line may contain
`Xid ... 149` plus encoded fields rather than the literal text `149.4`; the
attribution rules decode the subcode.

### Representative evidence

```text
2026-06-11T02:14:08Z example-node-07 kernel: NVRM: Xid (PCI:0000:82:00): 149, (0x00400011 0x00000000 0x00000000 0x00000000)
```

### Normalized summary

```text
Attribution complete: 1/1 execution contexts succeeded.
- team-a/job-training-xid149-d4e5f attempt=91a8fb19-05c9-46a5-8591-6f7e8d9c0b1a terminal_state=fail sub_domain=hardware root_symptom=XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0 symptoms=XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0
```

### Persisted summary

| Field | Representative value | Meaning |
|---|---|---|
| `execution_context.scheduler_type` | `kueue` | The context came from the Kueue adapter. |
| `execution_context.status` | `FINISHED_SUCCEEDED` | The example workload exited successfully from Kubernetes' perspective. |
| `execution_context.raw_scheduler_json.state` | `FINISHED_SUCCEEDED` | Kueue-derived state preserved in the raw context. |
| `attribution_run.terminal_state` | `fail` | Blueprint-normalized terminal taxonomy. |
| `attribution_run.sub_domain` | `hardware` | Hardware is the attributed failure domain. |
| `attribution_run.responsible_nodes_json` | `["example-node-07"]` | Syslog evidence identifies the affected node. |

### Enriched result excerpt

```json
{
  "verdict": {
    "root_symptom": {
      "symptom": "XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0",
      "source": {
        "observation_id": -3413379767676014989,
        "source": "syslog",
        "pos": 1,
        "line": "2026-06-11T02:14:08Z example-node-07 kernel: NVRM: Xid (PCI:0000:82:00): 149, (0x00400011 0x00000000 0x00000000 0x00000000)",
        "rule": "xid-nvlink-event",
        "time": "2026-06-11T02:14:08Z"
      },
      "source_channel": "syslog"
    },
    "detail": "Known HW Cause - Node"
  },
  "symptoms": {
    "XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0": [
      {
        "type": "NodeSource",
        "node": "example-node-07",
        "source": {
          "observation_id": -3413379767676014989,
          "source": "syslog",
          "pos": 1,
          "line": "2026-06-11T02:14:08Z example-node-07 kernel: NVRM: Xid (PCI:0000:82:00): 149, (0x00400011 0x00000000 0x00000000 0x00000000)",
          "rule": "xid-nvlink-event",
          "time": "2026-06-11T02:14:08Z"
        },
        "source_channel": "syslog"
      }
    ]
  },
  "model_type": "AttributionResponseV2",
  "terminal_state": "fail",
  "sub_domain": "hardware",
  "connector_reports": [
    {
      "connector": "KubectlLogsConnector",
      "source": "applog",
      "status": "ok",
      "reason": "",
      "observation_count": 4
    },
    {
      "connector": "LokiConnector",
      "source": "syslog",
      "status": "ok",
      "reason": "",
      "observation_count": 1
    }
  ]
}
```

`responsible_nodes_json` is a dedicated SQLite column derived from node-scoped
symptom evidence. It is not a required top-level key in `raw_result_json`.
This example also shows why scheduler success and attribution health must be
read separately. A diagnostic workload can exit with code zero after provoking
an error, while the collected syslog still proves that the execution context
experienced a hardware fault.

## Inspect the Stored Result

Run this command from the bundle root after using the Kueue configuration:

```bash
uv run --project blueprint --frozen python - <<'PY'
import json
import sqlite3

database = ".attribution-kueue/attributions.db"
connection = sqlite3.connect(database)
row = connection.execute(
    """
    SELECT
        execution_context.job_id,
        execution_context.scheduler_type,
        execution_context.status,
        attribution_run.outcome,
        attribution_run.terminal_state,
        attribution_run.sub_domain,
        attribution_run.responsible_nodes_json,
        attribution_run.raw_result_json
    FROM attribution_run
    JOIN execution_context
      ON execution_context.id = attribution_run.execution_context_id
    ORDER BY attribution_run.started_at DESC
    LIMIT 1
    """
).fetchone()
connection.close()

if row is None:
    raise SystemExit("No attribution result was stored")

raw_result = json.loads(row[7])
verdict = raw_result.get("verdict")
if isinstance(verdict, dict):
    root = verdict.get("root_symptom")
else:
    root = None
if isinstance(root, dict):
    root_symptom = root.get("symptom", "none")
else:
    root_symptom = "none"

print(f"job_id={row[0]}")
print(f"scheduler_type={row[1]}")
print(f"scheduler_status={row[2]}")
print(f"outcome={row[3]}")
print(f"terminal_state={row[4]}")
print(f"sub_domain={row[5]}")
print(f"responsible_nodes={row[6]}")
print(f"root_symptom={root_symptom}")
print(f"connector_reports={raw_result.get('connector_reports', [])}")
PY
```

For the complete database contract, see
[SQLite Output Schema](../output_schema.md).
