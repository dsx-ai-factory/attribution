# Scheduler Requirements

## SLURM

This document describes what must be available for the blueprint to attribute SLURM jobs.

Important distinction: the attribution service does not query SLURM. The blueprint process queries SLURM with `sacct`, converts the result into workload metadata, collects logs and metrics, then sends that data to the service over HTTP.

Minimum supported version: SLURM 25.11. Older versions are not supported in this build, because `SLUID` and the required `sacct` format fields were not available before 25.11.

The blueprint also depends on the SLURM accounting database still containing the jobs being queried. If `MaxJobAge` (in `slurm.conf`) has rolled the job out of the accounting history, `sacct` returns no rows and the blueprint cannot attribute it. Check `MaxJobAge` first if you see "no rows" for older jobs.

### Required Access

Run the blueprint on a host where all of the following are true:

- SLURM 25.11 or later on both the controller and the `sacct` client
- `sacct` is installed and can talk to the target cluster accounting database
- the user running the blueprint can query the target jobs
- SLURM accounting history still contains the jobs you want to attribute (see `MaxJobAge` above)
- `sacct` can return `SLUID`
- the host can read application log paths returned by `StdOut` and `StdErr` if using local file APPLOG collection
- the host can reach the attribution service URL over HTTP. The blueprint talks to the service over HTTP only; it has no direct dependency on the service's storage layer.
- the host can reach Loki, Prometheus, or Mimir if those collectors are configured

If SLURM is configured with private accounting data, the blueprint can only see jobs that the operating user is allowed to see. Run it as an admin or service account if you need cluster-wide attribution.

### SLURM Version And SLUID

The current SLURM adapter requires SLURM 25.11 or later. This is a hard floor: `SLUID` and the required `sacct` format fields are not available in earlier versions. Rows without `SLUID` are skipped.

Check support with:

```bash
sacct --jobs=123456 --format=JobID,SLUID --parsable2 --noheader
```

A healthy response looks like:

```text
123456|sluid-...
```

If `sacct` reports that `SLUID` is an invalid field, the cluster or client is too old for this adapter. Upgrade SLURM or use a blueprint build that does not require `SLUID`.

### What The Blueprint Runs

For job ID mode, the blueprint runs `sacct` with the selected job IDs:

```bash
sacct --jobs=123456,123457 \
  --format=JobID,JobIDRaw,SLUID,State,Start,End,NodeList,FailedNode,TimelimitRaw,AllocTRES,StdOut,StdErr,User,JobName \
  --parsable2 \
  --noheader \
  --duplicates
```

For time-range mode, it runs:

```bash
sacct --starttime=2026-05-25T00:00:00 \
  --endtime=2026-05-25T06:00:00 \
  --format=JobID,JobIDRaw,SLUID,State,Start,End,NodeList,FailedNode,TimelimitRaw,AllocTRES,StdOut,StdErr,User,JobName \
  --parsable2 \
  --noheader \
  --duplicates
```

`--duplicates` is required so requeued attempts are visible. Without `--duplicates`, `sacct -X` collapses requeues to the latest attempt only, which breaks per-attempt and per-step attribution. Do not add `-X`, because it can collapse requeue history and hide earlier attempts.

### Required sacct Fields

| Field | Needed for |
|---|---|
| `JobID` | User-facing job ID and step suffix detection. |
| `JobIDRaw` | Numeric job ID for resolving SLURM output filename patterns. |
| `SLUID` | Stable attempt identifier, including requeue attempts. |
| `State` | Workload status sent to the attribution service. |
| `Start` | Evidence collection window start. |
| `End` | Evidence collection window end. |
| `NodeList` | Node list for service workload metadata and node-scoped log or metric queries. |
| `FailedNode` | Optional failed node signal. |
| `TimelimitRaw` | Optional timeout signal. |
| `AllocTRES` | Resource details, including GPU allocation when present. |
| `StdOut` | Application log path for local file APPLOG collection. |
| `StdErr` | Application error log path for local file APPLOG collection. |
| `User` | Workload owner metadata. |
| `JobName` | Workload name metadata and output path pattern substitution. |

The adapter filters `.batch` and `.extern` rows and keeps numeric step rows such as `.0` and `.1`. Numeric steps can produce separate attribution contexts because they can have different windows and evidence.

### Timezone Requirement

Many `sacct` deployments return timestamps without timezone offsets. Configure the SLURM controller timezone in `attribution-blueprint.toml`:

```toml
[eca.slurm]
cluster_tz = "America/Los_Angeles"
```

Use the IANA timezone for `slurmctld`, not necessarily the timezone of the
laptop or container running the blueprint. Always set it for a zone that
observes daylight-saving changes or uses a non-hour UTC offset. Without
`cluster_tz`, the host fallback keeps only numeric offsets. A job that spans an
offset change or uses a non-hour offset is rejected instead of being sent with
an incorrect timezone.

When you pass CLI times with no timezone offset, the blueprint interprets them as UTC. Prefer explicit offsets:

```bash
--connector eca slurm start_time "2026-05-25T00:00:00+00:00"
--connector eca slurm end_time "2026-05-25T06:00:00+00:00"
```

### Application Logs

For local file APPLOG collection:

```toml
[log.applog]
component = "local_file"

[log.applog.local_file]
paths = ["{std_out}", "{std_err}"]
```

The blueprint host must be able to read the resolved `StdOut` and `StdErr` paths. This usually means the same shared filesystem is mounted at the same paths visible to SLURM.

Validate with:

```bash
sacct --jobs=123456 --format=JobID,StdOut,StdErr --parsable2 --noheader
ls -l /path/from/StdOut
```

If the path exists but the blueprint user cannot read it, the local file connector logs a permission warning and continues without that application log. The service can still return a verdict from scheduler metadata, syslog, or metrics, but attribution quality may be lower.

### Syslog And Metrics

SLURM itself does not provide kernel logs, fabric logs, or Prometheus metrics.
If you want the service to use those signals, configure collectors:

- Loki for `syslog` or centralized `applog`
- PromQL for `node_metrics`

The blueprint needs endpoint access and credentials. The service still does not access those systems directly.

For Loki, confirm that the stream labels can identify nodes in the same spelling used by SLURM `NodeList`. The default `node_label` is `host_name`.

For PromQL, confirm that host labels can match the SLURM node names. The bundled queries expect `net_host_name` for host matching and `s_cluster` for the cluster label.

### SLURM Epilog Hooks

Some clusters run SLURM epilog hooks that emit additional failure signals after
a job completes. The blueprint's SLURM ECA does not currently collect epilog
output directly.

As a result, attribution produced by the blueprint uses the signals it does
collect, including `sacct` metadata, local APPLOG, Loki, and PromQL. It may not
include signals that only appear in epilog output.

### Validation Checklist

Use this checklist before the first production run:

- `uv run --project blueprint --frozen nvidia-attribution-blueprint --connector eca slurm help` lists `ids`, `start_time`, `end_time`, `nodelist`, `sacct_path`, and `cluster_tz`
- `sacct --jobs=<jobid> --format=JobID,SLUID --parsable2 --noheader` returns a non-empty `SLUID`
- the full `sacct --format=...` command returns `Start`, `End`, `NodeList`, and `State`
- the blueprint user can read at least one application log file if local APPLOG is configured
- `cluster_tz` is set when the blueprint host timezone differs from the SLURM controller timezone, the controller observes DST, or its UTC offset is not a whole hour
- `curl -fsS http://<service-host>:8000/v2/healthz` succeeds from the blueprint host
- Loki and PromQL token environment variables are set if those collectors use `token_env`

### Common Failures

#### No verdicts are written

Run the `sacct` command manually for the same job ID. If it returns no rows, the blueprint has no execution contexts to send to the service.

#### Rows are skipped because `SLUID` is empty

The adapter requires `SLUID`. Confirm that both the SLURM server and the `sacct` client support it.

#### Time-range run finds the wrong jobs

Check `cluster_tz`. `sacct` does not accept timezone offsets in `--starttime` or `--endtime`, so the blueprint converts your requested instant into the configured cluster timezone before calling `sacct`.

#### Local APPLOG is missing

Check `StdOut` and `StdErr` from `sacct`, then verify that the blueprint user can read those exact paths on the blueprint host.

#### Loki or PromQL returns no evidence

Check node label spelling. The service can only correlate evidence to nodes already listed in the workload metadata, and the blueprint lowercases node names before sending the workload to the service.
