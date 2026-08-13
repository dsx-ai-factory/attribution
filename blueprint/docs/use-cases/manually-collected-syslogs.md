# Attribute a Workload From Manually Collected Syslogs

Use this workflow when syslogs have already been copied to the host running
Attribution Blueprint and a live scheduler or centralized log service is not
available. The static execution-context adapter supplies the workload metadata,
and the local-file log connector reads the saved syslogs.

This workflow does not query SLURM, Kueue, or the source nodes. It requires an
Attribution Service that the blueprint host can reach.

## Information You Need

Before you start, record:

- the workload start and end times, including a UTC offset
- the exact node names associated with the workload
- the number of GPUs used per node

Use the narrowest accurate start and end times for the failed attempt. The
blueprint assigns this interval to every local log observation.

The node names are especially important. Each name in the static context must
exactly match a directory name under the syslog base directory.

## Required Directory Structure

Create one child directory per workload node. Put the files collected from a
node directly inside that node's directory:

```text
/data/manual-syslog-case/
├── attribution-blueprint.toml
├── syslog/
│   ├── node-a/
│   │   ├── journal.log
│   │   └── kernel.log
│   └── node-b/
│       ├── journal.log
│       └── kernel.log
└── results/
```

In this example, the static context must contain:

```toml
nodes = ["node-a", "node-b"]
```

The local-file SYSLOG connector applies these rules:

- Each configured `paths` entry is a syslog base directory, not an individual
  log file.
- Each workload node must have a child directory named exactly like the node.
- Every regular file directly inside a node directory is read.
- Nested directories are not scanned.
- File names and extensions are not significant.
- Empty files are skipped.
- Files must be readable by the account running Attribution Blueprint.

Do not combine logs from several nodes into one node directory. The directory
name is how the connector assigns each observation to a node.

Create the output directory before running attribution:

```bash
mkdir -p /data/manual-syslog-case/results
```

## Syslog File Format

Save syslogs as plain text, with one original log record per line when the
source provides line-oriented output. JSON wrapping is not required. Preserve
the original timestamp, hostname, facility or process name, and message when
they are available. For example:

```text
2026-06-10T23:50:03Z node-a kernel: NVRM: Xid (PCI:0000:8a:00): 43, pid=1234
2026-06-10T23:50:04Z node-a systemd[1]: training-worker.service: Main process exited
```

The connector reads each file as bytes and decodes it as UTF-8. Invalid UTF-8
bytes are replaced instead of failing the run. It first creates one `raw_log`
observation per non-empty file. Before upload, Attribution Blueprint splits the
body into non-empty lines and converts it to the service-compatible syslog
format. If the first token in a line is an ISO 8601 timestamp, that timestamp is
used. Otherwise, the context `start_time` is used. The node identity comes from
the directory name.

By default, the connector reads at most the trailing 2 GiB of each file. Set
`max_bytes` to a smaller value when manually collected files are large. Crash
signatures near the end of a file are retained when the file is truncated.

## Configure One Static Execution Context

Create `/data/manual-syslog-case/attribution-blueprint.toml`:

```toml
[attribution]
# Base URL of the Attribution Service that analyzes the collected syslogs.
service_url = "http://127.0.0.1:8000"

[eca]
# Select the static adapter because this workflow does not query a scheduler.
component = "static"

[eca.static]
# Exact beginning of the workload attempt. A UTC offset is required.
start_time = "2026-06-10T23:30:00+00:00"
# Exact end of the same workload attempt. A UTC offset is required.
end_time = "2026-06-10T23:55:00+00:00"
# IANA zone used to interpret any log timestamps that omit an offset.
timezone = "UTC"
# Node names must exactly match the child directory names under the syslog path.
nodes = ["node-a", "node-b"]

[workload]
# GPUs, and therefore training ranks, assigned to each node in this workload.
gpus_per_node = 8

[log.syslog]
# Read SYSLOG evidence from files that are already present on this host.
component = "local_file"

[log.syslog.local_file]
# Parent directory of the node-a and node-b directories shown above.
paths = ["/data/manual-syslog-case/syslog"]

[output.db]
# Persist the execution context and attribution result for later inspection.
component = "sqlite"

[output.db.sqlite]
# SQLite file created by the output connector. Its parent must already exist.
path = "/data/manual-syslog-case/results/attributions.db"
```

The example contains only the values needed for this workflow. `component =
"static"` selects the static execution-context adapter. Its required inline
fields are `start_time`, `end_time`, and `nodes`. Both timestamps must include a
timezone, such as `Z` or `+00:00`. When `id` is omitted, the adapter derives a
stable `static-<hash>` ID from those three fields. Set `timezone` to an IANA
name when evidence timestamps omit their offset. It is required when the
context timestamps use a non-hour numeric offset, such as `+05:30`.

`component = "local_file"` selects the host-filesystem log connector for the
`syslog` source. `paths` is required and accepts one base-directory string or a
list of base directories; this example uses a list. `gpus_per_node` must match
the workload allocation because the blueprint uses it to map training ranks to
nodes. `component = "sqlite"` selects persistent output, and `path` is required
for that output component. The CLI summary is still printed to the terminal.

Absolute paths avoid ambiguity. Relative `contexts_path`, log, output, and
observation-dump paths are resolved from the shell's current working directory,
not from the directory containing the TOML file.

## Run Attribution

From the bundle root, run:

```bash
uv sync --project blueprint --frozen
uv run --project blueprint --frozen nvidia-attribution-blueprint \
  --config /data/manual-syslog-case/attribution-blueprint.toml \
  --dump-observations /data/manual-syslog-case/results/observations
```

No scheduler selector or workload ID argument is needed. The static adapter
returns the context defined in the TOML file.

A successful run reports that one execution context succeeded. The SQLite
database contains the context and returned attribution result. The observation
dump provides a debug view of the collected `raw_log` body under a
`body_raw` key. The dump is keyed by context and source, so treat it as a spot
check rather than a copy of every input file.

## Configure Multiple Static Execution Contexts

Use `contexts_path` when you have several workload attempts. The JSON file may
be either a non-empty array of context objects or an object with a non-empty
`contexts` array. Context IDs must be unique.

For multiple workloads, add a workload-ID level to the syslog tree:

```text
/data/manual-syslog-cases/
├── contexts.json
├── attribution-blueprint.toml
├── syslog/
│   ├── training-job-1234/
│   │   ├── node-a/
│   │   │   └── kernel.log
│   │   └── node-b/
│   │       └── kernel.log
│   └── training-job-5678/
│       └── node-c/
│           └── journal.log
└── results/
```

Create `contexts.json`:

```json
{
  "contexts": [
    {
      "id": "training-job-1234",
      "start_time": "2026-06-10T23:30:00+00:00",
      "end_time": "2026-06-10T23:55:00+00:00",
      "timezone": "UTC",
      "nodes": ["node-a", "node-b"]
    },
    {
      "id": "training-job-5678",
      "start_time": "2026-06-11T01:10:00+00:00",
      "end_time": "2026-06-11T01:40:00+00:00",
      "timezone": "UTC",
      "nodes": ["node-c"]
    }
  ]
}
```

Then replace the inline `[eca.static]` section and update the syslog path:

```toml
[eca]
# Select the static execution-context adapter.
component = "static"

[eca.static]
# Load execution contexts from this JSON file instead of defining one inline.
contexts_path = "/data/manual-syslog-cases/contexts.json"

[log.syslog]
# Read saved system logs from the host filesystem.
component = "local_file"

[log.syslog.local_file]
# Replace {job_id} for each context, then find one child directory per node.
paths = ["/data/manual-syslog-cases/syslog/{job_id}"]
```

`{job_id}` is replaced for each context. When `job_id` is omitted from a JSON
context, it defaults to the context `id`. A normal run processes every context
loaded from the JSON file.

## Add Manually Collected Application Logs

Application logs are optional. They do not use the per-node directory contract.
Configure a file path or a directory containing regular files directly within
it:

```toml
[log.applog]
# Read optional application logs from the host filesystem.
component = "local_file"

[log.applog.local_file]
# Read every regular file directly inside this directory.
paths = ["/data/manual-syslog-case/applog"]
```

See [Evidence Coverage](../CONFIGURATION.md#evidence-coverage) for the accuracy
tradeoffs when APPLOG or SYSLOG is unavailable.

For multiple contexts, use a supported placeholder such as `{job_id}` in the
path and keep each workload's application logs in a separate directory.
Supported placeholders are `{std_out}`, `{std_err}`, `{job_id}`, `{name}`,
`{user_name}`, `{start_time_ns}`, and `{end_time_ns}`.

## Behavior and Limitations

- The static adapter trusts the supplied context. It does not query a scheduler
  or discover workloads by time range.
- The local-file connector does not filter lines to the execution-context time
  range. Prepare files that contain evidence relevant to the attempt.
- Missing or unreadable files are skipped. Permission failures are logged.
- Missing node logs reduce evidence coverage but do not automatically stop the
  run when other evidence is available.
- A normal `contexts_path` run processes every context in the JSON file.

## Troubleshooting

### No syslog directories found

Confirm that the configured `paths` entry points to the parent of the node
directories and that every directory name exactly matches an entry in `nodes`.
Also confirm that you did not add an extra directory level.

### Files are ignored

Confirm that files are directly inside the node directory, are non-empty, and
are readable by the blueprint process. The connector does not recurse into
subdirectories.

### Only the end of a file is processed

The file exceeded the connector's default 2 GiB per-file limit. Add the optional
`max_bytes` setting when a different limit is required. Accepted suffixes
include `KB`, `MB`, `GB`, and `TB`, using binary 1024-based multipliers.

### The context fails validation

Confirm that `id`, `start_time`, `end_time`, and `nodes` are present when using
`contexts_path`, and that the JSON file contains no duplicate IDs. Start and end
times must include a timezone, `nodes` must be a non-empty array of strings, and
`gpu_count` must be zero or greater when supplied. If `timezone` is supplied,
it must be a valid IANA timezone name. It is required when the context
timestamps use a non-hour numeric offset, such as `+05:30`.

For the complete connector reference, see
[Configuration](../CONFIGURATION.md#ecastatic). For a committed static-data
example, see [Static Sample Data](../sample-data/static/README.md).

For live Kubernetes batch workloads, see
[Attribute Kueue Workloads](./kueue.md).
