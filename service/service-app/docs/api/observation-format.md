# Observation entry format

This page is the reference for the entries a producer sends to the Attribution
Service on `POST /v2/attributor/{id}/observation`. It covers the entry fields,
how `ts` and `resources` are read, and what each source does when a field is
missing. The HTTP calls around it are in the
[REST walkthrough](../../../../blueprint/docs/rest-api.md).

An observation is a `context` plus a `body`. The `body` is a string holding a JSON array
of entries.

## Entry fields

### `body` | required string

* One log line as plain text, per entry
* Rules match the string verbatim, so if the source emits structured records,
  serialize each one as a string; that string is what the rules and the
  per-source body parsing logic see

**Examples:**

```json
[
  {"body": "Jun 26 23:30:00 node-a kernel: NVRM: Xid (PCI:0000:0f:00): 79, pid=1234, GPU has fallen off the bus."},
  {"body": "[rank3] RuntimeError: NCCL communicator was aborted"},
  {"body": "{\"level\": \"ERROR\", \"msg\": \"CUDA error: an illegal memory access was encountered\"}"}
]
```

The third is a structured record serialized as a string. The rules see the
braces and quotes as part of the line.

### `ts` | optional integer

* Unix time in nanoseconds, as an integer. Always UTC; the service converts to
  the workload timezone itself
* It means "this entry occurred no later than this time"
* Use the time the line was written or scraped, not the start of a collection batch
* Optional in the schema only: `syslog`, `health_check` and `csp_event` drop an entry
  without it, the other sources keep it (see the Timestamp table)

**Examples:**

```json
{"ts": 1792514715000000000}
{"ts": null}
```

The first is Tue Oct 20 2026 16:45:15 UTC. The second says the producer has no timestamp
for this entry; what happens next depends on the source.

### `resources` | optional

Omitted or null means no resources are declared. Otherwise either the exact string `"AllJobResources"` or an array of resource objects:

```json
"resources": [{"kind": "node", "hostname": "node-a"}]
"resources": [{"kind": "node", "hostname": "node-a"}, {"kind": "replica", "name": "worker-0"}]
"resources": "AllJobResources"
```

* The only valid kinds are `node` (key `hostname`) and `replica` (key `name`), and no other kinds or keys are supported
* Exactly one `node` per entry is honored; more than one is handled as the Node table says
  * Replicas are recorded alongside it but are not attributed on
  * An array with replicas and no node counts as declaring no node
* The hostname is compared in lowercase against the workload's node list, and a
  fully qualified name matches its short form. **IP addresses are not resolved.**
* `"AllJobResources"` counts the entry against every node of the workload and
  skips node parsing only; a missing `ts` is still read from the body as the
  Timestamp table says. Only `applog` keeps such an entry; every other source
  drops it, because they need exactly one declared node and that node must be
  in the workload.

Declare `resources` whenever the producer knows the node. Body parsing is a fallback with narrow
accepted shapes, and on `applog` an unparseable line is counted against every
node of the workload. The tables below say what each source does when `resources` names no node.

## Per-source handling

`ts` and `resources` are read the same way by every source. What differs is
what happens when they are missing, and whether the body is parsed to fill
them in.

### Timestamp

| Source | `ts` required | `ts` null |
|---|---|---|
| `applog` | no | parsed from body, else kept untimed |
| `syslog` | yes | dropped |
| `health_check` | yes | dropped |
| `ufmlog` | no | kept untimed |
| `node_metrics` | no | kept untimed |
| `nvlink_metrics` | no | kept untimed |
| `csp_event` | yes | dropped |

Only `applog` attempts to parse a timestamp from the body if one is not given, recognizing
`[EMMDD HH:MM:SS` (year taken from the workload end time) or `[YYYY-MM-DD HH:MM:SS]`.

### Node

| Source | No node declared | Hostname not in workload, or more than one node | Skipped when workload has no nodes |
|---|---|---|---|
| `applog` | parsed from body, else all job nodes | all job nodes | no |
| `syslog` | parsed from body, else dropped | dropped | yes |
| `health_check` | parsed from body, else dropped | dropped | no |
| `ufmlog` | parsed from body, else dropped | dropped | yes |
| `node_metrics` | dropped | dropped | yes |
| `nvlink_metrics` | dropped | dropped | yes |
| `csp_event` | dropped | dropped | yes |

Sources that parse a node from the body shape differ in what they recognize as well:

* `applog`: 
  * a hostname prefix like `node-a: ...` or `node-a:4121:4199 [1] ...`
  * rank prefix `[rank3] ...` or `3: ...` (a rank maps to the node that hosts it)
* `syslog` & `health_check`:
  * only RFC 3164 `Jun 26 23:30:00 node-a kernel: ...`
* `ufmlog`:
  * a full link record in either direction, like `(Computer:node-a (a1b2)) a1b2:1 - (Switch:sw-1:1/1) c3d4:1/1`
  * `(node-a) node is Up` / `node is Down`

"Skipped when workload has no nodes" means the whole observation is ignored
when the attributor's workload lists no nodes, because node checks would pass
anything. "All job nodes" means the entry is counted against every node of the
workload, the same as `"AllJobResources"`.

## Examples

Both examples are shown decoded. On the wire, the outer `body` is the entry
array serialized to a JSON string.

`applog`, every entry stamped by the producer:

```json
{
  "context": {
    "time_interval": {"start": "2026-06-10T22:00:00Z", "end": "2026-06-11T00:00:00Z"},
    "source": "applog"
  },
  "body": [
    {
      "ts": 1781135403000000000,
      "resources": [{"kind": "node", "hostname": "node-a"}],
      "body": "[rank0]: RuntimeError: NCCL communicator was aborted"
    },
    {
      "ts": 1781135404000000000,
      "resources": [{"kind": "node", "hostname": "node-b"}],
      "body": "node-b:4121:4199 [1] NCCL INFO [Service thread] Connection closed by localRank 1"
    }
  ]
}
```

`syslog`, one entry per line, node taken from where the file was collected:

```json
{
  "context": {
    "time_interval": {"start": "2026-06-10T22:00:00Z", "end": "2026-06-11T00:00:00Z"},
    "source": "syslog"
  },
  "body": [
    {
      "ts": 1781135390000000000,
      "resources": [{"kind": "node", "hostname": "node-a"}],
      "body": "Jun 10 23:49:50 node-a kernel: NVRM: Xid (PCI:0000:0f:00): 79, pid=1234, GPU has fallen off the bus."
    }
  ]
}
```

A `syslog` entry without `ts` is dropped, and one without `resources` is kept
only if the line is in RFC 3164 form. Set both.
