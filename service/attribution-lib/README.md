> **Disclaimer** FACT (Failure Attribution and Characterization) is
> provided for informational purposes only. Its outputs are advisory
> and do not constitute a determination of hardware fault, an
> authorization or justification for hardware return, or a substitute for
> vendor-approved field diagnostics.

# FACT Attribution Library

This is the core attribution functionality for FACT. It is intended to be used as a library to power other services and utilities.

For the most part, this package will be consumed by other tools and not be directly run by end-users.

## Rule Creatiion

Rules live in the top-level `rules/` directory and are loaded by the rules engine. There are four kinds of rules, each running at a different stage of the attribution pipeline

### Module-level variables

Rule files support two optional module-level variables:

- **`rule_author`** — default author for all rules in the file when `author` is not specified per-rule.
- **`rule_sources`** — set of `ObservationSource` values restricting which observation types the extractors in this file apply to (e.g. `{ObsSource.SYSLOG}`, `{ObsSource.APPLOG}`). Only relevant for extractor rules.

### Adding a new extractor rule

Extractors match on observation content (log lines, metrics, etc.) and emit events into the event stream. Add your rule in the appropriate file under `rules/` based on the observation source:

| Observation source | Rule file |
|--------------------|-----------|
| Application logs | `applog_rules.py` |
| System logs | `syslog_rules.py` |
| Health check / epilog | `epilog_rules.py` |
| UFM logs | `ufmlog_rules.py` |
| Prometheus node metrics | `node_metrics_rules.py` |
| NVLink metrics | `nvlink_metrics_rules.py` |
| Straggler detection | `straggler_rules.py` |

- Decorate a processor with `@extractor_def(name="YourRuleName", match_re=r"your regex")`.
- The processor receives `(groups, node, source, ctx)` and returns a list of `EventTriple` (event type, payload, source).
- Use `ctx.job_info` for job metadata; do not perform I/O.
- Use `_bad(node, symptom, source)` and `_interesting(node, symptom, source)` helpers from `rule_helpers` for common event patterns.

**Examples:** `rules/syslog_rules.py` (`SyslogXid`, `SyslogLustreError`), `rules/applog_rules.py` (`MFU`, `NCCLWarnCudaFailure`).

### Adding a new filter rule

Filters run over the event stream before events are materialized into symptoms. They query, demote, deactivate, or aggregate events. Add filter rules in `rules/filter_rules.py`.

- Decorate with `@filter_def(name="YourFilterName", author="...", priority=FilterPriority.STANDARD)`.
- The processor receives a `FilterContext` with `event_stream` and `job_info`.
- **Priority** (lower runs first): `EARLY` (100), `MID` (500), `STANDARD` (1000), `LATE` (1500).
- Use `event_stream` to query and modify events (deactivate with `status=0`, create new events with `event_adder()`).
- Choose filter over refiner when your rule needs event-level granularity (e.g. per-GPU payload fields, line numbers, event deactivation) that is lost after materialization.

**Examples:** `rules/filter_rules.py` — `DemoteTflopsSymptoms` (STANDARD), `HandleXidBurstEvents` (EARLY), `CleanInformationalXids` (LATE), `FmNodeFailure` (STANDARD).

### Adding a new refiner rule

Refiners run on materialized `symptoms` (SymptomSet) and `my_bad_nodes` (JobNodeAttributions) after events have been consumed from the event stream. They cleanup cascaded symptoms, resolve conflicts, and apply final symptom transformations. Add refiner rules in `rules/refiner_rules.py`.

- Decorate with `@refiner_def(name="YourRuleName", author="...", priority=_priority("YourRuleName"))`.
- The processor receives a `RefinerContext` with `symptoms`, `my_bad_nodes`, `event_stream`, `job_info`, and `raw_keywords`.
- **Execution order** is defined by `_REFINER_ORDER` list in `refiner_rules.py`. Insert your rule name at the desired position in the list; priorities are computed automatically.
- Use `symptoms.discard()` to remove symptoms, `symptoms.add()` to add new ones.
- `event_stream` is available for non-attribution queries (e.g. pytorch timeout, bad pairs) but attribution events are already consumed.
- Choose refiner over filter when your rule operates on the final symptom/bad-node level (e.g. discarding cascaded symptoms, resolving conflicting outcomes).

**Examples:** `rules/refiner_rules.py` — `SlurmStateSymptoms`, `IBSymptomCleanup`, `CudaXidCascades`, `FabricManagerCascade`.

### Adding a new collector rule

Collectors run post-attribution to query the event stream and produce digest key-value pairs for the report. They do not affect attribution results. Add collector rules alongside extractors in their source file (e.g. `applog_rules.py`, `syslog_rules.py`).

- Decorate with `@collector_def(name="Collect.YourName", formatter=your_formatter)`.
- The processor receives `(event_stream, job_info)` and returns raw data.
- The `formatter` function converts the raw data to a `dict[str, Any]` merged into the report digest.


**Examples:** `rules/applog_rules.py` (`Collect.NCCLCudaVersion`, `Collect.BadPairs`, `Collect.RuntimeArgs`), `rules/syslog_rules.py` (`Collect.XidPciMapping`).

### Choosing between filter and refiner

| | Filter | Refiner |
|---|--------|---------|
| **Runs during** | `from_event_stream()` (before materialization) | `calculate_symptoms()` (after materialization) |
| **Operates on** | Event stream (SQL queries on events table) | Materialized `symptoms` and `my_bad_nodes` |
| **Can access** | Individual event payloads, line numbers, per-GPU fields | Job-level symptom strings, per-node attributions |
| **Can create events** | Yes (`event_adder()`) | No |
| **Can deactivate events** | Yes (`UPDATE SET status=0`) | No (use `symptoms.discard()`) |
| **Use when** | Need event-level granularity or need to create new events | Need to resolve symptom conflicts, cascades, or apply final cleanup |

## Targets

The FACT Attribution library has been largely developed to support recent NVIDIA datacenter-class GPUs and will work best in those environments.

## Dependencies

Most of the dependencies for this library can be readily found in public package indexes. The exception is the NVIDIA XID Analyzer, which is available through internal NVIDIA access or may be vendored with this library in the `extern/` directory.

## Setup

We use [`uv`](https://docs.astral.sh/uv/) for project and dependency management. If you do not have `uv` installed in your environment, run the following to install the latest:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

You will likely need to restart your shell or source an rcfile to enable functionality.

## Versioning

Package versions come from git tags in source checkouts and installed package
metadata in wheels. Untagged source builds fail intentionally rather than
falling back to `0.0.1`. See `docs/versioning.md` for the exact version format,
release build expectations, and `get_version()` / `get_commit_sha()` behavior.
