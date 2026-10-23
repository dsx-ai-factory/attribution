# Adding a Log Connector

A log connector answers one question:

*Given an [execution context](#what-collect-receives) describing a workload run, which
logs should be fetched, and in what shape?*

It's only responsible for fetching data; parsing and attribution
happen downstream. Write one when the target logs live somewhere that
[existing connectors](./CONFIGURATION.md#log-collection) (e.g. `local_file`, `loki`,
`kubectl_logs`) cannot reach.
Connectors are located under [`src/attribution_blueprint/connectors/logs/`](../src/attribution_blueprint/connectors/logs/).

## The Contract

A connector is a class satisfying the
[`LogConnector`](../src/attribution_blueprint/connectors/logs/__init__.py) protocol. Conformance is structural: a class whose members match the protocol satisfies it.

Additionally, new log connectors need a module-level `_build(...)` that turns a resolved config into an
instance, and one `register_component(...)` call at module scope that hands that
factory over. That call runs when the module is imported, and the
[MCA registry](../../docs/ARCHITECTURE.md#connector-frameworks) imports every module under
`connectors/` at startup. Keep heavy third-party dependencies from always importing — see [Third-Party Libraries](#third-party-libraries).

## Write It

The following is an example skeleton that would live in `connectors/logs/example/__init__.py`:

```python
from attribution_blueprint import mca_registry as _registry
from attribution_blueprint.connectors.logs import entries_from_text, log_observation
from attribution_blueprint.exceptions import MCAError
from attribution_blueprint.mca_registry import _PARAM
from attribution_blueprint.types import ExecutionContext, Observation, Source


class ExampleLogConnector:
    def __init__(self, *, endpoint: str, source: Source):
        self._endpoint = endpoint
        self._source = source

    @property
    def source(self) -> Source:
        return self._source

    def collect(self, context: ExecutionContext) -> list[Observation]:
        body = self._fetch(context)  # the backend query
        return [log_observation(source=self._source, entries=entries_from_text(body), context=context)]

    def _fetch(self, context: ExecutionContext) -> str:
        raise NotImplementedError()


def _build(spec: dict, ctx: dict) -> ExampleLogConnector:
    endpoint = spec.get("endpoint", "")
    if endpoint == "":
        raise MCAError("log.example: 'endpoint' is required")
    return ExampleLogConnector(endpoint=endpoint, source=ctx["source"])


_registry.register_component(framework="log", name="example", schema={"endpoint": _PARAM}, build=_build)
```

The [local file connector](../src/attribution_blueprint/connectors/logs/local_file/__init__.py)
is a complete working copy in a single file.

### What `collect` Receives

`ExecutionContext` contains the following guaranteed fields that can be used by the query:

| Field | Type | |
|---|---|---|
| `id` | `str` | the workload run |
| `nodes` | `list[str]` | node names |
| `start_time`, `end_time` | `datetime` | must be timezone-aware |

Other optional fields may be included  - see the declared fields in [`types.py`](../src/attribution_blueprint/types.py).

### What `_build` Receives

This is where config validation belongs before returning the instance. Any failed validation should raise a `MCAError`. The function takes:

* `spec`: the values for the keys declared in `schema`, e.g. `{"endpoint": "..."}`
* `ctx`: passed by framework depending how the component was configured, containing the framework (in this case *log*), the source, and source name

[Configure It](#configure-it) covers where these values get passed down from.

Every key read from `spec` must also appear in `schema=`, which accepts sub-tables.
Leaves map to `_PARAM`, or to `{"_open": True}` for a sub-table whose keys are
user-defined pass-through. See the
[Loki connector](../src/attribution_blueprint/connectors/logs/loki/__init__.py)
schema for both, and for using
[`connectors/auth`](../src/attribution_blueprint/connectors/auth/) if needed by
the new connector.

### Building The Observation

`collect` must emit a list of observations. The `log_observation` helper emits an entry timeseries: one entry per log line. Pass
`entries=` built by `entries_from_text`, or by hand when the backend already gives
structured entries with their own `ts` and `resources`. Whether an entry may omit
those is a per-source contract of the service: for `applog` it falls back to
parsing timestamp and node out of the body, so bare `{"body": ...}` entries are
enough. For `syslog` every entry needs a `ts`; build them with
`syslog_entries_from_text`, which dates each line from its own timestamp and
drops lines it cannot date. The entry shape and the full per-source rules are in
[Observation entry format](../../service/service-app/docs/api/observation-format.md).

`time_interval` defaults per source through `log_observation`
(`DEFAULT_TIME_INTERVAL` in
[`connectors/logs`](../src/attribution_blueprint/connectors/logs/__init__.py)).
Pass `time_interval=` to `log_observation` when a connector needs a more specific
 bound.

Node identity belongs on the entry: when a connector knows which host a line came from, it declares `resources: [{"kind": "node", "hostname": ...}]` on that entry, as the syslog branch of [local file](../src/attribution_blueprint/connectors/logs/local_file/__init__.py) does with the directory name. An entry that declares no node is left to the
service, which for `syslog` reads the hostname field of an RFC 3164 line and
drops a line that names none.

## Third-Party Libraries

A connector may depend on a third-party library, but connector-specific dependencies should be kept optional
and the import deferred. Add a line to the existing `[project.optional-dependencies]`
table in [`pyproject.toml`](../pyproject.toml) and rerun `uv lock`:

```toml
thirdpartylib = ["thirdpartylib>=1.0"]
```

Then import it inside `_build` and hand the client to the constructor, so
`collect` never imports the library itself.

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import thirdpartylib


class ExampleLogConnector:
    def __init__(self, *, client: thirdpartylib.Client, source: Source): ...


def _build(spec: dict, ctx: dict) -> ExampleLogConnector:
    try:
        import thirdpartylib
    except ImportError as exc:
        raise MCAError(
            "log.applog.example needs the 'thirdpartylib' extra: "
            "uv sync --extra thirdpartylib"
        ) from exc
    ...
    return ExampleLogConnector(client=thirdpartylib.Client(spec["endpoint"]), source=ctx["source"])
```

> **NOTE:** The library is not imported at the module level, as it would get auto imported by MCA discovery. This way, it does not need to load if the component was never configured. Please note that a connector's tests must pass with the library absent.

## Configure It

Config goes in a TOML file passed with `--config` like
`./attribution-blueprint.toml`. The example log connector would be configured as such:

```toml
[log.applog]
component = "example"

[log.applog.example]
endpoint = "https://logs.example"
```

A `[log.syslog]` block works the same way. Each source takes exactly one
component. The same settings can also come from
environment variables or CLI flags, covered in the
[configuration reference](./CONFIGURATION.md#environment-variables).

A log block on its own will not discover logs; a run also needs an execution-context
block, which [`static-sample.toml`](./sample-data/static/static-sample.toml)
supplies.

Dropping the module under `connectors/logs/` is enough for discovery to pick it
up. Confirm the registry sees it:

```bash
uv run --project blueprint --frozen nvidia-attribution-blueprint --connector log applog help
uv run --project blueprint --frozen nvidia-attribution-blueprint --connector log applog example help
```

## Test It

Write tests in `tests/connectors/logs/<name>/test_connector.py` (include accompanying `__init__.py` file too).

Execute relevant tests

```bash
uv run --extra dev pytest \
    tests/connectors/logs \
    tests/test_mca_registry.py \
    tests/test_mca_discovery.py \
    tests/test_cli.py -q
```

and run formatting

```bash
uv run ruff format --config ruff.toml --check .
```

To see what the connector actually emitted, dump observations to disk. One file
per context and source is written, so only the last observation for each survives.
They are written before anything is submitted, so no service need be reachable
(though without one the run will not provide an attribution):

```bash
uv run --project blueprint --frozen nvidia-attribution-blueprint --config ./attribution-blueprint.toml \
  --dump-observations ./observations
```

For a complete runnable setup, copy
[`static-sample.toml`](./sample-data/static/static-sample.toml), swap its
`[log.applog]` and `[log.applog.local_file]` blocks for the new component, and
run it with the command above. The
[sample corpus](./sample-data/static/README.md) describes what the static
workloads contain.

## Before Opening The PR

- [ ] New first-party files that support comments carry the Apache-2.0 SPDX header, copied from an existing first-party file
- [ ] `schema=` declares every key `_build` reads, and a bad config raises
      `MCAError`
- [ ] Any third-party library is an extra in [`pyproject.toml`](../pyproject.toml) with
      `uv.lock` regenerated, imported inside `_build`, never at module top level
      ([Third-Party Libraries](#third-party-libraries))
- [ ] Tests pass and `ruff format --check` is clean
- [ ] The new connector is documented appropriately in
      [CONFIGURATION.md](./CONFIGURATION.md#log-collection), including a subsection
      for it

## Other Connector Kinds

* Metrics connectors under
[`connectors/metrics/`](../src/attribution_blueprint/connectors/metrics/) use
this contract unchanged with `framework="metrics"`; metrics declares an
identically-shaped protocol,
[`MetricsConnector`](../src/attribution_blueprint/connectors/metrics/__init__.py), in its
own framework package
* Output backends under
[`connectors/outputs/`](../src/attribution_blueprint/connectors/outputs/) use
`framework="output"` but instead implement `.write(...)` and `.close(...)`.
