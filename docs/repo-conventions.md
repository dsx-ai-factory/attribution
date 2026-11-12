# Repo Conventions

This document covers general conventions on how this repository is shaped and how we work in it.

## Where code goes

### R1. Dependencies point one way

The service imports the library. The blueprint reaches the service only through the generated API client and never imports the library or the service directly. Nothing imports in the other direction. See [docs/ARCHITECTURE.md](ARCHITECTURE.md) for what each component does.

**Why:** With one-way dependencies, each component can be tested and released on its own, and the library stays free of service and client code.

### R2. Shared behavior goes in the library

Behavior that every client needs goes in the library, and the service exposes it. A client converts its own sources into the observation format the library and service accept. It does not re-implement the library's parsing, matching or models.

**Why:** A client that copies the library's logic, such as its own regexes for matching logs, stops getting the library's fixes. Its verdicts then drift from the library's without anyone noticing.

### R3. The library makes no network calls or storage assumptions

The library never opens a network connection or resolves a hostname. Its data comes from the caller, from files the caller names, or from its own bundled data. It requires no storage assumptions: any disk use goes through options the caller controls, and the library still works in memory when those options are off.

**Why:** Attribution then runs the same offline, in tests and behind firewalls, and a slow or missing network can't change a verdict.

### R4. Specific behavior goes in rules, not the library core

Behavior tied to one scheduler, failure or symptom goes in a rule ([R6](#r6-rule-specific-code-and-tests-live-with-the-bundled-rules)), not in the library core. The job model follows the same idea ([R5](#r5-the-job-model-stays-generic)).

**Why:** Clients run on different schedulers and see different failures, and a core built around one case breaks or misleads the rest.

### R5. The job model stays generic

The job model holds facts that any scheduler or framework can provide, such as the job's ID, nodes, time range and exit state. Facts that only one scheduler, framework, tool or data source provides arrive as observations or in their own model.

```python
# Yes
exit_code: int | None
time_limit: int | None

# No
slurm_node_infos: dict[str, dict[str, str]]  # only Slurm fills this
```

**Why:** A field that only one source can fill becomes part of the contract for every client. The model keeps growing, and readers can't tell which fields they can rely on.

### R6. Rule-specific code and tests live with the bundled rules

Rules and rule data go under [`service/attribution-lib/src/attribution_lib/bundled_rules/`](https://github.com/dsx-ai-factory/attribution/tree/main/service/attribution-lib/src/attribution_lib/bundled_rules/). Declare a new symptom in the [symptom registry](https://github.com/dsx-ai-factory/attribution/blob/main/service/attribution-lib/src/attribution_lib/bundled_rules/symptom_defaults.py), not in the legacy enum. Test a rule with its inline cases first, and keep Python rule tests under [`bundled_rules/tests/`](https://github.com/dsx-ai-factory/attribution/tree/main/service/attribution-lib/src/attribution_lib/bundled_rules/tests/). Code and tests outside `bundled_rules/` don't depend on a specific rule. Follow [adding-a-rule.md](https://github.com/dsx-ai-factory/attribution/blob/main/service/attribution-lib/docs/adding-a-rule.md) for the full steps.

**Why:** Keeping rule data in one place keeps core code free of rule details. Tests next to a rule show what the rule is meant to catch.

## Logging and errors

### R7. The log level matches the consequence

Log a failure the code doesn't recover from at error, with its traceback, and only in the handler that stops it (style guide [S17](style-guide.md#s17-log-handled-errors-with-their-traceback)). Elsewhere, let it propagate without logging it, so it is logged once. Log a condition the code recovers from at warning or info, and include its cause.

```python
# Yes: the verdict is still returned, so this is a warning
except DAG_SHAPE_ERRORS as e:
    logger.warning("omitting decision graph from the verdict: {}", e)

# No: wrong level, and the cause is lost
except DAG_SHAPE_ERRORS:
    logger.error("Graph failed")
```

**Why:** Operators alert on error logs. A recovered condition logged as an error trains them to ignore errors, and a message without its cause can't be acted on.

### R8. Each component has its own exception base

Library exceptions derive from `AttributionError`. Service HTTP errors derive from `AttributionErrorModel` and are raised with `.to_http_exception()`, which wraps them in `AttributionHTTPException`. Blueprint exceptions derive from `BlueprintComponentError`.

```python
# Yes
class SnapshotFormatError(AttributionError): ...

# No
class SnapshotFormatError(ValueError): ...
```

**Why:** A caller can handle every error from a component with one `except` clause, without also catching unrelated built-in errors.

## Service API, schema, versioning and generated code

### R9. A released API version never changes incompatibly

Once an API version such as `/v1` or `/v2` is released, change it only in backward-compatible ways. Put incompatible behavior in a new API version; adding one is not a breaking change. Update the [API reference](https://github.com/dsx-ai-factory/attribution/blob/main/service/service-app/docs/api/v2.md) for that version in the same pull request. [SUPPORT.md](https://github.com/dsx-ai-factory/attribution/blob/main/SUPPORT.md#interface-compatibility) and [RELEASE.md](https://github.com/dsx-ai-factory/attribution/blob/main/RELEASE.md) define the compatibility commitments and version bumps.

**Why:** Clients pin an API version and upgrade on their own schedule. A silent change to a released version breaks them without warning.

### R10. The generated API client is regenerated, never hand-edited

The API client and `openapi.json` are generated. Regenerate them with [service/service-app/openapi/generate-client.sh](https://github.com/dsx-ai-factory/attribution/blob/main/service/service-app/openapi/generate-client.sh) and commit the result with the API change that caused it. To change a generated file, change its template or the hand-maintained extras instead, as the [openapi README](https://github.com/dsx-ai-factory/attribution/blob/main/service/service-app/openapi/README.md) describes. [CI fails](https://github.com/dsx-ai-factory/attribution/blob/main/.ci/check-client-drift.sh) if the committed API client differs from a fresh build.

**Why:** A hand edit is lost on the next regeneration, and the generated API client stops matching the API it describes.

## Pull request size

### R11. Prefer small code changes

Split large changes into focused pull requests. As a rough guide, aim for under about 500 changed lines, not counting generated code, lockfiles and docs.

**Why:** Reviews of changes with thousands of lines are slow and miss problems.
