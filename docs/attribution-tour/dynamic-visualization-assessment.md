# Execution-driven attribution visualization

**Status:** Deferred assessment; not an active implementation plan.

## Summary

An interactive visualization driven by an actual attribution run is feasible
without redesigning the attribution engine. The repository already has an
optional decision recorder and can assemble a presentation-independent decision
DAG. The current storyboard could therefore evolve from a hard-coded example
into a viewer for trace data produced by the attribution engine.

The main work would be completing the execution trace and making the visual
presentation cope gracefully with arbitrary input sizes. A small, honest
prototype is a moderate project. A general hosted tool would be substantially
larger because it introduces service operations, authorization, and log-data
security concerns.

## Desired experience

Given a completed job record and an ordered collection of observations, the
tool would run the real attribution implementation and let a viewer step through
the result:

1. Initialize the attributor with job state and topology.
2. Ingest each observation.
3. Show extractor matches and the events they emit.
4. Replay each filter stratum as a stream-to-stream transformation.
5. Materialize per-node symptoms, run SCRUB rules, and aggregate globally.
6. Replay FINAL rules, outcome mapping, and POST_OUTCOME rules.
7. Show collectors and the final `AttributionResult`.

Objects should retain stable visual identities throughout the sequence. For
example, a source line should visibly become an event, an event should move to
its node during materialization, and a symptom should remain traceable through
demotion or removal.

## Existing foundation

The implementation is already partly instrumented:

- `Attributor.attribute()` accepts an optional `DecisionRecorder`.
- `DecisionRecorder` records ordered actions and their evidence.
- Refiner execution compares state before and after each rule and records
  changes.
- Cascade pruning and outcome selection emit trace events.
- `build_decision_graph()` combines recorded decisions with source provenance
  from the final result to produce a domain-level DAG.

This is enough to explain many final decisions, but it is an explanation trace,
not yet a complete replay of execution.

## Missing trace information

To drive the full storyboard faithfully, trace mode would also need to record:

- Initialization and observation-ingestion boundaries.
- Extractor matches and emitted events as they happen, including accumulator
  lifecycles where useful.
- Before/after event-stream state for each filter rule.
- Explicit execution of rules that make no change. This is necessary to show
  that a phase ran and no rule fired, rather than merely omitting the phase.
- Materialization and global-aggregation boundaries.
- Collector execution and per-rule outputs.
- Final state changes made by POST_OUTCOME rules in a form suited to replay.

Filter rules can perform arbitrary event-table mutations, so the safest initial
instrumentation would compare event identities and relevant fields before and
after each rule. Because tracing would be optional, this additional work need
not affect the normal attribution path.

The existing decision DAG also connects evidence to decisions more than it
models the sequence between decisions. Animated replay needs both: causal edges
for explanation and ordered state transitions for presentation.

## Proposed boundary

The visualizer should consume a versioned, serializable trace rather than know
about attribution internals:

```text
job record + observations
            |
            v
attribution engine in trace mode
            |
            v
versioned trace JSON
            |
            v
Vue/Slidev trace viewer
```

A trace step would identify its phase, sub-phase, rule, sequence number, input
and output object IDs, mutations, evidence, and result. Stable IDs would allow
the browser to animate an element between layouts instead of destroying and
recreating it on every frame.

This separation would support two execution models:

- **Generated documentation:** Run attribution during the documentation build
  against checked-in, sanitized fixtures and publish the resulting trace JSON
  with the static GitHub Pages site.
- **Arbitrary input:** Run attribution through a local or hosted Python service
  and return trace JSON to the same viewer.

GitHub Pages can host and animate previously generated traces, but it cannot run
the Python and DuckDB attribution implementation for newly uploaded jobs.

## Keeping real traces understandable

A literal replay of every event and every rule would quickly become an
eyechart. The viewer should expose two related views:

- **Story view:** Default to the causal slice that contributes to the outcome,
  group repeated evidence, and collapse runs of unchanged rules.
- **Full trace:** Allow inspection of every rule invocation, including
  unchanged phases and evidence that did not contribute to the outcome.

Large observations should be represented by selected matching lines plus local
context, not by rendering entire logs. Large event and symptom collections need
grouping by type and node, with drill-down on demand.

## Rough scope

These are directional estimates for one engineer familiar with the codebase:

- **Small execution-driven example:** approximately 4–7 engineering days. Add
  the missing trace events needed by one representative fixture, serialize the
  trace, and make the current storyboard data-driven.
- **Useful local explorer:** approximately 2–3 weeks. Accept arbitrary supported
  inputs, handle variable numbers of observations and nodes, add trace
  navigation and grouping, and cover the full phase lifecycle.
- **Hosted production tool:** additional work beyond the explorer for an API,
  authentication, resource limits, persistence, redaction, auditing, and safe
  handling of potentially sensitive logs.

The estimates depend strongly on whether the first version must cover every
rule type and large production jobs or may intentionally target small traces.

## Risks and constraints

- Attribution behavior and the trace schema must be versioned together so a
  viewer does not silently misinterpret older runs.
- Extraction caching and replay should report what actually happened, including
  restored versus newly extracted data.
- Trace snapshots can be expensive; trace mode should remain opt-in and may
  need limits for large jobs.
- Raw source data may contain sensitive material. Published examples must use
  sanitized fixtures, and an upload-based service would require a deliberate
  data-retention and access-control design.
- Presentation summaries must never invent causality. Any collapsed story view
  should remain expandable into the underlying trace events.

## Recommended first milestone if revisited

Use one existing, sanitized test fixture and run it through the real attribution engine to
produce a checked-in or build-generated trace JSON file. Refactor the existing
storyboard to render that trace while preserving its current visual language.

That milestone would validate the trace contract, stable visual identities, and
causal-story reduction before introducing arbitrary uploads or a hosted
execution service.
