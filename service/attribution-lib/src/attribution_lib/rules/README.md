# rules

This folder holds rule logic that is codified as Python and can't be expressed
as a table.

- `csp_events_to_symptom.py`, maps CSP maintenance/fault event categories to a
  `symptom` (and decides which categories are actionable vs. skipped).

The extractor rules that turn raw observations into `symptom` strings live in
`../bundled_rules/` (for example `applog_rules.py`, `epilog_rules.py`,
`syslog_rules.py`, `node_metrics_rules.py`, and `common/xid_errors.py` for the
XID/SXID ground-truth catalog).

## Symptom → outcome mapping

There is no longer a `symptom_to_outcome` table. The symptom vocabulary, outcome
priority tiers, and cascade metadata are declared at runtime from
`../bundled_rules/symptom_defaults.py`. The generated symptom registry consumed
at runtime is built from those declarations
(`symptoms/registry_generation.py::registry_from_declarations`).

Declarations use three helpers from `attribution_lib.rules_engine`:

- `outcome_symptom(prefix, *, outcome, priority, doc, ...)` — a symptom that can
  drive final outcome selection. `outcome` is one of `Fail InfraSW`,
  `Fail InfraHW`, `Fail User`, `Incomplete Info`, etc.; `priority` is an
  `OutcomePriority` tier (lower wins).
- `cascade_symptom(prefix, *, cascade_discards, ...)` — a cascade-only symptom
  used for pruning, not outcome selection.
- `known_symptom(prefix, *, fallback_prefix=False)` — a symptom that belongs to
  the vocabulary only.

## How to add a symptom

1. Emit the symptom string from the rule that produces it. Most symptoms come
   from the extractor rules in `../bundled_rules/` (e.g. a keyword mapping in
   `applog_rules.py` or `epilog_rules.py`), but refiner rules can also synthesize
   a symptom from ones already present (e.g. `refiner_rules.py` emits
   `node offline` by combining the NS connection lost, link down, and FM lost GPU
   symptoms).
2. Declare it in `../bundled_rules/symptom_defaults.py`:
   - use `outcome_symptom(...)` if the symptom should select an outcome,
   - use `cascade_symptom(...)` if it only prunes other symptoms,
   - use `known_symptom(...)` if it is vocabulary-only.
3. Use the exact same symptom string in both places (the declaration's `prefix`
   is matched against emitted symptoms via `startswith`).
