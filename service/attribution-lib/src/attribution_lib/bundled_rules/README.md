# bundled_rules

Rule definitions loaded at runtime: the extractors, refiners, and filters that
turn raw observations into `symptom` strings, plus the declarative symptom
defaults.

- Extractor/refiner/filter rules (`applog_rules.py`, `syslog_rules.py`,
  `node_metrics_rules.py`, `refiner_rules.py`,
  `filter_rules.py`, ...). `common/xid_errors.py` holds the XID/SXID ground-truth
  catalog.
- `symptom_defaults.py` declares the symptom vocabulary, outcome priority tiers,
  and cascade metadata.

## Symptom → outcome mapping

There is no longer a `symptom_to_outcome` table. The symptom vocabulary, outcome
priority tiers, and cascade metadata are declared at runtime from
`symptom_defaults.py`. The generated symptom registry consumed at runtime is built
from those declarations
(`../symptoms/registry_generation.py::registry_from_declarations`).

Declarations use three helpers from `attribution_lib.rules_engine`:

- `outcome_symptom(prefix, *, outcome, priority, doc, ...)` — a symptom that can
  drive final outcome selection. `outcome` uses the library's raw compatibility taxonomy; `priority` is an
  `OutcomePriority` tier (lower wins).
- `cascade_symptom(prefix, *, cascade_discards, scope=CascadeScope.NODE, ...)` — a
  cascade-only symptom used for pruning, not outcome selection. Use
  `scope=CascadeScope.NVL_DOMAIN` when the targets should be pruned from sibling
  nodes in the same NVL domain; NVL-domain targets may use shell-style wildcards.
- `known_symptom(prefix, *, fallback_prefix=False)` — a symptom that belongs to
  the vocabulary only.

Set `category=SymptomCategory.CSP_EVENT` for CSP event symptoms. Runtime code and
tests can query that membership with `registry.symptoms_in_category(...)`
instead of matching the symptom text.

## How to add a symptom

1. Emit the symptom string from the rule that produces it. Most symptoms come
   from the extractor rules in this folder (e.g. a keyword mapping in
   `applog_rules.py`), but refiner rules can also synthesize
   a symptom from ones already present (e.g. `refiner_rules.py` emits
   `node offline` by combining the NS connection lost, link down, and FM lost GPU
   symptoms).
2. Declare it in `symptom_defaults.py`:
   - use `outcome_symptom(...)` if the symptom should select an outcome,
   - use `cascade_symptom(...)` if it only prunes other symptoms,
   - use `known_symptom(...)` if it is vocabulary-only.
3. Use the exact same symptom string in both places (the declaration's `prefix`
   is matched against emitted symptoms via `startswith`).

See `../../../docs/using-the-symptom-registry.md` for the registry API and how to
test symptoms through it (and why new symptoms never go in the legacy
`RegisteredSymptoms` enum).
