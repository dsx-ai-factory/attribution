<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Add a New Extractor Rule

## Scope: Extractor Rules

NVIDIA Attribution's rules engine has several stages with distinct rule types: extractors recognize input evidence, filters correlate or suppress the resulting events, and refiners adjust symptoms and attribution state in phases before and after outcome selection.
This guide covers only **extractor rules**.

An extractor is a first-stage rule that recognizes a repeatable piece of workload evidence and emits a structured event.
For example, an extractor can recognize a checkpoint error in an application log and emit an event that records the error against the node that produced the message.
Later rule stages can correlate or suppress these events, turn them into symptoms (stable labels for recognized evidence), and refine the attribution result around outcome selection.

Adding a rule here means changing the checked-in bundled rule catalog and shipping that change with the Attribution Service.
It is not a rule uploaded at runtime through an API or user interface.
Propose changes through [CONTRIBUTING.md](../../../CONTRIBUTING.md).
The Library does not offer a stable external interface. Upstream bug reports
must reproduce on the unmodified main branch, as described in
[SUPPORT.md](../../../SUPPORT.md).

## Why Add an Extractor Rule?

The engine can recognize only the evidence patterns represented in its rule catalog.
Add or extend an extractor rule when available workload evidence contains a reliable pattern that the extractor catalog does not yet represent correctly.
Common reasons include:

- A new application, driver, platform, or software version reports a known failure with a previously unseen log or metric signature.
- A signal describes a new symptom that Attribution should retain when it determines affected nodes or the final diagnosis.
- A later correlation rule needs a structured event that is not currently extracted.

A rule is not always the right change.
If evidence is not reaching the Service, add or modify the connector that collects and submits it.
If the Service cannot parse the submitted source or format, add support for that input instead.
Extend an existing rule when the new input has the same meaning and output as evidence it already handles.

## Key Terms

- **Observation:** One submission of related evidence plus context describing its source, format, time interval, and timezone. Each entry can name the resources it concerns, such as a node. An observation can contain application-log lines, syslog entries, Prometheus samples, or cloud-provider events.
- **Observation source:** The kind of evidence in an observation, such as `APPLOG`, `SYSLOG`, `NODE_METRICS`, or `CSP_EVENT`. Extractor rules declare which sources they accept.
- **Observation entry:** One item within an observation, such as one log line or metric sample. Most extractors evaluate entries individually.
- **Extractor rule:** A rule that looks for a pattern in observation entries and emits a structured event when it matches. A simple extractor can be declarative TOML, where the pattern and output are configuration data. More complex processing can use Python referenced from TOML.
- **Event:** An internal record created from matched evidence. It carries a type and payload together with provenance such as the timestamp, affected resources, matching log line or metric sample, observation context, and credited rule.
- **Symptom:** A stable label for recognized evidence, such as `checkpoint permission denied`. Bad-node and interesting-node events carry symptoms so different raw inputs can be compared consistently.
- **Bad-node event:** Evidence that associates a symptom with a node suspected of contributing to the workload failure.
- **Interesting-node event:** Node-associated evidence retained as context that does not, by itself, mark the node as bad.
- **Generic event:** A typed fact with an arbitrary payload for later rules to consume. It does not directly emit a bad-node or interesting-node symptom.
- **Symptom registry:** The list of known symptom prefixes and metadata that tells later stages whether a symptom can influence the outcome or suppress less useful symptoms caused by the same failure.
- **Outcome or verdict:** The final workload-level diagnosis selected after the engine combines and refines the available evidence.

## How Extractor Rules Fit into Attribution

At a high level, evidence moves through this pipeline:

`raw evidence` → `observations` → `extractor rules` → `events` → `filter rules` → `symptoms linked to nodes` → `pre-outcome refiners` → `outcome` → `post-outcome refiners` → `attribution result`

For example, an `APPLOG` observation might contain hundreds of lines from one workload.
If one entry says `checkpoint save failed: Permission denied`, an extractor can emit a bad-node event with the symptom `checkpoint permission denied`.
The event retains the matching line, affected node, and rule name as provenance.
Later stages combine that symptom with other evidence and decide whether it contributes to the final diagnosis.

The concrete processing flow is:

1. A client, commonly the Blueprint after a connector collects evidence, submits an observation and its context to the Attribution Service.
2. The library converts the observation into source-specific entries and applies rules that accept that source.
3. A matching extractor emits a bad-node event, an interesting-node event, or a generic event.
4. Filter rules correlate or suppress events before the engine turns them into node symptoms.
5. Pre-outcome refiners combine symptoms, change whether they implicate a node, or discard redundant evidence. They use the symptom registry's vocabulary and metadata where needed.
6. The remaining symptoms determine the outcome.
7. Post-outcome refiners can adjust node attribution and other result state after the outcome is known. The final attribution result retains the provenance chain that supports it.

At startup, `setup_rules()` loads every top-level `*.py` and `*.toml` file under `src/attribution_lib/bundled_rules/`.
Files ending in `*.test.toml` are test data and are not loaded as production rules.
You do not need to add a declarative rule to a separate registration list.

This guide is the canonical workflow for adding extractor rules.
The [`bundled_rules` README](../src/attribution_lib/bundled_rules/README.md) is a focused reference for the symptom registry and links back to this workflow.

This guide assumes that you are comfortable editing Python and TOML, but that you have not worked in this codebase.
Run all commands from `service/attribution-lib` unless a command states otherwise.

## Gather the Required Evidence

Define these inputs before you edit a rule file.

- Record the observation source and at least one exact input that must match.
- Record a similar input that must not match when false positives are possible.
- Decide whether the evidence identifies a bad node, provides interesting context, or emits a non-attribution event.
- Reuse an existing symptom when it has the same meaning, or obtain an approved outcome, priority, and documentation anchor for a new symptom.
- Decide which rule name must appear in provenance as `source.rule`.

Search the existing catalog before adding a new rule or symptom.

```bash
grep -R -n -F 'distinctive text from the observation' src/attribution_lib/bundled_rules
grep -n -F 'proposed symptom text' src/attribution_lib/bundled_rules/symptom_defaults.py
```

Extend an existing rule when the evidence has the same meaning and output.
Create a new rule when the evidence needs a distinct matcher, symptom, provenance name, or test boundary.

## Choose the Smallest Rule Form

Prefer declarative TOML for fixed matching and emission behavior.
Consult the accepted keys in [`declarative/loader.py`](../src/attribution_lib/declarative/loader.py) and the emission forms in [`declarative/emit.py`](../src/attribution_lib/declarative/emit.py) when you need details beyond the examples below.
Use the following mapping to choose a form.

| Required Behavior | Rule Form | Example |
| --- | --- | --- |
| Emit one event after one regular-expression match. | Use a declarative leaf with `bad_node`, `interesting_node`, or `generic_event`. | [`applog_leaf_rules.toml`](../src/attribution_lib/bundled_rules/applog_leaf_rules.toml). |
| Choose the first matching result from ordered conditions. | Use `pick_one`. | [`applog_combinator_rules.toml`](../src/attribution_lib/bundled_rules/applog_combinator_rules.toml). |
| Emit several independently guarded events from one match. | Use `try_all`. | [`syslog_combinator_rules.toml`](../src/attribution_lib/bundled_rules/syslog_combinator_rules.toml). |
| Map a captured key through a table. | Use `catalog_lookup`. | [`syslog_sxid_rule.toml`](../src/attribution_lib/bundled_rules/syslog_sxid_rule.toml). |
| Generate similar literal-match rules from rows. | Use a `family`. | [`generic_keyword_family.toml`](../src/attribution_lib/bundled_rules/generic_keyword_family.toml). |
| Perform secondary parsing, stateful work, context lookup, or typed payload construction. | Use a registered Python processor referenced by a TOML extractor. | [`applog_processor_rules.toml`](../src/attribution_lib/bundled_rules/applog_processor_rules.toml). |

Filters, refiners, collectors, and extractor accumulators remain Python rule types.
Follow the nearest existing implementation and its co-located tests when your change belongs to one of those stages.

## Add a Declarative Leaf

Add a new stanza to the existing source-specific leaf file when one exists.
The following example detects an application-log error and emits one bad-node symptom.

```toml
[[extractor]]
name     = "ApplogCheckpointPermissionDenied"
match_re = 'checkpoint save failed: Permission denied'
emit     = { type = "bad_node", symptom = "checkpoint permission denied" }

[[extractor.cases]]
line    = "checkpoint save failed: Permission denied"
expect  = "bad"
symptom = "checkpoint permission denied"

[[extractor.cases]]
line   = "checkpoint save completed"
expect = "none"
```

The enclosing TOML file normally provides file-level `author` and `sources` values.
If you create a new file, set `author` to the contributor identity that should appear in the rule inventory, such as your name, GitHub handle, team, or organization.
External contributors must not use `author = "NVIDIA Attribution team"`; NVIDIA maintainers use that value only when their team accepts ownership and ongoing maintenance of the rule.
Declare the required `sources` by enum member name.
The available member names are defined by `Source` in `src/attribution_lib/observation/inbound.py`.

Keep `name` unique and stable because inventories, tests, and provenance expose it.
If `emit.derived_rule` is absent, the emitted event receives the extractor `name` as its credited rule.
Set `derived_rule` only when the required `source.rule` value intentionally differs from the extractor name, such as when preserving an established provenance slug while porting a Python rule or when a generated family gives each row its own credited rule.
Do not add it to an ordinary new extractor whose credited rule should be its `name`.

Use named regular-expression captures when the symptom or a condition needs a matched value.
Reference a capture in a symptom template with braces, as in `symptom = "error code {code}"`.
Use a bare expression in `when`, as in `when = "int(code) > 0"`.
The loader rejects unknown names, invalid regular expressions, unsafe optional-capture access, and unsupported keys.

## Test a Declarative Extractor with Co-located Cases

Each declarative `emit` rule must contain inline `[[extractor.cases]]` entries or an explicit `untested` waiver.
Do not use `untested` for ordinary rule additions.

Inline cases are the primary tests for ordinary declarative extractors and are validated when the catalog loads.
They exercise matching and expected emissions for every leaf guard, `pick_one` branch and default, and independently guarded `try_all` event.
A leaf with `when` needs cases for both the true and false sides.
A leaf without `when` needs at least one emitting case.

Inline cases do not exercise the complete attribution pipeline.
Add another test layer only when the behavior requires assertions that the inline case schema cannot express.

## Add a Specialized Behavior Test When Needed

Top-level `*.test.toml` files under `src/attribution_lib/bundled_rules/` support Python-based extractors and special compound or parity scenarios through centralized data runners.
Do not add one merely to duplicate the co-located cases for an ordinary declarative extractor.
When a Python-based or compound case fits an existing runner, add it to the existing source-specific `*.test.toml` file.
The centralized discovery and runners live in `tests/rule_data_runner.py` and `tests/test_rule_data.py`.

The following example uses the `applog_single_line` runner for a case that needs that runtime layer.

```toml
[meta]
runner = "applog_single_line"

[[cases]]
description = "checkpoint permission failure identifies the application rule"
line = "checkpoint save failed: Permission denied"
expect = "bad"
expected_symptom = "checkpoint permission denied"
expected_rule = "ApplogCheckpointPermissionDenied"

[[cases]]
description = "successful checkpoint does not match the permission failure rule"
line = "checkpoint save completed"
expect = "no_match"
```

Use a co-located Python test when the case requires mocks, multi-step event choreography, or detailed event-stream inspection.
Keep bundled-rule Python tests under `src/attribution_lib/bundled_rules/tests/` so production rule discovery does not load them.

These focused tests do not replace a future end-to-end test that starts with a complete job definition and inputs and asserts the final attribution result.
The repository does not currently provide that test layer for new extractor rules, so do not claim end-to-end coverage from inline cases or the rule-data runners.

## Register a New Symptom

Skip this step only when existing registry metadata already gives the emitted symptom its intended vocabulary, outcome, and cascade behavior.
Check exact declarations, approved fallback prefixes, and outcome prefixes before deciding to skip it.
Add a new declaration to `src/attribution_lib/bundled_rules/symptom_defaults.py` when the symptom is new.

Choose the declaration helper based on the symptom's role.

- Use `outcome_symptom(...)` when the symptom can select the final outcome.
- Use `known_symptom(...)` when the symptom belongs to the vocabulary but does not select an outcome.
- Use `cascade_symptom(...)` when the declaration only defines pruning behavior.

Do not add new symptoms to the legacy `RegisteredSymptoms` enum.
Do not guess an outcome, `OutcomePriority`, hardware cause, cascade edge, or documentation anchor.
These values change the final diagnosis and require evidence or an accepted domain decision.

An undeclared symptom is preserved as evidence with an `unknown-symptom-` prefix and does not produce the intended classification.
Outcome selection matches declaration prefixes with `startswith`, so verify that a new prefix does not unintentionally shadow another declaration.

### Document the Symptom

For a new outcome-bearing symptom, complete this checklist alongside the declaration:

1. Set `doc` to an existing documentation ID or create the corresponding page in the documentation source.
2. Make the declaration's exact `prefix` appear in that page's ID, title, or aliases.
   Matching preserves case, punctuation, whitespace, and Unicode characters; it does not convert names into URL slugs.
   For example, `XID-48-` does not match `xid-48` or `XID 48` unless `XID-48-` is explicitly declared as an alias.
3. Check that the page actually explains the symptom's evidence, classification, and limitations.
   An alias means the page covers that name, not that symptoms share an outcome or are deduplicated at runtime.
   If an existing page cannot accurately cover the new symptom, create a separate page.
4. Regenerate the documentation manifest using the documentation project's generator; do not hand-edit generated output.
   Redirect IDs resolve to the target page's declared names and their own exact ID.
5. Run the documentation integration check against the intended manifest and the proposed declarations.
   Do not add new symptoms to the historical baseline. Baseline entries are exact `(prefix, doc)` pairs;
   remove an entry when that declaration disappears or its documentation resolves.

For example, a declaration with `prefix="XID-48-"` and `doc="xid-48"` can use this manifest entry:

```json
{
  "id": "xid-48",
  "title": "XID 48",
  "aliases": ["XID-48-"],
  "redirect_ids": []
}
```

The pure checks live in [`symptoms/doc_validation.py`](../src/attribution_lib/symptoms/doc_validation.py).
The integration layer supplies the live declarations, generated manifest, and historical baseline;
library unit tests alone do not validate a documentation deployment.
Cascade-only declarations, generic fallback prefixes, and declarations without an outcome are exempt from this check.
Passing the name check verifies an explicit mapping, not the technical correctness of the page, which still requires review.

## Verify the Rule

For an ordinary declarative extractor, load the catalog to execute its co-located cases.

```bash
uv run --frozen python -c 'from attribution_lib.core import setup_rules; setup_rules()'
```

If you added a `*.test.toml` or Python behavior test, run that focused test too.
Replace `YOUR_DISTINCTIVE_TEST_TOKEN` with a distinctive token from the new test description.

```bash
test_token='YOUR_DISTINCTIVE_TEST_TOKEN'
uv run --frozen pytest -n 4 tests/test_rule_data.py -q -k "$test_token"
```

Capture an inventory and confirm that the new rule is present with the expected matcher, source, author, tags, and emit definition.

```bash
uv run --frozen python scripts/rule_inventory.py dump -o /tmp/attribution-rules.json
rule_name='YOUR_NEW_RULE_NAME'
grep -n -F "$rule_name" /tmp/attribution-rules.json
```

Run the complete Attribution Library test gate from the repository root.

```bash
(cd ../.. && bash .ci/run-tests.sh attribution-lib)
```

Review the final diff and confirm that every changed line belongs to the rule, its symptom metadata, its tests, or its documentation.

```bash
git diff --check
git diff -- .
```
