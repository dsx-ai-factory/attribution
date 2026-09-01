<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent Guide: Add a New Extractor Rule

Use this procedure to add an extractor rule to the NVIDIA Attribution rules engine with a minimal, evidence-backed diff.
Do not apply it to filter, refiner, collector, or other rule types; use their nearest implementation and tests instead.
This procedure permits local repository edits and local verification only.
Do not push a branch, create a pull request, or perform another remote write unless the user explicitly requests it.

Run commands from `service/attribution-lib` unless a step states otherwise.
Read [`adding-a-rule.md`](adding-a-rule.md) before using this procedure for the first time.

## Completion Criteria

Complete the task only when all of the following statements are true.

- The new extractor has a unique, stable name and uses the correct observation source.
- An appropriate focused test demonstrates the new behavior and passes; a specialized test failed before implementation when that test-first path was applicable.
- An ordinary declarative extractor has complete co-located cases, and a separate behavior test exists only when the inline case schema cannot express the required assertion.
- Every emitted symptom resolves through the symptom registry with approved metadata.
- The complete Attribution Library test gate passes, and the diff contains no unrelated changes.

## Step 1: Establish Scope

Inspect the checkout before editing.

```bash
git status --short --branch
grep -R -n -F 'distinctive input text' src/attribution_lib/bundled_rules
grep -n -F 'proposed symptom text' src/attribution_lib/bundled_rules/symptom_defaults.py
```

Preserve all pre-existing user changes.
Do not rename or rewrite an existing rule unless the request requires that behavior.

Record these acceptance criteria in the working notes.

1. Record the observation source and input format.
2. Record at least one exact matching input and one relevant near-miss.
3. Record the emitted event type, symptom, expected node or resources, and credited rule name.
4. Record whether the symptom is existing or new.
5. Record the approved outcome metadata when the symptom is new and outcome-bearing.

Stop and ask one concise question if a missing item requires a semantic decision.
In particular, do not infer whether evidence is `bad_node` or `interesting_node`, and do not invent outcome or cascade metadata.

## Step 2: Select the Implementation Path

Use the smallest path that implements the required behavior.
Before authoring new declarative syntax, consult the accepted keys in [`declarative/loader.py`](../src/attribution_lib/declarative/loader.py) and the emission forms in [`declarative/emit.py`](../src/attribution_lib/declarative/emit.py).

1. Use a declarative leaf for one fixed `bad_node`, `interesting_node`, or `generic_event` emission.
2. Use `pick_one`, `try_all`, `catalog_lookup`, or `family` only when the requested control flow directly requires that form.
3. Use a Python processor when the rule requires secondary parsing, state, context lookup, or a typed payload.
4. Use an existing Python rule pattern for a filter, refiner, collector, or extractor accumulator.

For a declarative rule, edit the nearest source-specific TOML file under `src/attribution_lib/bundled_rules/`.
Do not create a new file when an existing file already owns that source and rule shape.

For a Python processor, register a module-level function with `@rule_processor("namespace:name")` in a loaded top-level `*.py` file.
Reference it from a TOML `[[extractor]]` through `processor = "namespace:name"` and declare every possible `derived_rules` and `derived_event_types` value.
Follow the nearest processor and its tests rather than introducing a new abstraction.

## Step 3: Choose the Test Layer

For an ordinary declarative extractor, use co-located `[[extractor.cases]]` as the primary test layer.
Add the matching, near-miss, and branch cases with the extractor in Step 4; the production catalog loader executes and validates them.

Use an existing top-level `*.test.toml` file for a Python-based extractor or a special compound or parity scenario that an existing data runner supports.
The available runner schemas are defined in `tests/rule_data_runner.py` and executed by `tests/test_rule_data.py`.
For example, a case for the `applog_single_line` runner has this shape:

```toml
[[cases]]
description = "checkpoint permission failure identifies the application rule"
line = "checkpoint save failed: Permission denied"
expect = "bad"
expected_symptom = "checkpoint permission denied"
expected_rule = "ApplogCheckpointPermissionDenied"
```

Use a co-located `*_test.py` module under `src/attribution_lib/bundled_rules/tests/` when TOML cannot express the assertion.
Use shared fixtures from `test_support/` instead of importing test helpers from `tests/`.

When using a `*.test.toml` or Python behavior test, run only the new behavior test before implementation and confirm that it fails for the expected missing-rule behavior.
Replace `YOUR_DISTINCTIVE_TEST_TOKEN` with a distinctive token from the new test description.

```bash
test_token='YOUR_DISTINCTIVE_TEST_TOKEN'
uv run --frozen pytest -n 4 tests/test_rule_data.py -q -k "$test_token"
```

Do not continue with a specialized test that passes before the implementation.
Find whether an existing rule already handles the case or whether the test failed to select the intended case.
For an ordinary declarative extractor, record that the co-located cases will be validated after the extractor is added; do not create a duplicate `*.test.toml` case solely to manufacture a pre-implementation failure.

## Step 4: Implement the Minimal Rule

For a declarative leaf, add the extractor and its inline coverage cases together.

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

Apply these constraints while editing.

- Inherit file-level `author` and `sources` values when editing an existing TOML file.
- When creating a file, set `author` to the contributor's name, GitHub handle, team, or organization. Never assign `author = "NVIDIA Attribution team"` for an external contributor; use it only when NVIDIA maintainers accept ownership and ongoing maintenance. Use enum member names such as `sources = ["APPLOG"]`.
- Use named regular-expression captures and reference them with `{name}` only where the emitted template needs them.
- Omit `derived_rule` when provenance should use the extractor name. Set it only to preserve an established provenance slug or to give generated family rows distinct credited rules.
- Do not use `untested` to bypass ordinary inline branch coverage.

For `when`, include inline cases that make the expression both true and false.
For `pick_one`, reach every branch and the default.
For `try_all`, exercise both sides of every guarded event.

## Step 5: Register the Symptom

Skip this step only when existing registry metadata gives the emitted symptom the intended vocabulary, outcome, and cascade behavior.
Verify exact declarations, fallback-prefix resolution, and outcome-prefix behavior in `src/attribution_lib/bundled_rules/symptom_defaults.py` before deciding to skip it.

Add the smallest approved declaration when the symptom is new.

- Use `outcome_symptom(...)` for a symptom that can select a final outcome.
- Use `known_symptom(...)` for vocabulary-only evidence.
- Use `cascade_symptom(...)` only for symptom-pruning metadata.

Do not edit the legacy `RegisteredSymptoms` enum.
Do not copy outcome metadata from a textually similar symptom unless checked-in behavior or an accepted design decision establishes the same semantics.
Stop and request the missing decision when `outcome`, `OutcomePriority`, `doc`, `hw_cause`, or cascade behavior is not established.

## Step 6: Reach Green on Focused Validation

Load the production catalog to catch TOML syntax, duplicate names, invalid sources, invalid regular expressions, unresolved expressions, incomplete inline coverage, and symptom declaration errors.

```bash
uv run --frozen python -c 'from attribution_lib.core import setup_rules; setup_rules()'
```

If Step 3 required a `*.test.toml` behavior test, run it again.
Replace `YOUR_DISTINCTIVE_TEST_TOKEN` with the distinctive token from that new test description.

```bash
test_token='YOUR_DISTINCTIVE_TEST_TOKEN'
uv run --frozen pytest -n 4 tests/test_rule_data.py -q -k "$test_token"
```

Skip this pytest command for an ordinary declarative extractor whose behavior is fully covered by the co-located cases already validated during catalog loading.
If the specialized test uses a co-located Python module, run that exact module instead.
Fix the implementation, not the expected result, unless source evidence proves that the expectation was wrong.

## Step 7: Verify the Catalog and Full Suite

Capture the loaded rule inventory.

```bash
uv run --frozen python scripts/rule_inventory.py dump -o /tmp/attribution-rules.json
rule_name='YOUR_NEW_RULE_NAME'
grep -n -F "$rule_name" /tmp/attribution-rules.json
```

Inspect the inventory entry and verify the rule name, source, matcher, author, tags, form, and emit or processor definition.

Run the complete Attribution Library gate from the repository root.

```bash
(cd ../.. && bash .ci/run-tests.sh attribution-lib)
```

Run Ruff only when the change includes Python.

```bash
uv run ruff format --check --config ruff.toml
uv run ruff check --config ruff.toml
```

Inspect the final local diff.

```bash
git diff --check
git status --short
git diff -- .
```

Report the created or modified rule files, test files, symptom declarations, and exact verification results.
Leave the changes uncommitted unless the user requested a commit.
Never push without explicit user authorization.

## Stop Conditions

Stop and ask for direction when any of these conditions applies.

- The requested evidence can reasonably map to more than one symptom or outcome.
- A new match overlaps an existing rule, and the expected coexistence or precedence is unclear.
- The implementation requires a new rule type, event type, expression helper, test runner, or engine behavior.
- The full test gate reveals a failure outside the changed rule and its direct metadata.
- Three attempted fixes fail, which makes the assumed input, source, or expected outcome doubtful.
