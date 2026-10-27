<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Repository agent instructions

These instructions apply to the full repository. A more specific `AGENTS.md`
file adds or replaces instructions for the files in its directory tree.

Read `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, and `LICENSE` before you
make a change. For product behavior and architecture, use
`docs/ARCHITECTURE.md`, `docs/attribution.md`, and the implementation as the
sources of truth. Do not copy that material into this file.

## Permitted operations

- Read and search repository files and local Git history.
- Make focused changes that the assigned task requires in an isolated branch
  or worktree.
- Add or update tests and documentation that directly support the change.
- Run the local build, test, lint, formatting, and generation commands that
  apply to the changed files.
- Use committed test data and synthetic data. Keep outputs and generated files
  only when the repository tracks them.
- Inspect the final diff and local Git status before handoff.

These permissions do not include remote publication, production access, or a
license change.

## Human confirmation required

Get explicit human confirmation before you do any of these actions:

- Send any outbound message, including a pull-request or issue comment, email,
  external form submission, vulnerability report, security report, or other
  message. The human must approve the exact destination and exact content
  before transmission.
- Push a branch, open or change a pull request or issue, or publish a package,
  container image, documentation site, or other artifact.
- Merge a change, create or push a tag, or create a release.
- Change license terms, license files, copyright statements, or SPDX
  identifiers.
- Read production data or credentials, connect to a production system, or run
  a command that can change a production or shared external system.
- Delete data, discard uncommitted work, rewrite Git history, force-push, or do
  another action that is difficult to reverse.

Confirmation for an action does not authorize a different action or target.

## Out-of-scope work

- Do not make unrelated refactors, dependency upgrades, formatting sweeps, or
  generated-file changes.
- Do not change public interfaces, data formats, rule behavior, or deployment
  behavior unless the assigned task requires that change.
- Do not modify another repository, a parent workspace, or user configuration.
- Do not operate customer infrastructure, diagnose live clusters, or perform
  incident response from this repository.
- Do not publish, merge, tag, release, relicense, or access production as part
  of routine implementation work.

If required work is outside these limits, stop and ask the human owner.

## Build and test commands

Follow [Run the test gate in CONTRIBUTING.md](CONTRIBUTING.md#run-the-test-gate)
for prerequisites, tool and runtime setup, staged-source requirements, and
full and focused test commands. Maintain those shared requirements in
`CONTRIBUTING.md`.

Run commands from the repository root unless a command says otherwise. Build
the documented bundle image with:

```bash
export ATTRIBUTION_IMAGE="attribution-blueprint-web:local"
docker build \
  -f blueprint/docker/Dockerfile \
  -t "$ATTRIBUTION_IMAGE" \
  "$PWD"
```

For changes under `docs/attribution-tour/`, follow its nested `AGENTS.md` and
run its documented build from that directory:

```bash
corepack pnpm build
```

Run the narrowest applicable suite during development. Before handoff, run the
full gate when the environment has its required tools. Report each command's
actual result and any check that you could not run.

## Commit and PR titles

Pull requests are squash-merged, so the PR title becomes the commit subject on
`main`. Write it as a [Conventional Commit](https://www.conventionalcommits.org/):
`type(scope)!: summary`. CI checks the format and type; the rest is guidance.

- Types: `feat`, `fix`, `build`, `chore`, `ci`, `docs`, `perf`, `refactor`,
  `revert`, `style`, `test`.
- Scope is optional. Common scopes are `attribution-lib`, `service-app`,
  `blueprint`, `frontend`, `docs`, and `ci`.
- Write the summary in the imperative mood with no trailing period.
- Mark a removed or incompatible public API with `!` and add a brief
  `BREAKING CHANGE:` footer to the PR body that says what callers must change.
- The type signals the version bump for the next release of each touched
  component: `!` is major, `feat` is minor, `fix` is patch, and other types do
  not bump. A component still at `0.y.z` shifts each bump down one place:
  `!` is minor, and `feat` and `fix` are patch.

## Secret handling

- Never put credentials, tokens, private keys, customer data, or other secrets
  in source files, tests, fixtures, prompts, logs, commits, or documentation.
- Use environment variables or approved local secret storage. Keep local
  secret files untracked and use clear placeholder values in examples.
- Review staged changes for secrets before commit. Do not bypass the repository
  secret scan or add an exclusion to hide a real secret.
- If you find a possible secret, stop. Do not print, copy, validate, use, or
  transmit it. Notify the human owner in the current task without including the
  possible secret. Do not send an email, external form submission,
  vulnerability report, security report, or other outbound message about it
  unless the human later approves the exact destination and exact content.

## Repository boundaries

- Keep all edits inside this repository and inside the worktree for the
  assigned branch.
- Keep each change within its assigned file ownership. Preserve unrelated
  changes and do not revert another contributor's work.
- Respect nested instruction files. In particular,
  `docs/attribution-tour/AGENTS.md` governs the interactive walkthrough.
- Treat generated content, lock files, and vendored content as owned by their
  documented generator or update process. Do not edit them by hand unless that
  process explicitly requires it.
- Do not commit build outputs, caches, local databases, virtual environments,
  or `node_modules/`.

## License boundaries

NVIDIA Attribution project code is licensed under Apache-2.0. Read and preserve
the closest applicable `LICENSE` and its notices before you edit, copy, move, or
distribute a file. The root `LICENSE` applies only when no closer nested
`LICENSE` applies. Do not assume that the root `LICENSE` alone governs all
files.

The tracked nested license boundaries are:

- `blueprint/LICENSE`
- `service/attribution-lib/LICENSE`
- `service/service-app/LICENSE`
- `service/service-app/nvidia_attribution_client/LICENSE`
- `service/service-app/openapi/extras/LICENSE`
- `service/service-app/openapi/templates/LICENSE`

For files under `blueprint/`, read `blueprint/LICENSE`. For files under
`service/`, read the closest applicable service license and preserve its
license notices.

Preserve third-party licenses and the exceptions described in
`THIRD_PARTY_NOTICES.md`. Changes to license terms and notices require
explicit human approval. Distribute material only under its applicable license.
Do not publish a package, image, release, or source bundle without
explicit human approval and the required NVIDIA authorization. Human approval
does not override the license.
