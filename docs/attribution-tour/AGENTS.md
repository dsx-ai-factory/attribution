<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent instructions for the attribution walkthrough

These instructions apply to all files under `docs/attribution-tour/`.

## Before editing

- Read `README.md` and `MAINTAINING.md` completely.
- Treat `docs/attribution.md`, the implementation, and bundled rules as the
  sources of truth for technical claims. Verify behavior before depicting it.
- Keep the story about the shared attribution process; do not make it specific
  to one subordinate project unless the task explicitly requires that.

## Preserve the storyboard model

- Keep the deck as one 1280×720 Slidev page with internal Vue scenes unless a
  requested redesign requires a different architecture.
- Treat `scenes`, `phases`, and `subphaseGroups` as coordinated navigation data.
  When adding a scene, update its template state, styling, phase placement, and
  documentation anchor together.
- Preserve scene IDs because `?scene=<id>` URLs are shared externally. When
  renaming one, add a backward-compatible alias in the URL initialization code.
- Reuse the same DOM element across scenes when an observation, event, or
  symptom is meant to persist; express movement through scene-dependent CSS.
- Keep phase boundaries, per-node/global scope, and no-op rule phases explicit.

## Preserve the experience

- Center the main figure and favor legibility over completeness. Avoid dense
  schemas, small text, overlapping connectors, and decorative complexity.
- Preserve phase color coding and keep motion slow enough to follow. Retain the
  motion controls and the `prefers-reduced-motion` behavior.
- Keep native controls keyboard-accessible. Global navigation must not consume
  Space or arrow-key interaction while a button, link, or form control is
  focused.
- Point each scene at an existing anchor in `docs/attribution.md` through
  `docAnchor`; do not hardcode a second documentation base URL.
- Add the repository-standard NVIDIA SPDX header to new maintained source files.
  Do not edit or commit `dist/`, `.slidev/`, or `node_modules/`.

## Validate

- Run `corepack pnpm build` in `docs/attribution-tour/`.
- Manually review changed scenes plus their incoming and outgoing transitions
  with `corepack pnpm dev`.
- Test scene dots, phase/subphase navigation, source tabs, motion buttons,
  documentation links, keyboard navigation, and shareable `scene`/`motion`
  query parameters as applicable.
- If publishing behavior changes, follow `docs/github-pages.md` and preserve the
  read-only PR build, trusted publishing boundary, stable production path, and
  serialized `gh-pages` updates.
