<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->

# Maintaining the attribution walkthrough

This is the reference for the walkthrough's architecture, editing conventions,
and review gate. If Node, Corepack, Slidev, or Vue are unfamiliar, start with
[Editing the walkthrough: a beginner's tutorial](TUTORIAL.md).

## Architecture

The walkthrough is one 1280×720 Slidev page containing a Vue storyboard. The
navigation controls change the storyboard's internal scene; they do not advance
between ordinary Slidev slides.

```text
slides.md
  └─ mounts <AttributionStoryboard />
       ├─ script: example data, scene order, and navigation state
       ├─ template: visible cards, labels, connectors, and controls
       └─ CSS: layout and transitions between scene states
```

`currentIndex` selects an entry from `scenes`. Vue bindings add state-dependent
classes and attributes to the template, and `styles/index.css` moves or reveals
the corresponding elements. Reusing an element across states is what makes an
observation, event, or symptom appear to travel through the process.

## Source map

| Concern | Source |
| --- | --- |
| Deck metadata and canvas size | `slides.md` |
| Scene copy and order | `scenes` in `components/AttributionStoryboard.vue` |
| Major phase navigation | `phases` in `AttributionStoryboard.vue` |
| Filtering and refinement rails | `subphaseGroups` in `AttributionStoryboard.vue` |
| Observation and event examples | `observations` and `events` in `AttributionStoryboard.vue` |
| Diagram structure and behavior | The `<template>` in `AttributionStoryboard.vue` |
| Position, color, and motion | `styles/index.css` |
| Written explanation and anchors | `docs/attribution.md` |
| Viewer controls and URL parameters | `README.md` |
| PR previews and production publishing | `docs/github-pages.md` |

Slidev discovers `components/` and `styles/index.css` by convention. See its
[directory-structure reference](https://sli.dev/custom/directory-structure)
for details.

## Editing conventions

- Verify technical claims against `docs/attribution.md`, the implementation,
  and bundled rules. The walkthrough explains behavior; it does not define it.
- Keep each scene focused on one idea. Make phase boundaries, per-node versus
  global scope, and sub-phases in which no rule fires visible.
- Reuse the same template element when evidence persists across scenes. Animate
  it with scene-dependent classes instead of replacing it with a duplicate.
- Preserve scene IDs when possible because `?scene=<id>` URLs are shared. If an
  ID changes, add an alias in the `onMounted` URL initialization logic.
- Give every scene a `docAnchor` that exists in `docs/attribution.md`.
- Keep the example about the shared attribution process rather than a single
  subordinate project unless the change specifically calls for that scope.
- Center the main figure and favor legibility over completeness. Avoid small
  text, dense schemas, overlapping connectors, and decorative complexity.
- Use `--motion-fast`, `--motion-medium`, and `--motion-slow` for transitions so
  the viewer's speed control remains effective.
- Preserve native control semantics, keyboard focus behavior, global keyboard
  navigation, and the `prefers-reduced-motion` override.
- Add the repository-standard NVIDIA SPDX header to new maintained files.

Do not edit or commit `dist/`, `.slidev/`, or `node_modules/`; they are generated
locally and ignored by Git.

## Validation gate

From the repository root:

```sh
cd docs/attribution-tour
corepack pnpm install --frozen-lockfile
corepack pnpm build
corepack pnpm dev
```

In the browser, inspect every changed scene and its incoming and outgoing
transitions. Check:

- Previous/Next, scene dots, and phase or sub-phase navigation.
- Application-log and system-log tabs.
- The `1×`, `½×`, and `¼×` motion settings.
- Arrow, Page Up/Page Down, Home, End, and Space navigation.
- Space activation when a button or link has keyboard focus.
- **Read the docs** links and copied `scene`/`motion` query parameters.
- Text clipping, connector overlap, and centering at the native 16:9 size.

If dependencies change, update and commit both `package.json` and
`pnpm-lock.yaml`; do not edit the lockfile manually. For hosted review and
publishing, follow
[Reviewing and publishing interactive documentation](../github-pages.md).
