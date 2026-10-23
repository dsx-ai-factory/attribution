<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Editing the walkthrough: a beginner's tutorial

This tutorial gets the walkthrough running, explains enough of its structure to
make a change confidently, and shows how to check the result. You do not need
previous Slidev or Vue experience.

## 1. Install the tools

The walkthrough uses [Slidev](https://sli.dev/guide/), a tool for building web
presentations. Slidev runs on Node.js, and this repository's workflows use
Node.js 22.

Install Node.js 22 from the [official download page](https://nodejs.org/en/download)
or with your usual Node version manager. Then open a terminal and check it:

```sh
node --version
```

The result should begin with `v22`.

The next tool, Corepack, selects the right package-manager version for a
project. Check whether it came with your Node installation:

```sh
corepack --version
```

If the shell says `corepack` is not found, install it through the `npm` command
that comes with Node.js:

```sh
npm install --global corepack
```

The [Corepack installation guide](https://github.com/nodejs/corepack#how-to-install)
has more troubleshooting detail. You do not need to install pnpm separately.
Whenever this guide says `corepack pnpm`, Corepack reads the `packageManager`
entry in `package.json` and runs the project's pinned pnpm version.

## 2. Install the project and start it

In a terminal, move to the root of your local repository checkout. Then run:

```sh
cd docs/attribution-tour
corepack pnpm install --frozen-lockfile
corepack pnpm dev
```

The first command downloads the packages needed by the walkthrough. The second
starts a local web server. Open the URL printed in the terminal, typically
`http://localhost:3030/`.

Leave that terminal running while you edit. Slidev refreshes the browser after
you save a file. When you are done, return to the terminal and press `Ctrl+C`.

If the usual port is already occupied, Slidev prints a different URL; use that
one. If it cannot compile a change, the terminal normally shows the affected
file and line number.

## 3. Learn the mental model

Most Slidev decks put each frame in `slides.md`. This walkthrough is different:
`slides.md` contains one page and mounts the custom
`AttributionStoryboard.vue` component. The component owns all the frames shown
by the dots along the bottom.

The component has two main sections:

- `<script setup>` holds the example data, list of scenes, current selection,
  and functions such as Next and Previous.
- `<template>` describes the cards, labels, arrows, and controls that appear in
  the browser.

The separate `styles/index.css` file decides where those elements sit and how
they move. When the current scene changes, Vue adds or removes classes such as
`visible`, `docked`, and `materializing`; CSS transitions animate the existing
elements into place.

Here is enough Vue syntax to read the component:

| Form | Plain-language meaning |
| --- | --- |
| `ref(...)` | A value that can change, such as the current scene. |
| `computed(...)` | A value calculated from other state. |
| `v-if="..."` | Show this element only when the condition is true. |
| `v-for="item in items"` | Repeat this element for every item in a list. |
| `:class="..."` | Add CSS classes based on the current state. |
| `@click="..."` | Run a function when the element is clicked. |
| `{{ value }}` | Put a value into visible text. |

If you need more background, see Vue's introductions to
[template syntax](https://vuejs.org/guide/essentials/template-syntax.html) and
[class and style bindings](https://vuejs.org/guide/essentials/class-and-style.html),
or Slidev's guide to
[components in slides](https://sli.dev/guide/component).

## 4. Make a copy or data change

Start with the development server running and navigate to the scene you want to
change.

1. Open `components/AttributionStoryboard.vue` in your editor.
2. Use **Find in Files** to search for a phrase or value visible in the browser.
3. Edit the matching text and save the file.
4. Return to the browser and confirm that it refreshed.
5. Move backward and forward one scene to make sure the edit still fits in
   neighboring states.

Scene headings and summaries live in the `scenes` array near the top of the
component. Example log lines live in `observations`, and the extracted events
live in `events`. A longer label may require a small width or position change in
`styles/index.css`.

Before changing what the example claims about attribution, check
`docs/attribution.md` and the relevant implementation or bundled rule. The
visual should describe existing behavior accurately.

## 5. Add a scene

The `scenes` array is the table of contents for the storyboard. Its order
controls Next/Previous and the dots along the bottom. Each entry looks like:

```ts
{
  id: 'filter-example',
  phase: 'filtering',
  kicker: 'Filtering · EXAMPLE',
  title: 'One clear idea for this frame',
  summary: 'A short explanation of what changed and why.',
  short: 'Example',
  docAnchor: '#filter',
}
```

Copy a nearby entry and adjust it:

- `id` is a short, lowercase, hyphenated name. It appears in shareable URLs,
  so preserve existing IDs whenever possible.
- `phase` connects the scene to the colored phase rail.
- `kicker`, `title`, and `summary` are the visible explanation.
- `short` labels the bottom navigation.
- `docAnchor` must match a heading anchor in `docs/attribution.md`.

If the scene belongs to Filtering or Refinement, also add its ID to the matching
list in `subphaseGroups`. That creates its upper navigation dot.

Next, decide what should appear in the main figure. Reuse an existing template
stage when possible. Conditions such as `sceneIs('filter-example')` show content
for one scene, while dynamic classes trigger positions defined in CSS. For a
completely new kind of figure, add a clearly named `<section>` in the template
and a matching CSS block.

When an observation, event, or symptom continues into the next scene, keep the
same template element and move it with CSS. A duplicate element will look as if
the evidence disappeared and a replacement appeared.

Use the CSS variables `--motion-fast`, `--motion-medium`, and `--motion-slow`
for transition durations. This keeps the `1×`, `½×`, and `¼×` viewer
controls working.

## 6. Check the visual story

For every changed scene, also review the scenes immediately before and after
it. Look for:

- A centered main figure and comfortably readable text.
- Cards, labels, arrows, and controls that do not overlap.
- Clear phase and sub-phase navigation.
- Visible per-node versus global boundaries where relevant.
- An explicit indication when a rule phase does nothing.
- Smooth continuity for observations, events, and symptoms that persist.

Exercise Previous/Next, scene dots, source tabs, documentation links, arrow
keys, Space, and all three motion settings. Tab to buttons and links and confirm
that Space activates the focused control instead of unexpectedly advancing the
story.

## 7. Build before committing

Stop the development server with `Ctrl+C`, then run from
`docs/attribution-tour/`:

```sh
corepack pnpm build
```

A successful build ends with a completion message and writes generated files
to `dist/`. Do not commit `dist/`, `.slidev/`, or `node_modules/`; Git already
ignores them.

If you intentionally change a dependency, update and commit both `package.json`
and `pnpm-lock.yaml`. Do not edit the lockfile by hand.

For a compact reference during later changes, use [MAINTAINING.md](MAINTAINING.md).
For hosted pull-request previews and production publishing, see
[Reviewing and publishing interactive documentation](../github-pages.md).
