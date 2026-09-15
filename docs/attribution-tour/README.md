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

# Interactive attribution walkthrough

This standalone [Slidev](https://sli.dev/) deck is an interactive companion to
the repository-level [How Attribution Works](../attribution.md) documentation.
It follows one compact example across the shared NVIDIA Attribution process,
verified against the bundled rules.

The deck is a self-contained static application so a broader documentation
site can mount it without coupling its runtime or navigation. Set
`VITE_ATTRIBUTION_DOCS_BASE` at build time to point its phase links at a
rendered documentation route; by default, they point to the canonical Markdown
file on GitHub.

## Preview locally

From this directory:

```sh
corepack pnpm install
corepack pnpm dev
```

Use the on-screen controls or the arrow keys to page through the storyboard.
Each frame links directly to the corresponding section of
[`docs/attribution.md`](../attribution.md), and every frame has a shareable URL:

```text
http://localhost:3030/?scene=filter-standard
```

Other useful scene IDs include `initialize`, `ingest`, `extraction`,
`refine-scrub`, `outcome`, `post-outcome`, `collection`, and `result`. The old
`?scene=observations` URL remains an alias for `ingest`.

The Motion control defaults to half speed. Its `1×`, `½×`, and `¼×` settings
can also be shared in the URL as `motion=normal`, `motion=slow`, or
`motion=study`:

```text
http://localhost:3030/?scene=filter-late&motion=study
```

To verify the production bundle:

```sh
corepack pnpm build
```

## Maintain the walkthrough

New to Node, Slidev, or Vue? Start with the
[beginner's editing tutorial](TUTORIAL.md). The
[maintenance reference](MAINTAINING.md) summarizes the architecture,
conventions, and review gate. Coding agents working in this directory should
also follow the scoped [agent instructions](AGENTS.md).

## Publish and review on GitHub Pages

See [Reviewing and publishing interactive documentation](../github-pages.md)
for the PR-preview workflow, production URL, and initial GitHub Pages setup.
