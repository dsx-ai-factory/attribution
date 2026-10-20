<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Reviewing and publishing interactive documentation

The [attribution walkthrough](attribution-tour/README.md) is a Slidev
application that builds to static files and is hosted with the repository's
GitHub Pages site.

## Review locally

Run the development server from the repository root:

```sh
cd docs/attribution-tour
corepack pnpm install
corepack pnpm dev
```

The terminal prints the local URL. Use `corepack pnpm build` in the same
directory to verify a production build.

The build writes `assets/THIRD_PARTY_NOTICES.txt`. It contains
[the repository notices](../THIRD_PARTY_NOTICES.md) and the license and NOTICE
texts of every npm package whose code or icon data ships in the bundle. The
build fails if a bundled package has no license text. For a package that does
not ship one, add its upstream license to
`attribution-tour/third-party-licenses/` and map the exact package version to
it in `attribution-tour/vite.config.ts`. The published site serves this file
with the deck, and the Pages landing page links to it. The Blueprint WebUI
build uses the same plugin.

## Review a pull request

For a same-repository pull request, the `Attribution Process Walkthrough PR
Preview` workflow builds the proposed code with read-only permissions, passes
the static artifact to a separate publishing job, and posts a temporary preview
link on the pull request. The preview is refreshed after each push and removed
when the pull request closes. Fork pull requests do not receive hosted previews.

Use the temporary preview URL when reviewing or describing an open pull
request. Do not put it in permanent repository documentation because its
`pr-preview/pr-<number>/` path is deleted when the pull request closes.

## Publish from `main`

After a change merges, the `Attribution Process Walkthrough Pages` workflow
publishes the deck to the `gh-pages` branch at:

<https://dsx-ai-factory.github.io/attribution/walkthroughs/attribution-process/>

Use this stable URL in repository documentation. Production and PR-preview
deployments share the `gh-pages` branch, and the workflows serialize their
updates so they preserve one another's files. The Pages root is a small landing
page reserved for a future documentation site.

## Configure Pages for the first deployment

The first workflow run creates the `gh-pages` branch. A repository administrator
must then open **Settings → Pages**, choose **Deploy from a branch**, and select
`gh-pages` and `/(root)`. Keep the site private unless wider access has been
approved, then rerun the initial workflow. Until this setup and deployment
finish, the permanent URL may return 404.
