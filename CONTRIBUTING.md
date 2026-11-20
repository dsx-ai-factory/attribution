# Contributing

Contributions to NVIDIA Attribution are welcome, including bug fixes,
documentation, rules, and connectors. Maintainers review proposed changes
for scope, correctness, tests, and compatibility with the project.

## Before you start

- Report reproducible upstream bugs through [GitHub Issues](https://github.com/dsx-ai-factory/attribution/issues/new?template=bug.yml).
  Follow the reproduction and evidence requirements in [SUPPORT.md](SUPPORT.md).
- Use [GitHub Discussions](https://github.com/dsx-ai-factory/attribution/discussions)
  for usage questions, feature requests, and design proposals. A maintainer opens
  a tracking issue when a proposal is accepted. Agree on the scope before starting
  implementation.
- Report security vulnerabilities through [SECURITY.md](SECURITY.md).
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
- Keep the issue and pull request free of confidential, proprietary, personal,
  security-sensitive, and other non-public information.

## Submit a pull request

Create a branch for an accepted issue, make the change, and open a pull
request that links the issue. Include the test commands and results in the
pull request description. A maintainer reviews the pull request and must
approve it before it can merge. To keep a branch current, refer to
[Keep your branch current](#keep-your-branch-current). To split a large change
into dependent pull requests, refer to
[Stack dependent pull requests](#stack-dependent-pull-requests).

## Keep your branch current

Rebase your branch onto `main` to bring in new changes. The
[merge and branch policy](RELEASE.md#merge-and-branch-policy) explains the
preference. Push the rebased branch with
`git push --force-with-lease --force-if-includes`, not `--force`. After a
`git fetch`, `--force-with-lease` alone accepts the fetched remote commits as
expected and overwrites them. Adding `--force-if-includes` (Git 2.30 or later)
makes the push fail when the remote branch has commits that you fetched but did
not integrate into your local branch. If you share a branch with another
contributor, tell them before you rebase.

## Stack dependent pull requests

Stacked pull requests are encouraged when a change is easier to review as a
sequence of small pull requests than as one large one, or when later work
depends on an earlier pull request that has not merged. Typical cases are a
refactor followed by the feature that uses it, or a format or API change
followed by its consumers. Keep each pull request in the stack reviewable on its
own, keep the stack short enough to review, and merge from the bottom up. Land
independent changes as separate pull requests, each branched from `main`,
not as a stack.

Create and maintain stacks with the `gh stack` extension for GitHub CLI. It
requires GitHub CLI 2.0 or later and Git 2.36 or later. These steps were checked
with `gh stack` v0.3.0. The extension is new and changing, so refer to its
documentation when a command behaves differently.

1. Install it with `gh extension install github/gh-stack`.
2. Create the stack with `gh stack init`, and add a layer with `gh stack add`.
   Commit your work to each layer before you add the next one.
3. Open the pull requests with `gh stack submit`.
4. Keep the stack current with `gh stack sync`. Run it when `main` has new
   commits and after a lower pull request merges. It fast-forwards `main`,
   rebases each layer onto its updated parent, and pushes the layers.
5. Merge with `gh stack merge`. The `gh pr merge` command does not merge a pull
   request that belongs to a stack.

If `gh stack sync` hits a rebase conflict, it restores every branch to its
original state and pushes nothing. Run `gh stack rebase`, resolve each conflict
and stage the files with `git add`, then run `gh stack rebase --continue` and
`gh stack push`. After you change a layer in the middle of a stack, run
`gh stack rebase --upstack` from that layer, then `gh stack push`.

For the full command reference, refer to the
[`gh stack` documentation](https://github.github.com/gh-stack/) and the
[GitHub stacked pull requests overview](https://gh.io/stacks), or run
`gh stack --help`.

Do not build a stack by hand. Branching from an unmerged feature branch, or
merging one feature branch into another, breaks after a squash merge, because
the dependent branch still carries the original commits. If you cannot use
`gh stack`, wait for the lower pull request to merge and branch from `main`.

## Documentation review

Apply these standards when writing or reviewing guides, reference pages,
examples, API documentation, and failure-catalog prose. Include reviewed
documentation with each newly public rule in the same release.

### Scope and wording

- Describe the documented behavior, observable symptom, or task, its impact
  where applicable, and the supported next step.
  Distinguish evidence from an inferred cause. A failure classification alone
  does not establish that hardware must be replaced or returned.
- Use active voice, present tense, and second person when addressing the
  reader. Avoid contractions, spell out uncommon abbreviations on first use,
  and prefer "refer to" over "see."
- State uncertainty when the evidence cannot distinguish application and
  infrastructure causes. Keep normal or informational observations distinct
  from failures; state when no corrective action is required.

### Actions and evidence

- State prerequisites, required permissions, and relevant software-version or
  deployment assumptions. A publicly documented command is not necessarily
  available to a tenant. Use numbered steps when their order matters.
- Give supported actions within the reader's access. Explain how to confirm
  recovery and when to stop troubleshooting or retrying. Do not invent an
  infrastructure recovery sequence for an application error.
- Direct application changes outside the tenant's control to the person or
  team responsible for the application. Direct restricted host, network, or
  driver investigation to the tenant's cluster administrator or cloud provider.
  Hardware replacement decisions belong to the provider and hardware vendor;
  do not prescribe vendor return procedures or make service commitments.
- Explain what evidence to provide: the NVIDIA Attribution diagnosis (verdict,
  symptom chain, and source logs), relevant identifiers, timestamps, and
  symptom-specific records when available. Identify evidence the provider
  must collect. Missing diagnostics or restricted logs must not delay asking
  for help.

### References and publication safety

- Confirm that a rule is approved for public release before publishing its
  details. Keep private rules and their documentation out of public branches.
- Cite publicly accessible documentation, specifications, or maintained public
  source repositories that support the actual instruction. Check links and
  anchors. For provider health or lifecycle events, verify that the event type
  and schema are documented in the provider's public reference; this check is
  not applicable to other entries.
- Review all fields, URLs, filenames, and examples for secrets, private
  infrastructure or paths, customer data, confidential product information,
  unpublished configuration, and private procedures or references. Use
  synthetic examples and preserve required third-party attribution.
- If public eligibility or technical support for an instruction is uncertain,
  ask a maintainer before publishing it, including on a public PR branch.
  Do not paste private evidence into a public discussion. Automated checks
  support human review; they do not grant publication approval.

### Failure-catalog entries

Keep tenant actions in `personas.tenant_sre.action` and handoff evidence in
`personas.tenant_sre.support_details`. Keep private notes and provider-only
procedures out of the public catalog, rendered pages, and exports. Use
publicly observable log strings or error codes in signal patterns, and explain
the conditions under which each pattern applies. Confirm that related entry
IDs and reference anchors exist in the public documentation. Review the
resulting UI guidance and exports when a change affects those surfaces.

## Signing Your Work

Sign each commit with the Developer Certificate of Origin (DCO):

```bash
git commit -s
```

The `-s` option adds the required `Signed-off-by` trailer. By signing off, you
certify the [Developer Certificate of Origin](https://developercertificate.org/).

### Full text of the DCO

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Run the test gate

Run the commands below in Bash from the repository root. Before running the
full gate, install:

- [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage Python
  runtimes and locked project dependencies. CI uses uv **0.12.3**; use that
  version to match CI.
- Node.js 22 and [pnpm 10.11.0](https://pnpm.io/10.x/installation), matching the
  frontend CI setup.
- [Docker](https://docs.docker.com/get-started/get-docker/), with the daemon
  running and accessible to your user. The end-to-end suite builds images and
  starts containers; installing the Docker CLI alone is not enough.
- Git and `curl`, used by the test scripts.

With Node.js 22 installed, enable Corepack and select the same pnpm version as CI:

```bash
corepack enable
corepack prepare pnpm@10.11.0 --activate
```

If your Node.js installation does not include Corepack, follow the
[pnpm installation instructions](https://pnpm.io/10.x/installation) to install
Corepack or pnpm 10.11.0 directly.

The full gate also needs Python **3.14.6** on the host, even when Docker is
available. This is the Service's pinned runtime in
[`service/service-app/.python-version`](service/service-app/.python-version)
and the version used for all Python suites in CI. The Library and Blueprint
declare Python 3.11+ support, but that minimum is not sufficient for the Service.

Install the pinned runtime and select it for this shell, as CI does:

```bash
export UV_PYTHON="$(cat service/service-app/.python-version)"
uv python install "$UV_PYTHON"
uv run --no-project --python "$UV_PYTHON" python --version
```

`UV_PYTHON` is set only for the current shell. Repeat the setup above when
opening a new terminal to run the tests.

If installation fails, update uv and check that the pinned runtime is available
for your platform before continuing. Do not change the project's Python pin to
work around a local setup failure.

Check that Docker is reachable and the frontend tools have the required versions:

```bash
docker info
node --version
pnpm --version
```

Before running the full gate, stage the changes you want to test with
`git add <files>`. The `service-app-client-drift` suite uses the Git index
(staged source), and `end-to-end` builds its Docker images from the Git index.
Unstaged edits are not included in those snapshots. The end-to-end suite also
reads host-side tests and configurations from the working tree, so keep those
changes staged too. The same requirement applies when running either suite
independently.

Once the setup checks succeed and your changes are staged, run the full gate:

```bash
bash .ci/run-tests.sh all
```

The command runs the Docker end-to-end tests, the Library, Service, and Blueprint
Python suites, the generated-client and license checks, the frontend tests and
build, and the CI-script tests. The first run needs network access to download
dependencies and container images.

### Run a focused suite

Run one suite independently while working on a change, for example:

```bash
bash .ci/run-tests.sh attribution-lib
bash .ci/run-tests.sh service-app
bash .ci/run-tests.sh frontend
```

| Suite | Requirements |
| --- | --- |
| `attribution-lib`, `blueprint` | uv and Python 3.11+; use the pinned runtime above to match CI |
| `service-app`, `service-app-client-drift` | uv and the pinned Service Python runtime |
| `license-compliance`, `ci-tests` | uv and Python; use the pinned runtime above to match CI |
| `frontend` | Node.js 22 and pnpm 10.11.0; no host Python or Docker required |
| `end-to-end` | uv, host Python, Git, `curl`, and a running Docker daemon; use the pinned runtime above |

Python-only suites do not require Docker or frontend tools. A focused suite
does not replace the full gate before review; state which checks you ran and
which you could not run in the pull request.

## Install the pre-commit hook

```bash
uv tool install pre-commit
pre-commit install
```

The license hook checks staged files for license compliance. When OpenAPI client
inputs change, the client hook regenerates the client and stops the commit if
the generated files need to be staged. Stage those files and retry the commit.
To regenerate the client manually, run
`service/service-app/openapi/generate-client.sh`.

## Title commits and pull requests

Pull requests are squash-merged, so the PR title becomes the commit subject on `main`. Write it as a [Conventional Commit](https://www.conventionalcommits.org/), `type(scope)!: summary`. CI checks the format and type; the rest is guidance.

- Types: `feat`, `fix`, `build`, `chore`, `ci`, `docs`, `perf`, `refactor`, `revert`, `style`, `test`.
- Scope is optional. Common scopes are `attribution-lib`, `service-app`, `blueprint`, `frontend`, `docs`, and `ci`.
- Write the summary in the imperative mood with no trailing period.
- Mark a removed or incompatible public API with `!` and add a `BREAKING CHANGE:` footer to the PR body that says what callers must change.
- The type signals the version bump for the next release of each touched component: `!` is major, `feat` is minor, `fix` is patch, and other types do not bump. A component still at `0.y.z` shifts each bump down one place: `!` is minor, and `feat` and `fix` are patch.

## Preview the published documentation

The pages listed in `docs/index.yml` are published as a [Fern](https://buildwithfern.com) site. The interactive walkthrough in `docs/attribution-tour/` and `docs/github-pages.md` are not part of it.

Install the Fern CLI with `npm install -g fern-api@latest`. The CLI is intentionally not pinned. Then run these from the repository root, the same checks that `Fern Check` runs on pull requests:

```bash
python3 tools/check-docs-mdx --exclude 'docs/attribution-tour/*' --exclude 'docs/github-pages.md' docs fern
python3 tools/check-docs-paths --exclude 'docs/attribution-tour/*' --exclude 'docs/github-pages.md' --allow '../docs/index.yml' docs fern
fern check
fern docs md check
fern docs broken-links
```

To preview the site locally, run `fern docs dev`. It needs network access to fetch the shared theme and serves the pages under `/attribution`, for example `/attribution/get-started/overview`.

Keep these rules in mind when editing a published page:

- Fern compiles pages as MDX, so a bare `{`, a `<digit`, an HTML comment, or an unclosed `<img>` in prose fails the build. `tools/check-docs-mdx` reports these.
- A link to a file outside `docs/` must be an absolute GitHub URL. A relative link works on GitHub but leads to a missing page on the published site, and `fern docs broken-links` does not report it.
- Keep every file path inside `docs/`. Fern reads any file a page names, such as the `src` of a `Code` or `Markdown` component, a markdown image, or a `path` in `docs/index.yml`, and it accepts `../` and absolute paths. The publish job runs with a repository secret in its environment, so a page that reaches outside `docs/` could expose it. `tools/check-docs-paths` rejects these paths in `docs/` and `fern/`, along with symbolic links. The one path it allows to leave its directory is the `../docs/index.yml` entry in `fern/docs.yml`. It also rejects YAML it cannot read as plain `key: value` lines, such as quoted keys, block scalars, and multi-line values, so write navigation entries in that plain form. `Fern Check` and the publish job run it.
- A new page needs an entry in `docs/index.yml`.

A pull request from a branch in this repository that changes the published pages gets a preview link in a comment. Pull requests from forks do not get a preview, because the preview build runs with a repository secret. Merging a change to the published pages, to `fern/`, or to the publish workflow itself into `main` publishes the site. A merge that touches none of these does not.

## Review interactive documentation

The attribution walkthrough is a Slidev application published through GitHub
Pages. Start with the [editing tutorial](docs/attribution-tour/TUTORIAL.md); the
[maintenance reference](docs/attribution-tour/MAINTAINING.md) and
[publishing guide](docs/github-pages.md) cover ongoing work and deployment.
