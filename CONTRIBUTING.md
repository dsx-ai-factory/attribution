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
approve it before it can merge.

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
