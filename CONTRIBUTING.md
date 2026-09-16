# Contributing

Early Access contributions are welcome. Start with a GitHub issue before you
begin implementation. The issue records the problem, the proposed change, and
the discussion that reviewers use during review.

## Before you start

- Open a [bug report](https://github.com/dsx-ai-factory/attribution/issues/new?template=bug.yml) or a
  [feature request](https://github.com/dsx-ai-factory/attribution/issues/new?template=feature.yml). Do not include
  security vulnerabilities in a public issue; follow the
  [security policy](SECURITY.md).
- Follow the [Code of Conduct](https://github.com/dsx-ai-factory/attribution/blob/c6be17b9a01143eeaa765f961d1ef9fc539f8dcd/CODE_OF_CONDUCT.md).
- Keep the issue and pull request free of confidential, proprietary, personal,
  security-sensitive, and other non-public information.

## Submit a pull request

Create a branch for an accepted issue, make the change, and open a pull
request that links the issue. Include the test commands and results in the
pull request description. A maintainer reviews the pull request and must
approve it before it can merge.

Sign each commit with the Developer Certificate of Origin (DCO):

```bash
git commit -s
```

The `-s` option adds the required `Signed-off-by` trailer. By signing off, you
certify the [Developer Certificate of Origin](https://developercertificate.org/).

## Run the test gate

Install `uv`, Node.js 22, and pnpm 10.11, then run:

```bash
bash .ci/run-tests.sh all
```

The command runs the 118-workload Docker end-to-end tests, the Attribution Library, Attribution Service, and Blueprint Python suites, then tests and builds the Blueprint frontend. Pass a suite name shown by `.ci/run-tests.sh` to run one part of the gate.

## Install the pre-commit hook

```bash
uv tool install pre-commit
pre-commit install
```

The hook regenerates the service-app's OpenAPI client when a commit touches something the client is built from, blocking the commit until the regenerated files are staged. Skip with `git commit --no-verify`.

## Review interactive documentation

The attribution walkthrough is a Slidev application published through GitHub
Pages. Start with the [editing tutorial](docs/attribution-tour/TUTORIAL.md); the
[maintenance reference](docs/attribution-tour/MAINTAINING.md) and
[publishing guide](docs/github-pages.md) cover ongoing work and deployment.
