# Contributing

Contributions open at GA.

During Early Access the rule schema, cascade-pattern syntax, and test framework are published so you can author and validate rules against your own Service instance. The rule catalog is curated by NVIDIA. The contribution workflow, reviewer register, and CLA publish at GA.

## Run the test gate

Install `uv`, Node.js 22, and pnpm 10.11, then run:

```bash
make test
```

The root Makefile is the stable public entry point for repository-wide test
commands. It delegates to `.ci/run-tests.sh`, which keeps local and CI suite
selection consistent without replacing the native tools used by each
component. Run `make help` to list the supported targets, `make test-unit` for
the unit and integration suites, or `make test-e2e` for the Docker end-to-end
suite.

`make test` runs the 118-workload Docker end-to-end tests, the Attribution
Library, Attribution Service, and Blueprint Python suites, then tests and
builds the Blueprint frontend. To run one suite, pass a suite name shown by
`.ci/run-tests.sh` directly to that script.

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
