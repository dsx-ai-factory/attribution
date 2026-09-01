# Contributing

Contributions open at GA.

During Early Access the rule schema, cascade-pattern syntax, and test framework are published so you can author and validate rules against your own Service instance. The rule catalog is curated by NVIDIA. The contribution workflow, reviewer register, and CLA publish at GA.

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
