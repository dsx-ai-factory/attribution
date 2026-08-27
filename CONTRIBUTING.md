# Contributing

Contributions open at GA.

During Early Access the rule schema, cascade-pattern syntax, and test framework are published so you can author and validate rules against your own Service instance. The rule catalog is curated by NVIDIA. The contribution workflow, reviewer register, and CLA publish at GA.

## Run the test gate

Install `uv`, Node.js 22, and pnpm 10.11, then run:

```bash
bash .ci/run-tests.sh all
```

The command runs the 118-workload Docker end-to-end tests, the Attribution Library, Attribution Service, and Blueprint Python suites, then tests and builds the Blueprint frontend. Pass a suite name shown by `.ci/run-tests.sh` to run one part of the gate.
