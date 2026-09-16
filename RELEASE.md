# Release policy

## Versioning

NVIDIA Attribution release versions follow [Semantic Versioning
2.0.0](https://semver.org/) and use tags in the form `vMAJOR.MINOR.PATCH`.

Before version 1.0, a minor release can contain breaking changes. After version
1.0:

- a major release can contain incompatible API or configuration changes;
- a minor release adds backward-compatible functionality; and
- a patch release contains backward-compatible fixes.

Release notes identify known breaking changes and required migration steps.

## Supported release artifact

The supported release artifact is the versioned release bundle. The bundle
contains the Attribution Service, the Attribution Library, the Blueprint
reference orchestrator, and the Blueprint web UI. These components are tested
and released as one unit.

Separately versioned packages and prebuilt container images are not release
artifacts unless the release notes explicitly list them.

## Release process

For each release:

1. Select the semantic version and update the bundle and component version
   metadata.
2. Run the complete test gate with `bash .ci/run-tests.sh all`.
3. Prepare release notes that summarize user-visible changes, compatibility or
   migration requirements, known limitations, and the supported artifacts.
4. Have the release changes reviewed and merged to the default branch.
5. Create the version tag from the reviewed default-branch commit.
6. Publish the GitHub release for that tag with the release notes and release
   bundle.

Only project maintainers create or publish release tags. Contributors must not
create release tags for the project.
