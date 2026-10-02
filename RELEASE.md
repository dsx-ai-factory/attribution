# Release policy

## Versioning

NVIDIA Attribution components have individual versions that follow [Semantic
Versioning 2.0.0](https://semver.org/). Component release tags include the
component name: `service-app-vMAJOR.MINOR.PATCH` for the Attribution Service and
`blueprint-vMAJOR.MINOR.PATCH` for the Blueprint.

The Blueprint version is also the top-level release-bundle version. There is no
separate bundle version. `bundle-manifest.json` records the included components;
their versions do not need to be equal to each other. Release notes identify the
included component versions.

Before version 1.0, a minor release can contain breaking changes. After version
1.0:

- a major release can contain incompatible API or configuration changes;
- a minor release adds backward-compatible functionality; and
- a patch release contains backward-compatible fixes.

Release notes identify known breaking changes and required migration steps.

## Release artifacts

The initial release artifact is tagged source for the release bundle. The source
contains the Attribution Service, Attribution Library, Blueprint reference
orchestrator, Blueprint web UI, and `blueprint/docker/Dockerfile` to build the
bundle image. The bundle is tested with the included component versions.

Prebuilt container images, including an NGC image, and a public Helm chart are
planned future artifacts. No publication date is committed for those artifacts.
Release notes must list the artifacts actually available for each release.

This policy does not change the current Early Access license or authorize
publication. Publication requires the applicable NVIDIA authorization.

## Merge and branch policy

Pull requests use squash merges into the default branch. Rebase feature branches
onto the default branch to keep them current; do not merge the default branch
into a feature branch.

## Release process

For each release:

1. Select the Blueprint version and the versions of affected components. Set
   `project.version` in `blueprint/pyproject.toml` and `version` in
   `bundle-manifest.json` to the Blueprint version. Update the affected
   component metadata. Keep `project.version` in
   `service/attribution-lib/pyproject.toml` equal to `attribution_lib_version`
   in `bundle-manifest.json`; this library-version requirement does not require
   all component versions to match.
2. Run the complete test gate with `bash .ci/run-tests.sh all`.
3. Prepare release notes that summarize user-visible changes, compatibility or
   migration requirements, known limitations, and the supported artifacts.
4. Have the release changes reviewed and squash-merged to the default branch.
5. Create the Blueprint and affected component release tags from the reviewed
   default-branch commit, using each component's version and tag prefix.
6. Publish the authorized source release with its release notes. List any other
   artifacts only when they are available and authorized for publication.

Only project maintainers create or publish release tags. Contributors must not
create release tags for the project.
