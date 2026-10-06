# Release policy

## Versioning

NVIDIA Attribution components have individual versions that follow [Semantic
Versioning 2.0.0](https://semver.org/). Component release tags include the
component name: `service-app-vMAJOR.MINOR.PATCH` for the Attribution Service,
`attribution-lib-vMAJOR.MINOR.PATCH` for the Attribution Library, and
`blueprint-vMAJOR.MINOR.PATCH` for the Blueprint.

Components release independently. A release of one component does not require
a new version or tag for another component.

The planned version source is the literal `__version__` in each component's
`__init__.py`:

| Component | Version source |
| --- | --- |
| Attribution Service | `service/service-app/app/__init__.py` |
| Attribution Library | `service/attribution-lib/src/attribution_lib/__init__.py` |
| Blueprint | `blueprint/src/attribution_blueprint/__init__.py` |

The Service and Blueprint already define version literals. The Library currently
derives `__version__` from Git; its migration to a literal is follow-up work.
The planned process does not use `pyproject.toml` or `bundle-manifest.json` as
the release-version source. Removal of the manifest is also follow-up work.
Until that migration is complete, existing package metadata and tests remain
applicable.

Before version 1.0, a minor release can contain breaking changes. After version
1.0:

- a major release can contain incompatible API or configuration changes;
- a minor release adds backward-compatible functionality; and
- a patch release contains backward-compatible fixes.

Release notes identify known breaking changes and required migration steps.

## Release artifacts

For the initial process, a component tag on a reviewed default-branch commit is
the release. The process does not publish packages, prebuilt container images,
or Helm charts.

Prebuilt container images, including an NGC image, and a public Helm chart are
planned future artifacts. No publication date is committed for those artifacts.
Release notes must list the artifacts actually available for each release.

This policy does not change the current Early Access license or authorize
publication. Publication requires the applicable NVIDIA authorization.

## Merge and branch policy

Pull requests use squash merges into the default branch. Rebase feature branches
onto the default branch to keep them current; do not merge the default branch
into a feature branch.

## Planned release process

CI tagging is not implemented yet. Maintainers must implement and validate it,
and complete the version-source migration, before the first release under this
process.

For each component release:

1. Change the component's version literal in its `__init__.py` in a pull
   request. Use that component's version and tag prefix.
2. Run the complete test gate with `bash .ci/run-tests.sh all`.
3. Prepare release notes that summarize user-visible changes, compatibility or
   migration requirements, known limitations, and any available artifacts.
4. Have the release changes reviewed and squash-merged to the default branch.
5. CI detects the changed version literal and creates the corresponding
   component tag on that merged commit. Components whose version literals did
   not change receive no new tag.

Project maintainers review version changes and control the CI release workflow.
Contributors propose version changes through pull requests. Release tags are
created by CI; no contributor or maintainer pushes release tags by hand.
