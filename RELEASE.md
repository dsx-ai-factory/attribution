# Release policy

## Versioning

NVIDIA Attribution components have individual versions that follow [Semantic
Versioning 2.0.0](https://semver.org/). Component release tags include the
component name: `service-app-vMAJOR.MINOR.PATCH` for the Attribution Service,
`attribution-lib-vMAJOR.MINOR.PATCH` for the Attribution Library, and
`blueprint-vMAJOR.MINOR.PATCH` for the Blueprint. A release candidate is tagged
`<component>-vX.Y.Z-rc.N`, and its literal uses the PEP 440 spelling
`X.Y.ZrcN`.

Components release independently. A release of one component does not require
a new version or tag for another component.

The release-version source is the literal `__version__` in each component's
`__init__.py`:

| Component | Version source |
| --- | --- |
| Attribution Service | `service/service-app/app/__init__.py` |
| Attribution Library | `service/attribution-lib/src/attribution_lib/__init__.py` |
| Blueprint | `blueprint/src/attribution_blueprint/__init__.py` |

The Service, Library, and Blueprint each define this literal. Builds report the
literal plus the number of commits since the component's last tag, for example
`2.0.0+7`, or `2.0.0+7.dirty` with uncommitted changes. This does not change
the committed release version.

Compatibility commitments and supported interfaces are defined in
[SUPPORT.md](SUPPORT.md#support-lifecycle). Apply those commitments when choosing
the version for a change. The Library and Blueprint interfaces have the
limitations described in [Interface compatibility](SUPPORT.md#interface-compatibility).

Each component's `CHANGELOG.md` identifies known breaking changes and required
migration steps.

## Release artifacts

For the initial process, a component tag on a reviewed default-branch commit is
the release. The process does not publish packages, prebuilt container images,
or Helm charts.

Prebuilt container images, including an NGC image, and a public Helm chart are
planned future artifacts. No publication date is committed for those artifacts.

Maintainers publish releases after the required reviews and checks are complete.

## Release bundles

A Blueprint release is the stamp for a known-good set: it is tested against the
exact Service and Library releases at that commit, and that combination is
supported together. The same holds one level down: a Service release is the
supported pairing with the exact Library release at that commit.

Releasing a component requires each in-repository dependency to be at distance
0 at that commit. A dependency that has moved since its tag gets its own
release in the same pull request. Version numbers stay independent; nothing is
renumbered to match. A stable release only bundles stable dependencies; a
release candidate may bundle stable or release candidate ones.

## Merge and branch policy

Pull requests use squash merges into the default branch. Rebase feature branches
onto the default branch to keep them current. Do not merge the default branch
into a feature branch.

## Release process

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
Contributors propose version changes through pull requests. CI creates release
tags. Contributors and maintainers do not push release tags by hand.

### CHANGELOG

Each component keeps its own `CHANGELOG.md` in its directory, in this format:

```markdown
## [<version>] - <date>

### BREAKING CHANGES
- <PR title> (#<PR number>)

### Features
- <PR title> (#<PR number>)

### Bug Fixes
- <PR title> (#<PR number>)
```

Omit a section that has no entries.

## Release checks

A pull request is a release pull request when ANY component's version literal
differs from the default branch. A component is releasing when its literal
changed and did not decrease. The Service depends on the Library, and the
Blueprint depends on both.

On every push to a branch other than the default branch, `.ci/release.py status`
reports each component's literal, latest tags, distance since them, and pull
requests merged since, and `.ci/release.py check` validates each releasing
component, reports every rule outcome, and fails on any violation:

- The literal is a PEP 440 final or release candidate: `X.Y.Z` or `X.Y.ZrcN`.
- The tag for that literal does not exist yet.
- The component's `CHANGELOG.md` has a `## [<literal>]` heading.
- Each dependency not releasing in the same pull request has a tag, stable when
  the component release is stable.
- Each such dependency is unchanged since that tag: no first-parent commits
  touch it or its own dependencies.
- A stable release cannot include a dependency whose latest tag is a release
  candidate, or one released as a candidate in the same pull request.

After a merge to the default branch that changes a version literal, the
`Release` workflow re-runs the same rules on the merged commit and, when they
pass, creates and pushes one `<component>-vX.Y.Z` tag per releasing component.
A rule violation tags nothing for that component; fix it in a new pull request.
Re-run the job only for a transient failure such as a rejected push.
