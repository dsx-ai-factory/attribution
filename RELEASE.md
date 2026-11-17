# Release policy

## Versioning

NVIDIA Attribution components have individual versions that follow [Semantic
Versioning 2.0.0](https://semver.org/). Component release tags include the
component name: `service-app-vMAJOR.MINOR.PATCH` for the Attribution Service,
`attribution-lib-vMAJOR.MINOR.PATCH` for the Attribution Library, and
`blueprint-vMAJOR.MINOR.PATCH` for the Blueprint. A release candidate is tagged
`<component>-vX.Y.Z-rc.N`, and its literal uses the PEP 440 spelling
`X.Y.ZrcN`.

Version numbers are independent: nothing is renumbered to match another
component. Releases are not: a component whose code, or the code of an
in-repository dependency, changed since its latest tag releases in the same
pull request as the dependency (see [Release bundles](#release-bundles)). A
component that did not change gets no new version and no tag.

The release-version source is the literal `__version__` in each component's
`__init__.py`:

| Component | Version source |
| --- | --- |
| Attribution Service | `service/service-app/app/__init__.py` |
| Attribution Library | `service/attribution-lib/src/attribution_lib/__init__.py` |
| Blueprint | `blueprint/src/attribution_blueprint/__init__.py` |

The Service, Library, and Blueprint each define this literal. A build reports
the version of the latest release tag that applies to the commit it builds:

| Build of | Reports |
| --- | --- |
| The tagged commit, or a later one with no commit since that touches the component or its dependencies | `2.0.0` |
| 7 such commits since the tag, at short commit SHA `1a2b3c4` | `2.0.0+7.g1a2b3c4` |
| Either of the above, with uncommitted changes | `2.0.0+dirty`, `2.0.0+7.g1a2b3c4.dirty` |
| `main`, using the latest tag of a release branch cut from `main`'s history | always a suffix, for example `2.0.0+7.g1a2b3c4` |
| `main` in a checkout with no `main` ref | the literal, with `+dirty` for uncommitted changes |
| No usable tag or git history, such as a container image build | the literal, with `+dirty` for uncommitted changes |

The suffix does not change the committed release version.

Compatibility commitments and supported interfaces are defined in
[SUPPORT.md](SUPPORT.md#support-lifecycle). Apply those commitments when choosing
the version for a change. The Library and Blueprint interfaces have the
limitations described in [Interface compatibility](SUPPORT.md#interface-compatibility).

Each component's `CHANGELOG.md` identifies known breaking changes and required
migration steps.

## Release artifacts

For the initial process, a component tag on a reviewed merge commit on a
release branch is the release. Each tag gets a GitHub Release page carrying
that component's `CHANGELOG.md` section. The process does not currently publish packages,
prebuilt container images, or Helm charts.

Each component has its own release line, so "the current release" is a
per-component question. The badges at the top of `README.md` show the newest
stable release of each component. GitHub itself marks only one release per
repository as "Latest"; here that is the newest stable Blueprint release,
because a Blueprint release stamps a supported Library and Service set (see
[Release bundles](#release-bundles)). Library and Service releases are never marked "latest", even when they are newer than the Blueprint release. "Latest" therefore means "latest
supported set", not necessarily "most recent tag".

Prebuilt container images, including an NGC image, and a public Helm chart are
planned future artifacts. No publication date is committed for those artifacts.

Maintainers publish releases after the required reviews and checks are complete.

## Release bundles

A Blueprint release is the stamp for a known-good set: it is tested against the
exact Service and Library releases at that commit, and that combination is
supported together. The same holds one level down: a Service release is the
supported pairing with the exact Library release at that commit.

Releasing a component requires each in-repository dependency to be at distance
0 at that commit. Distance is the number of commits since the tag that change the
component or its dependencies, not counting a commit the tag already includes as
a cherry-pick, and it is 0 when the files match the tag, ignoring the version
literal line, each `CHANGELOG.md` and the generated client files that a release
pull request rewrites. A dependency that has moved since its tag gets its own
release in the same pull request. The cascade runs upward:
when the Library changes, the Service and the Blueprint release with it, and
when the Service changes, the Blueprint releases with it. A component whose
closure is unchanged gets no tag. A stable release only bundles stable
dependencies; a release candidate may bundle stable or release candidate ones,
so a Library release candidate makes its changed dependants release candidates
too.

## Choosing a release

Each release lives on one release branch, and the highest tags of different
components can sit on different branches. To choose a release, take the latest
release tag of the top-most component you use (Blueprint, then Service, then
Library), then use the release branch that tag is on. A Blueprint user takes the
latest `blueprint-v*` tag, not the latest `service-app-v*` tag. To find the
branch of a tag:

```bash
git fetch --tags origin
git branch --remotes --contains <tag>
```

## Merge and branch policy

Pull requests use squash merges into the default branch and into release
branches. Rebase feature branches onto the default branch to keep them current.
Do not merge the default branch into a feature branch.

Releases come from release branches named `release/YYYY.N`. The default branch
never freezes. The releaser cuts a release branch from a default-branch commit
when a release starts, and from then on the branch takes only fixes, not
features. A fix merges to the default branch first. It reaches a release branch
as a `git cherry-pick -x` of that commit, in its own pull request (see
[Backport a fix to a release branch](CONTRIBUTING.md#backport-a-fix-to-a-release-branch)).
The release commit is the exception: it goes into the release branch first
(see [Release the branch](#release-the-branch)). Tags exist only on release
branches, never on the default branch.

A release branch takes security and P0 patches only after its release ships, and
only the latest release branch takes them. Once the next release ships, the
previous branch is retired: it gets no more patches, and users move to the new
release.

## Release process

The releaser is the maintainer who runs the release. QA validates the release
branch.

### Cut the release branch

Cut the release branch about a week to ten days before the target date, from a
default-branch commit, and give QA the commit SHA:

```bash
git fetch origin
git push origin <main-sha>:refs/heads/release/YYYY.N
```

The branch must not already carry the new version literals. The tag job compares
each literal with the previous tip of the branch, and a new branch has no
previous tip, so a branch cut from a commit that already carries the release
literals tags nothing. The tooling has no recovery for this case; ask the
maintainers before you continue.

QA tests the branch. Fixes that QA finds follow
[Backport a fix to a release branch](CONTRIBUTING.md#backport-a-fix-to-a-release-branch).
The version literals on a release branch change only through the release commit
described next, so they always name the latest release on that branch; do not
change one in any other pick.

### Release the branch

A release takes two pull requests, in this order: the release commit into the
release branch, then a sync-back pull request that carries it into the default
branch. The default branch therefore carries only released version literals.

1. On a local branch from `origin/release/YYYY.N`, create the release commit,
   so the changelogs and generated files describe what the branch releases, not
   later work on the default branch. For each releasing component, change its
   version literal in its `__init__.py` and add its `CHANGELOG.md` section;
   `uv run .ci/release.py shortlog <component>` prints a skeleton. Release every
   component that changed, as the [cascade](#release-bundles) requires. Include
   the regenerated client files that the
   [pre-commit hook](CONTRIBUTING.md#install-the-pre-commit-hook) asks you to
   stage. Run `bash .ci/run-tests.sh all`, then make one commit.
2. Push the branch and open a pull request into the release branch. The full
   release checks and the cherry-pick check run on it.
3. QA approves the pull request from step 2. That approval is the sign-off: a
   maintainer does not merge the pull request without it, and no automated check
   enforces it. A maintainer squash-merges it.
4. CI detects the changed version literals and creates one tag per releasing
   component on the merge commit on the release branch. Components whose
   version literals did not change receive no new tag.
5. Run `git switch -c <branch> origin/main` and
   `git cherry-pick <squashed commit on the release branch>`, then open a
   sync-back pull request into the default branch that carries only that
   commit. The check on this pull request is the light check described under
   [Release checks](#release-checks). A maintainer squash-merges it.

If the release is abandoned before the merge in step 3, close its pull request; the
default branch never saw the new version.

### Release candidates and patches

A release candidate is a release of the same shape with an `X.Y.ZrcN` literal.
Its pull request only needs the releaser's approval and passing CI (QA
approval highly encouraged, but not enforced). A patch is a release with a new
patch number on the latest release branch: the fix merges to the default
branch, is picked onto the release branch, and the release follows steps 1
to 5. A patch is for security and P0 fixes only, and only on the latest
release branch. QA tests only the patch.

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
differs from its target branch. A component is releasing when its literal
changed and did not decrease. The Service depends on the Library, and the
Blueprint depends on both. A component's latest tag is its highest tag on any
branch, up to its literal.

On every pull request, the `Release checks` workflow runs `.ci/release.py
status` and `.ci/release.py check` against the pull request's target branch.
Status reports each component's literal, latest tags, distance since them, and
pull requests merged since. The check reports every rule outcome and fails on
any violation. On a push to a branch other than the default branch, the workflow
runs only the CI tooling tests. To run the check locally, use
`uv run .ci/release.py check --base origin/<target>`, adding `--light` for a
pull request into the default branch.

A pull request into a release branch gets the full check:

- It changes only version literals, `CHANGELOG.md` files and generated client
  files. A literal file may differ only in the version line.
- No literal decreases.
- The literal is a PEP 440 final or release candidate: `X.Y.Z` or `X.Y.ZrcN`.
- The tag for that literal does not exist yet.
- The component's `CHANGELOG.md` has a `## [<literal>]` heading.
- Each dependency not releasing in the same pull request has a tag, stable when
  the component release is stable.
- Each such dependency is at distance 0 from that tag.
- A stable release cannot include a dependency whose latest tag is a release
  candidate, or one released as a candidate in the same pull request.
- Every other component at a distance above 0 from its latest tag releases in
  the same pull request.

A pull request into the default branch that changes a literal is the sync-back
pull request of a release. The default branch holds later content and the tag
already exists, so only the first two rules apply.

A pull request into a release branch also runs the cherry-pick check. A commit
passes when its `(cherry picked from commit <sha>)` trailer names a commit
reachable from `main`, or when a patch-identical commit exists on `main`. A
non-merge commit that changes only version literal lines, `CHANGELOG.md` files
and generated client files, such as the release commit, also passes. A merge
commit fails, and so does any other commit that has no patch-identical commit on
`main` and whose trailer is missing or names a commit not on `main`; use `-x`
anyway so the trailer is there. Maintainers merge a pull request into a release branch only
when the cherry-pick check passes.

After a merge to a release branch that changes a version literal, the `Release`
workflow re-runs the rules above on the merged commit, except the file-set and
no-decrease rules, and, when they pass, creates one annotated `<component>-vX.Y.Z` tag per releasing component. It also
checks that each tagged component's build version at that commit equals its literal.
Tagging is all or nothing: any rule violation or failure in any component blocks
every tag, and the tags are pushed in one atomic push. A merge to the default
branch never starts the workflow. Re-run the job only for a transient failure
such as a rejected push; nothing reached the remote then, so the rerun reports
the tags as tagged. A tag already at the commit is reported as done. After a
rule violation or a build-version failure, nothing is tagged, and the job tags a
literal only in the push that changed it; ask the maintainers before you try
again.
