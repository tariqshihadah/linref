# linref Branching and Versioning Guide

This file is the standing reference for how this repository handles branches,
release preparation, version tags, and naming conventions after the v1.0
transition.

## Branch Roles

### Long-lived branches

- `main`
  - Latest released v1.x line
  - Protected branch
  - Every public v1.x release tag is created from `main`

- `develop`
  - Integration branch for the next unreleased v1.x work
  - Feature, fix, docs, and chore branches merge here first

- `maint/0.1`
  - Legacy maintenance branch for the old v0.1 API line
  - Only critical maintenance fixes should land here
  - No new feature work should target this branch

### Short-lived branches

- `feature/NNN-short-description`
  - New feature work
  - Branch from `develop`
  - Merge back into `develop`

- `fix/NNN-short-description`
  - Bug fixes that are not urgent production hotfixes
  - Branch from `develop`
  - Merge back into `develop`

- `docs/NNN-short-description`
  - Documentation-only work
  - Branch from `develop`
  - Merge back into `develop`

- `chore/NNN-short-description`
  - Tooling, CI, packaging, refactors, maintenance tasks
  - Branch from `develop`
  - Merge back into `develop`

- `release/X.Y.Z`
  - Temporary release stabilization branch
  - Branch from `develop` for normal releases
  - Merge into `main` when released
  - Merge back into `develop` so final release changes are preserved

- `hotfix/X.Y.Z-short-description`
  - Urgent production fix for an already released version
  - Branch from `main`
  - Merge into `main`
  - Merge back into `develop`

## Current Repository History

Before the v1.0 transition, the repo history looks like this:

- `master` is the legacy v0.1 line
- `dev` is an old historical integration branch
- `redesign` is the long-running v1.0 redevelopment branch

After the transition:

- `main` becomes the released v1.x line
- `develop` becomes the ongoing v1.x integration branch
- `maint/0.1` preserves the legacy line
- `redesign` is retired

Important:
- do not rename the old `dev` branch into `develop`
- create a fresh `develop` from the released v1.0 state instead

## Transition Schematic

```text
Before

origin/master ---> master
        \
         ---> dev
redesign -------> v1.0 work

After

origin/master/master ---> maint/0.1   (legacy only)

redesign ---> release/1.0.0 ---> main ---> tag v1.0.0
                                \
                                 ---> develop
```

## Standard Development Flow

For normal ongoing work:

```text
feature/* ----\
fix/* ---------> develop ---> release/X.Y.Z ---> main ---> tag vX.Y.Z
docs/* -------/                  \
chore/* ----/                    ---> merge back to develop
```

## Hotfix Flow

For urgent production fixes after a release:

```text
main ---> hotfix/X.Y.Z-* ---> main ---> tag vX.Y.Z
                     \
                      ---> develop
```

## Legacy Maintenance Flow

For old v0.1 support only:

```text
maint/0.1 ---> tag v0.1.x
```

Legacy fixes do not automatically flow into v1.x. If a fix matters for both
the legacy and v1 lines, it should be ported intentionally.

## Versioning Protocol

Use semantic versioning:

- `MAJOR` for breaking API changes
- `MINOR` for backward-compatible new features
- `PATCH` for backward-compatible bug fixes

Examples:

- `1.0.0` — first stable release of the redesign
- `1.0.1` — post-release bugfix
- `1.1.0` — new backward-compatible feature release
- `2.0.0` — next breaking redesign or major API shift

## Tagging Protocol

- Release tags use the format `vX.Y.Z`
- Tags are created on `main` only for v1.x releases
- Legacy tags for the old line may be created on `maint/0.1`

Examples:

- `v1.0.0`
- `v1.0.1`
- `v1.1.0`
- `v0.1.3` on `maint/0.1` if legacy support continues

## Release Protocol

For a normal v1 release:

1. Merge ready work into `develop`
2. Create `release/X.Y.Z` from `develop`
3. On the release branch, limit changes to:
   - version bumps
   - changelog updates
   - packaging fixes
   - docs fixes
   - release blockers
4. Run tests, build artifacts, and release smoke tests
5. Merge `release/X.Y.Z` into `main`
6. Tag `vX.Y.Z` on `main`
7. Merge `release/X.Y.Z` back into `develop`
8. Delete `release/X.Y.Z`

## Naming Conventions

Use branch names that are short, readable, and scoped to a single purpose.

Recommended patterns:

- `feature/123-add-migration-guide`
- `fix/145-project-buffer-edge-case`
- `docs/210-branching-guide`
- `chore/220-ci-wheel-smoke-test`
- `release/1.0.0`
- `hotfix/1.0.1-wheel-package-data`

If issue numbers are not available, a short descriptive name is still fine.

## What Not To Do

- Do not commit new feature work directly to `main`
- Do not use `redesign` as a permanent long-lived branch after v1.0
- Do not treat `master` as the future v1 mainline
- Do not rename old historical branches just to fit the new convention
- Do not merge legacy-only maintenance from `maint/0.1` into v1 without reviewing and porting it intentionally
