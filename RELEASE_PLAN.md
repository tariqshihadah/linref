# linref v1.0 Release Plan

## Release Readiness Summary

The redesign is substantially complete. The core deliverable — replacing the
old standalone-class API with a pandas accessor pattern (`.lr`) — is
functionally done, and the new codebase is large enough and well-tested enough
to justify moving into release stabilization rather than continued redesign.

At the same time, the release surface is not ready yet. The biggest remaining
risks are packaging, artifact correctness, docs/build consistency, and branch
transition. The remaining work is less about core algorithms and more about
making sure a clean install, published docs, and release workflow all match the
new v1.0 reality.

**Production code**: ~12,500 lines across a cleanly organized module structure:
- `linref/ext/base.py` — LRS config object + `LRS_Accessor` (~2,900 lines)
- `linref/events/relate.py` — relational ops: `relate`, `overlay`, `intersect`, `project`, etc. (~2,200 lines)
- `linref/events/geometry.py` — M-enabled geometry handling (~1,200 lines)
- `linref/events/base.py` — `EventsData` computation engine (~1,200 lines)
- Supporting modules for modification, selection, spatial projection, integration, datasets, errors

**Tests**:
- ~3,000 lines across 4 primary test files covering `ext/base`, `events/relate`, `events/geometry`, and end-to-end integration
- Local `unittest` suite currently passes (`188` tests) in the project virtualenv

**Documentation**:
- `README.rst` updated
- `USAGE.md` is comprehensive (~600 lines)
- `CHANGELOG.md` has a v1.0 section, but it is still undated
- `USAGE_examples.ipynb` is a useful working notebook
- Sphinx / ReadTheDocs sources still need redesign-era cleanup before release

---

## Pre-Release Checklist

### Must-fix before release

- [x] Update version from `0.1.1` to `1.0.0` in `pyproject.toml`
- [x] Reconcile runtime dependency declarations in `pyproject.toml` so they reflect the actual redesign requirements
- [x] Add missing runtime dependencies used by the codebase, especially `scipy`
- [x] Update Development Status classifier from `1 - Planning` to `4 - Beta` or `5 - Production/Stable`
- [x] Ensure only the v1 package is distributed:
  - exclude `linref_011` and its tests from built artifacts
  - remove or archive legacy code if we do not intend to ship it
- [x] Ensure package data is distributed correctly:
  - include `linref/datasets/_data/*` in wheel and sdist artifacts
  - verify `linref.datasets.load()` works from an installed artifact, not just from the repo checkout
- [x] Restore artifact validation as part of release readiness:
  - build wheel and sdist
  - install into a clean environment
  - verify `import linref`
  - verify sample dataset loading
  - verify at least one basic accessor workflow
- [x] Update Sphinx / ReadTheDocs config and source files to reflect the redesign API and current versioning
- [ ] Add release date to the v1.0 entry in `CHANGELOG.md`

### Strongly recommended before release

- [ ] Write a migration guide (v0.1.x → v1.0 side-by-side API comparison)
- [x] Align secondary docs with actual behavior:
  - removed `linref/datasets/README.md` and `linref/tests/README.md` (duplicative of Sphinx docs and docstrings; maintenance burden)
  - any usage snippets that assume old defaults or old module layout
- [x] Remove or fix stale redesign leftovers such as `linref/ext/default.py`
- [x] Restore or add CI workflows for:
  - unit tests
  - package build
  - artifact smoke tests
  - docs build
- [x] Add 5–10 integration tests covering end-to-end workflows (load → set LRS → dissolve → relate → integrate)

### Good hardening work if time allows

- [ ] Complete docstrings for the most-used public API methods
- [ ] Type hints on the public API surface (`origin/15_type_hinting` branch was started)
- [ ] Performance documentation (rough "works well up to N rows" guidance)
- [ ] Expand polished API documentation on ReadTheDocs beyond the minimum needed for release correctness

---

## Branching Strategy

### Transition from current branches

Current branches reflect the old development history:
- `master` currently represents the legacy v0.1 line
- `dev` is a historical integration branch
- `redesign` contains the v1.0 redevelopment

For the v1.0 transition, `redesign` should become a temporary staging branch,
not a permanent long-lived branch.

### Repo-specific transition outline

The target transition is:

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

Important notes:
- do not rename the current `dev` branch into `develop`
- create a fresh `develop` from the released v1.0 state instead
- once `release/1.0.0`, `main`, and `develop` exist, `redesign` should be retired
- `master` should stop being treated as the future mainline and instead serve as the source for `maint/0.1`

### Transition execution sequence

When we are ready to perform the branch transition, the intended sequence is:

1. Preserve any uncommitted work currently on `redesign`
2. Create `maint/0.1` from `master` or `v0.1.2`
3. Create `release/1.0.0` from `redesign`
4. Freeze `redesign` and do all final v1.0 stabilization on `release/1.0.0`
5. Create or merge into `main` from the stabilized `release/1.0.0`
6. Tag `v1.0.0` on `main`
7. Create a fresh `develop` from `main`
8. Retire `redesign`
9. Triage remaining old topic branches and port only what is still relevant

### Permanent branches after v1.0

| Branch | Purpose |
|---|---|
| `main` | Released v1.x line. Protected. Tagged releases live here. |
| `develop` | Integration branch for upcoming v1.x work. Feature, fix, docs, and chore branches target this branch. |
| `maint/0.1` | Legacy maintenance branch for v0.1.x if a patch or support update is needed after v1.0 ships. |

### Short-lived branches (deleted after merge)

| Pattern | Branched from | Merges into | When to use |
|---|---|---|---|
| `feature/NNN-short-description` | `develop` | `develop` | New capabilities |
| `fix/NNN-short-description` | `develop` | `develop` | Non-urgent bug fixes tied to an issue |
| `docs/NNN-short-description` | `develop` | `develop` | Documentation-only work |
| `chore/NNN-short-description` | `develop` | `develop` | Tooling, CI, packaging, maintenance tasks |
| `hotfix/NNN-short-description` | `main` | `main` + `develop` | Urgent fix to a released v1.x version |
| `release/X.Y.Z` | `develop` | `main` + `develop` | Pre-release stabilization (version bump, changelog, last-minute fixes) |

### Legacy branch policy

- `maint/0.1` receives only critical legacy fixes
- no new feature work lands on `maint/0.1`
- if legacy support is no longer desired, `maint/0.1` can remain as a frozen archival branch after v1.0

### Tagging convention

`v1.0.0`, `v1.0.1`, `v1.1.0` on `main` only, following semver.

---

## v1.0 Transition Steps

1. Create `maint/0.1` from the current legacy release line (`master` / `v0.1.2`) so the old API has a stable maintenance home.
2. Cut `release/1.0.0` from `redesign` for final stabilization.
3. Fix packaging and artifact blockers:
   - dependency declarations
   - package discovery
   - package data inclusion
   - removal/exclusion of `linref_011`
4. Restore CI and add artifact smoke tests.
5. Update Sphinx / ReadTheDocs config and remove stale API references.
6. Write the v0.1.x → v1.0 migration guide.
7. Align secondary docs and remove stale code leftovers.
8. Add end-to-end integration tests for the highest-value workflows.
9. Update version, classifier, and `CHANGELOG.md` release date.
10. Run final verification:
    - unit tests
    - integration tests
    - wheel + sdist build
    - clean install smoke tests
    - docs build
11. Create `main` from the stabilized release branch if needed, or merge `release/1.0.0` into the new `main`.
12. Tag `v1.0.0` on `main`.
13. Seed `develop` from the released v1.0 state and merge `release/1.0.0` into it.
14. Retire `redesign` once the release is complete.

### Post-v1.0 steady-state flow

After the transition is complete, normal work should flow like this:

```text
feature/* ----\
fix/* ---------> develop ---> release/X.Y.Z ---> main ---> tag vX.Y.Z
docs/* -------/                  \
chore/* ----/                    ---> merge back to develop

main ---> hotfix/X.Y.Z-* ---> main ---> tag vX.Y.Z
                     \
                      ---> develop
```

### Pending remote branches to triage

Review and either rebase onto `develop`, cherry-pick selectively, or close:
- `origin/13_merge_geoseries`
- `origin/15_type_hinting`
- `origin/8_project_matching`
- `origin/29_hardcoded_index`

### Immediate priority order

If time is tight, execute in this order:
1. ~~Packaging correctness~~ — done
2. ~~Artifact smoke tests and CI~~ — done
3. ~~Docs / RTD correctness~~ — done
4. Migration guide (v0.1.x → v1.0)
5. ~~Integration tests~~ — done (8 tests in `test_integration.py`)
6. ~~Secondary docs and cleanup~~ — done
7. Add release date to `CHANGELOG.md` and finalize version bump
8. Branch transition: `release/1.0.0` → `main` → tag `v1.0.0`
