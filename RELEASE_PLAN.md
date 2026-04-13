# linref v1.0 Release Plan

## Codebase Status

The redesign is substantially complete. The core deliverable — replacing the old standalone-class API with a pandas accessor pattern (`.lr`) — is functionally done.

**Production code**: ~12,500 lines across a cleanly organized module structure:
- `linref/ext/base.py` — LRS config object + `LRS_Accessor` (~2,900 lines)
- `linref/events/relate.py` — relational ops: `relate`, `overlay`, `intersect`, `project`, etc. (~2,200 lines)
- `linref/events/geometry.py` — M-enabled geometry handling (~1,200 lines)
- `linref/events/base.py` — `EventsData` computation engine (~1,200 lines)
- Supporting modules for modification, selection, spatial projection, integration, datasets, errors

**Tests**: ~3,000 lines across 3 test files covering `ext/base`, `events/relate`, and `events/geometry`.

**Documentation**: `README.rst` updated, `USAGE.md` comprehensive (~600 lines), `CHANGELOG.md` has v1.0 section (undated), `USAGE_examples.ipynb` is a working notebook.

---

## Pre-Release Checklist

### Must-fix before release

- [ ] Update version from `0.1.1` to `1.0.0` in `pyproject.toml`
- [ ] Reconcile dependency versions in `pyproject.toml` — currently stale (`shapely>=1.7`, `pandas>=1.1`, `geopandas>=0.10.2`); should reflect actual minimums (`shapely>=2.0`, `pandas>=2.0`, `geopandas>=0.14`, `numpy>=2.1`, `scipy>=1.4.1`) per `requirements.txt`
- [ ] Update Development Status classifier from `1 - Planning` to `4 - Beta` or `5 - Production/Stable`
- [ ] Add release date to the v1.0 entry in `CHANGELOG.md`

### Should do before release

- [ ] Write a migration guide (v0.1.x → v1.0 side-by-side API comparison)
- [ ] Complete docstrings for public API methods — gaps in `events/selection.py`, `events/modify.py`, `events/analyze.py`, and several `LRS_Accessor` methods
- [ ] Remove `linref_011/` directory (dead code from old v0.1 design)
- [ ] Add 5–10 integration tests covering end-to-end workflows (load → set LRS → dissolve → relate → integrate)

### Nice-to-have

- [ ] Type hints on the public API surface (`origin/15_type_hinting` branch was started)
- [ ] Performance documentation (rough "works well up to N rows" guidance)
- [ ] Finalize and deploy Sphinx docs to ReadTheDocs

---

## Branching Strategy

### Permanent branches

| Branch | Purpose |
|---|---|
| `main` | Always reflects a released, tagged version. Only receives merges from `release/` or `hotfix/` branches. |
| `develop` | Integration branch. All feature and fix branches target this. Formalize the existing `dev` branch as `develop`. |

### Short-lived branches (deleted after merge)

| Pattern | Branched from | Merges into | When to use |
|---|---|---|---|
| `feature/NNN-short-description` | `develop` | `develop` | New capabilities |
| `fix/NNN-short-description` | `develop` | `develop` | Non-urgent bug fixes tied to an issue |
| `hotfix/NNN-short-description` | `main` | `main` + `develop` | Urgent fix to a released version |
| `release/X.Y.Z` | `develop` | `main` + `develop` | Pre-release stabilization (version bump, changelog, last-minute fixes) |

### Tagging convention

`v1.0.0`, `v1.0.1`, `v1.1.0` on `main` only, following semver.

---

## v1.0 Transition Steps

1. Reconcile dependency versions in `pyproject.toml`
2. Remove `linref_011/` directory
3. Cut a `release/1.0.0` branch from `redesign` for final stabilization
4. Write migration guide
5. Complete docstrings for most-used public methods
6. Add integration tests
7. Update version, classifier, and CHANGELOG date
8. Final test run, build wheel, upload to PyPI
9. Merge `release/1.0.0` → `main`, tag `v1.0.0`
10. Merge `release/1.0.0` → `develop` to carry version bump forward
11. Delete `redesign` branch

### Pending remote branches to triage

Review and either rebase onto `develop` or close:
- `origin/13_merge_geoseries`
- `origin/15_type_hinting`
- `origin/8_project_matching`
- `origin/29_hardcoded_index`
