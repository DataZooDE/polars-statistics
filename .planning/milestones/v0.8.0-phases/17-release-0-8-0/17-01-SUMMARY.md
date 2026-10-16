---
phase: 17-release-0-8-0
plan: 01
subsystem: release-engineering
tags: [version-alignment, maturin, wheel, sdist, changelog, ci-gate, claude-code-plugin]

requires:
  - phase: 16-wheel-bundling
    provides: "Claude Code plugin tree bundled into the wheel at python/polars_statistics/_plugin/ (PKG-01), drift guard scripts/verify_wheel_bundling.py"
provides:
  - "All 18 version sources bumped and aligned on 0.8.0 (Cargo.toml, pyproject.toml, __version__, uv.lock, both plugin.json copies, both marketplace entries, all ten SKILL.md frontmatter fields)"
  - "scripts/check_version_alignment.py — single automated check (stdlib-only) + tests/test_version_alignment.py running it in every CI pytest job"
  - "scripts/release_smoke.py — reusable post-install smoke check (version, bundled plugin files/versions, expression API, PyModel API), reusable by 17-04/17-05"
  - "CHANGELOG.md [0.8.0] section finalized as release-notes source; fresh empty [Unreleased] above it"
  - "16/16 CI-equivalent local gate logs in target/release-gate/ bound to one gated HEAD SHA, all EXIT=0 (plan 17-02 reads these as push evidence)"
  - "Release wheel (polars_statistics-0.8.0-cp39-abi3-*.whl) and sdist (polars_statistics-0.8.0.tar.gz) built and smoke-verified on Python 3.9 and 3.12 clean venvs"
affects: [17-02-push-pr, 17-03-merge, 17-04-testpypi, 17-05-publish]

actuals:
  tokens: 8295
  tasks: 3
  commits: 4

tech-stack:
  added: []
  patterns:
    - "Version-alignment check: regex line-scan per TOML table / JSON field / SKILL.md frontmatter, no tomllib dependency (keeps the check runnable under Python 3.9)"
    - "release_smoke.py: stdlib + polars + numpy only, --expect CLI contract, single SMOKE_OK/SMOKE_FAIL terminal line for scripting"

key-files:
  created:
    - scripts/release_smoke.py
    - scripts/check_version_alignment.py
    - tests/test_version_alignment.py
  modified:
    - Cargo.toml
    - pyproject.toml
    - uv.lock
    - python/polars_statistics/__init__.py
    - plugins/polars-statistics/skills/polars-statistics-getting-started/SKILL.md
    - python/polars_statistics/_plugin/skills/polars-statistics-getting-started/SKILL.md
    - CHANGELOG.md

key-decisions:
  - "check_version_alignment.py locates TOML table values by line-scanning for the header then the next '[' header (no tomllib), so it runs under Python 3.9 exactly as CI's pytest wrapper does"
  - "release_smoke.py's expression-API check builds its own 60-row synthetic frame (g/x/y/a/b) rather than depending on any tests/ fixture, so it stays runnable against a bare installed wheel with no repo checkout"
  - "CHANGELOG Dependencies section's self-contradiction removed: the first anofox-regression bullet's '...are planned for a later release' deferral sentence was deleted (the Added section and the next bullet already show those additions fully exposed), and the next bullet's 'fully exposed by this phase' reworded to 'fully exposed in this release' to drop the internal-planning-unit wording"

requirements-completed: [REL-07]

coverage:
  - id: D1
    description: "All 18 version sources (Cargo.toml, pyproject.toml, __version__, uv.lock, both plugin.json copies, both marketplace entries, all ten SKILL.md) read 0.8.0"
    requirement: "REL-07"
    verification:
      - kind: unit
        ref: "tests/test_version_alignment.py#test_collect_versions_finds_at_least_18_sources_and_matches_runtime"
        status: pass
      - kind: other
        ref: "uv run python scripts/check_version_alignment.py --expect 0.8.0 (RESULT: PASS — 18 version sources aligned at 0.8.0)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Automated check fails provably on any version mismatch (negative case), run in every CI pytest job"
    requirement: "REL-07"
    verification:
      - kind: unit
        ref: "tests/test_version_alignment.py#test_check_fails_on_wrong_expect"
        status: pass
      - kind: unit
        ref: "tests/test_version_alignment.py#test_check_fails_on_mutated_skill_version"
        status: pass
      - kind: unit
        ref: "tests/test_version_alignment.py#test_check_fails_on_missing_source"
        status: pass
    human_judgment: false
  - id: D3
    description: "Release wheel built from the gated HEAD is named 0.8.0, bundles the plugin/5 skills at 0.8.0, and clean-venv installs on 3.9 and 3.12 pass the full pytest suite + release_smoke.py"
    requirement: "REL-07"
    verification:
      - kind: other
        ref: "target/release-gate/wheel312.log (SMOKE_OK 0.8.0; 1811 passed, 6 skipped)"
        status: pass
      - kind: other
        ref: "target/release-gate/wheel39.log (SMOKE_OK 0.8.0; 1811 passed, 6 skipped; imports from v39 site-packages)"
        status: pass
      - kind: other
        ref: "target/release-gate/bundling.log (RESULT: PASS — both halves)"
        status: pass
    human_judgment: false
  - id: D4
    description: "CHANGELOG [0.8.0] section finalized as the GitHub release-notes source, with a fresh empty [Unreleased] above it"
    requirement: "REL-07"
    verification:
      - kind: other
        ref: "Task 3 CHANGELOG_OK verify chain (heading order, empty Unreleased, compare links, required content bullets, no self-contradiction, >150 lines)"
        status: pass
    human_judgment: false
  - id: D5
    description: "Full CI-equivalent local gate (16 steps: fmt, clippy, rust_api, ruff, version, develop, pytest312, stubs, mypy, skills, mkdocs, wheel, bundling, sdist, wheel312, wheel39) passes against one gated HEAD SHA"
    requirement: "REL-07"
    verification:
      - kind: other
        ref: "target/release-gate/*.log (16 files, all header line == git rev-parse HEAD at commit 125f738, all EXIT=0)"
        status: pass
    human_judgment: false

duration: 50min
completed: 2026-10-16
status: complete
---

# Phase 17 Plan 01: Version Bump, Alignment Check, CHANGELOG, and Full Release Gate Summary

**Repository is a verified 0.8.0 release candidate: every version source aligned, an automated alignment check added, the stale getting-started skill note fixed, CHANGELOG finalized, and all 16 CI-equivalent local gate steps plus a 3.9/3.12 clean-venv wheel+sdist smoke passed on one HEAD — nothing pushed.**

## Performance

- **Duration:** ~50 min
- **Started:** 2026-10-16T00:04:00+02:00 (approx.)
- **Completed:** 2026-10-16T00:43:34+02:00
- **Tasks:** 3
- **Files modified:** 10 (3 created, 7 modified)

## Accomplishments

- Bumped `Cargo.toml`, `pyproject.toml`, `python/polars_statistics/__init__.py` `__version__`, and `uv.lock` from 0.7.0 → 0.8.0 (anofox pins `=0.5.22`/`=0.4.7` untouched); confirmed with `uv lock --check`.
- Added `scripts/release_smoke.py`: stdlib+polars+numpy post-install smoke (version, install-location-not-source-tree, bundled plugin files, bundled version fields, `ps.ols`/`ps.ttest_ind` expression API, `OLS` PyModel API) — reusable by 17-04/17-05.
- Built the 0.8.0 release wheel and installed it into a clean 3.12 venv; `release_smoke.py` printed `SMOKE_OK 0.8.0 ...` — proves the v0.6.0 misaligned-wheel-name failure cannot recur.
- Added `scripts/check_version_alignment.py` (stdlib-only, TDD RED→GREEN via `tests/test_version_alignment.py`) covering all 18 version sources (`source_paths`/`collect_versions`/`check`/`main`); wired into every CI pytest job (no new ci.yml step needed).
- Corrected the getting-started skill's stale "may still report 0.7.0" version notes (root + byte-identical bundled copy); confirmed the skill's examples still run `RESULT: GREEN`.
- Finalized `CHANGELOG.md`: renamed `[Unreleased]` → `[0.8.0] - 2026-10-16` with a fresh empty `[Unreleased]` above it; added the Claude Code plugin/skills headline + bundled-wheel bullets, the Python 3.9 `get_type_hints` note, the CI/tooling bullet, and the two compare links; removed a self-contradicting "planned for a later release" deferral.
- Ran the full 16-step CI-equivalent local gate (fmt, clippy, rust_api, ruff, version, develop, pytest312, stubs, mypy 1.11.2, skills, mkdocs --strict, wheel, bundling, sdist, wheel312, wheel39) against the gated CHANGELOG-commit HEAD `125f738b3ab165c8cb67d9a6f6602eb025529dcd` — **every step passed on the first run, no fix commits required**.

## Gate Evidence

**Gated HEAD SHA:** `125f738b3ab165c8cb67d9a6f6602eb025529dcd`

| Step | Result | Notes |
|------|--------|-------|
| fmt | EXIT=0 | `cargo fmt --all -- --check` |
| clippy | EXIT=0 | `cargo clippy --all-targets -- -D warnings` |
| rust_api | EXIT=0 | 22 passed; 0 failed |
| ruff | EXIT=0 | `uv run ruff check .` — All checks passed! |
| version | EXIT=0 | RESULT: PASS — 18 version sources aligned at 0.8.0 |
| develop | EXIT=0 | `uv run maturin develop --release` |
| pytest312 (editable) | EXIT=0 | 1811 passed, 6 skipped, 9 warnings in 6.94s |
| stubs | EXIT=0 | 45 classes, no drift (`git diff --exit-code`) |
| mypy 1.11.2 | EXIT=0 | Success: no issues found in 19 source files |
| skills | EXIT=0 | RESULT: GREEN — 5 skills, 58 PASS, 0 FAIL, 47 SKIP |
| mkdocs --strict | EXIT=0 | Documentation built in 1.83s |
| wheel | EXIT=0 | `polars_statistics-0.8.0-cp39-abi3-manylinux_2_39_x86_64.whl` |
| bundling | EXIT=0 | RESULT: PASS — both halves (drift guard + behavioral install) |
| sdist | EXIT=0 | `polars_statistics-0.8.0.tar.gz`; 5 bundled SKILL.md |
| wheel312 | EXIT=0 | SMOKE_OK 0.8.0; 1811 passed, 6 skipped, 9 warnings in 15.25s |
| wheel39 | EXIT=0 | SMOKE_OK 0.8.0; 1811 passed, 6 skipped, 9 warnings in 10.27s; imported from `target/release-gate/v39/lib/python3.9/site-packages/polars_statistics` |

**Wheel:** `target/wheels/polars_statistics-0.8.0-cp39-abi3-manylinux_2_39_x86_64.whl`
**Sdist:** `target/release-gate/sdist/polars_statistics-0.8.0.tar.gz`

**Three pytest summary lines:**
- 3.12 editable (pytest312): `1811 passed, 6 skipped, 9 warnings in 6.94s`
- 3.12 wheel (wheel312): `1811 passed, 6 skipped, 9 warnings in 15.25s`
- 3.9 wheel (wheel39): `1811 passed, 6 skipped, 9 warnings in 10.27s`

## Task Commits

Each task was committed atomically:

1. **Task 1: bump version sources + uv.lock, build release wheel, smoke it** — `6dc45dc` (chore)
2. **Task 2a: version-alignment check (TDD)** — `5fcba4f` (test)
2. **Task 2b: correct stale skill version notes** — `9d19f74` (docs)
3. **Task 3: finalize CHANGELOG [0.8.0]** — `125f738` (docs) — this is also the gated HEAD for the full local gate (Part B produced only gitignored logs/artifacts, no further commit)

**Plan metadata:** committed alongside this SUMMARY (see git log after this commit).

## Files Created/Modified

- `scripts/release_smoke.py` - Post-install smoke check (version/location/plugin files/plugin versions/expression API/PyModel API)
- `scripts/check_version_alignment.py` - Single automated check across 18 version sources
- `tests/test_version_alignment.py` - TDD RED→GREEN tests for the alignment check, incl. tmp_path mutation/missing-source negatives
- `Cargo.toml` - `[package] version` 0.7.0 → 0.8.0
- `pyproject.toml` - `[project] version` 0.7.0 → 0.8.0
- `uv.lock` - polars-statistics entry version 0.7.0 → 0.8.0 (1-line diff, `uv lock --check` clean)
- `python/polars_statistics/__init__.py` - `__version__` 0.7.0 → 0.8.0
- `plugins/polars-statistics/skills/polars-statistics-getting-started/SKILL.md` - corrected version-check example + "Version string vs. frontmatter version" Gotcha
- `python/polars_statistics/_plugin/skills/polars-statistics-getting-started/SKILL.md` - byte-identical bundled copy of the above
- `CHANGELOG.md` - `[0.8.0] - 2026-10-16` finalized; fresh empty `[Unreleased]`; plugin/skills/get_type_hints/ruff bullets added; Dependencies self-contradiction removed; compare links added

## Decisions Made

- `check_version_alignment.py` avoids `tomllib` (3.11+ only) by line-scanning TOML table headers directly, so the script itself runs unmodified under CI's Python 3.9 pytest jobs.
- Chose a pytest wrapper over a new `ci.yml` step for the alignment check (per plan's explicit discretion) since CI's 12 test-python jobs + coverage job already run `pytest tests/` across the full 3.9–3.12 × 3-OS matrix, and a new ci.yml step would enlarge the diff that already overlaps open PR #47.
- Removed the CHANGELOG's self-contradicting "planned for a later release" sentence (the Added section and the following Dependencies bullet already document those additions as shipped) rather than leaving both statements to stand.

## Deviations from Plan

None - plan executed exactly as written. (Two minor ruff import-order fixups inside `scripts/release_smoke.py`, found and fixed immediately via ruff's own output before the Task 1 commit, are routine authoring corrections, not deviations from plan intent.)

## Issues Encountered

None. All 16 gate steps and both negative-case tests passed on the first run; no retries, no node-repair escalation.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Gated HEAD `125f738b3ab165c8cb67d9a6f6602eb025529dcd` on local `main` is a fully verified 0.8.0 release candidate: 16/16 gate logs green, wheel + sdist built and smoke-verified on 3.9 and 3.12.
- Plan 17-02 (push + PR) can proceed: it reads the gate logs under `target/release-gate/` as push evidence and re-derives the gated SHA from `target/release-gate/fmt.log` line 1.
- No outward-facing action occurred in this plan: nothing pushed, tagged, or dispatched; `git status` shows only the untracked `.gsd/`.
- Note for 17-02: Open PR #47 (`ci/cost-guardrails`) touches `.github/workflows/ci.yml`; this milestone's own ci.yml edits (ruff/rust_api/wheel-bundling-drift steps, already landed in earlier phases) may conflict — surface to the user per CONTEXT, do not merge/modify #47 without approval.

---
*Phase: 17-release-0-8-0*
*Completed: 2026-10-16*

## Self-Check: PASSED

- All created files found on disk: `scripts/release_smoke.py`, `scripts/check_version_alignment.py`, `tests/test_version_alignment.py`.
- All task commit hashes found in `git log --oneline --all`: `6dc45dc`, `5fcba4f`, `9d19f74`, `125f738`.
- Plan-level verification re-run: `uv run python scripts/check_version_alignment.py --expect 0.8.0` → RESULT: PASS; `--expect 9.9.9` → exit 1. 16/16 `target/release-gate/*.log` bind to HEAD `125f738b3ab165c8cb67d9a6f6602eb025529dcd` with `EXIT=0`. Wheel and sdist exist and are named 0.8.0.
