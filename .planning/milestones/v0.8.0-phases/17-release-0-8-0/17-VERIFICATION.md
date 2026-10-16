---
phase: 17-release-0-8-0
verified: 2026-10-16T00:24:01Z
status: passed
score: 3/3 must-haves verified
covered_files:
  - ".planning/phases/17-release-0-8-0/17-01-PLAN.md"
  - ".planning/phases/17-release-0-8-0/17-01-SUMMARY.md"
  - ".planning/phases/17-release-0-8-0/17-02-PLAN.md"
  - ".planning/phases/17-release-0-8-0/17-02-SUMMARY.md"
  - ".planning/phases/17-release-0-8-0/17-03-PLAN.md"
  - ".planning/phases/17-release-0-8-0/17-03-SUMMARY.md"
  - ".planning/phases/17-release-0-8-0/17-04-PLAN.md"
  - ".planning/phases/17-release-0-8-0/17-04-SUMMARY.md"
  - ".planning/phases/17-release-0-8-0/17-05-PLAN.md"
  - ".planning/phases/17-release-0-8-0/17-05-SUMMARY.md"
  - ".planning/phases/17-release-0-8-0/17-CONTEXT.md"
  - "CHANGELOG.md"
  - "Cargo.toml"
  - "plugins/polars-statistics/skills/polars-statistics-getting-started/SKILL.md"
  - "pyproject.toml"
  - "python/polars_statistics/__init__.py"
  - "python/polars_statistics/_plugin/skills/polars-statistics-getting-started/SKILL.md"
  - "scripts/check_version_alignment.py"
  - "scripts/release_smoke.py"
  - "tests/test_version_alignment.py"
  - "uv.lock"
covered_digest: "v1:sha256:e92db5294aa28481c438036a9d1a1f81dc075b4765cfbacbaab10e53affcc0dd"
behavior_unverified: 0
overrides_applied: 0
---

# Phase 17: Release 0.8.0 Verification Report

**Phase Goal:** polars-statistics 0.8.0 — carrying the plugin, the five verified skills, and the wheel bundling — is built, published to production PyPI via the existing OIDC pipeline, and confirmed installable with skills present.
**Verified:** 2026-10-16T00:24:01Z
**Status:** passed
**Re-verification:** No — initial verification

All verification in this report was performed independently against the live GitHub repository, GitHub Actions API, PyPI JSON API, and a fresh scratch venv — not derived from SUMMARY.md claims. Every command below was re-run by the verifier in its own process.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Version reads 0.8.0 across all sources — `Cargo.toml`, `pyproject.toml`, `__version__`, `plugin.json`, every `SKILL.md` version field — all aligned | ✓ VERIFIED | `git archive v0.8.0` extracted to a scratch dir (independent of local working tree); `scripts/check_version_alignment.py --expect 0.8.0` run from that extracted copy → `RESULT: PASS — 18 version sources aligned at 0.8.0`. Manually spot-checked `Cargo.toml` (`version = "0.8.0"`), `pyproject.toml` (`version = "0.8.0"`), `python/polars_statistics/__init__.py` (`__version__ = "0.8.0"`), both `plugin.json` copies, `marketplace.json`, and all 5 `SKILL.md` frontmatter fields (root + bundled copies) — all read `0.8.0`. |
| 2 | sdist and platform wheels are built and published to production PyPI via the GitHub Actions OIDC pipeline (no hardcoded tokens) | ✓ VERIFIED | `git show v0.8.0 --no-patch`: tag resolves to merge commit `fd52d0781cd9487876c13638b38e9612cd9ee350` (PR #48 merge). `gh release view v0.8.0 --json tagName,isDraft,isPrerelease,targetCommitish` → `tagName=v0.8.0, isDraft=false, isPrerelease=false, targetCommitish=fd52d07...` (published, not draft/prerelease). `gh run view 38824647551 --json conclusion,event,headBranch,headSha,jobs` → `conclusion=success, event=release, headSha=fd52d07...` (matches the tag); all 6 build jobs (sdist, ubuntu x86_64/aarch64, macos x86_64/arm64, windows x86_64) `success`; `Publish to PyPI`=`success`; `Publish to TestPyPI`=`skipped`. Read `.github/workflows/publish.yml` directly: `publish-pypi` job declares `permissions: id-token: write` with no `password`/token input on the PyPI publish step (OIDC trusted publishing); the only stored-token usage (`secrets.TEST_PYPI_API_TOKEN`) is in the separate TestPyPI step, which did not run (skipped). `curl https://pypi.org/pypi/polars-statistics/0.8.0/json` → `info.version=0.8.0`, 6 files: sdist + wheels for manylinux x86_64/aarch64, macOS x86_64 (macosx_10_12)/arm64 (macosx_11_0), win_amd64. |
| 3 | A post-release smoke check confirms the published 0.8.0 installs, imports cleanly, and has the skill files present in the installed package | ✓ VERIFIED | Fresh scratch venv (not the repo checkout): `pip install polars-statistics==0.8.0` from the real PyPI index → resolved and installed cleanly. `python -c "import polars_statistics as ps; print(ps.__version__)"` → `0.8.0`, install location resolved to the venv's `site-packages` (not the repo source tree). `find _plugin -name SKILL.md` → exactly 5 files (`polars-statistics-{anova,getting-started,hypothesis-tests,regression,results}`), all with `version: "0.8.0"` frontmatter. Ran a functional check beyond import: `df.group_by('g').agg(ps.ols('y','x'))` returned a valid per-group regression struct, and `polars_statistics.OLS().fit(...)` produced usable coefficients. Scratch venv removed after the check. |

**Score:** 3/3 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `v0.8.0` git tag | Points at the reviewed PR #48 merge commit | ✓ VERIFIED | `git rev-list -n1 v0.8.0` == `fd52d0781cd9487876c13638b38e9612cd9ee350`; `gh pr view 48 --json mergeCommit` == same SHA; `state=MERGED`, `baseRefName=main` |
| GitHub release `v0.8.0` | Published, non-draft, non-prerelease, notes from CHANGELOG | ✓ VERIFIED | `gh release view v0.8.0` confirms `isDraft=false`, `isPrerelease=false`, `publishedAt=2026-10-15T23:37:00Z` |
| Production PyPI package `polars-statistics==0.8.0` | sdist + 5 platform wheels, OIDC-published | ✓ VERIFIED | PyPI JSON API: 6 files, matching filename patterns for all 5 target platforms + sdist |
| Installed `_plugin/skills/*/SKILL.md` (5 files) | Present in the wheel, each at version 0.8.0 | ✓ VERIFIED | Confirmed by direct filesystem inspection of a fresh `pip install` in a scratch venv |
| `scripts/check_version_alignment.py` | Single automated alignment check across all version sources | ✓ VERIFIED | Independently executed against an extracted copy of the `v0.8.0` tag tree (not trusting the SUMMARY's prior run) — passed |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `v0.8.0` tag | PR #48 merge commit | `git rev-list -n1` / `gh pr view --json mergeCommit` | ✓ WIRED | Both resolve to `fd52d0781cd9487876c13638b38e9612cd9ee350` |
| GitHub `release:published` event | `.github/workflows/publish.yml` `publish-pypi` job | `on: release types: [published]` trigger, run 38824647551 has `event=release` | ✓ WIRED | Confirmed via `gh run view` — the release event fired the exact workflow run that uploaded to PyPI |
| `publish-pypi` job | PyPI (OIDC) | `permissions: id-token: write`, no stored credential on the PyPI step | ✓ WIRED | Read directly from `.github/workflows/publish.yml` lines 118-145; `pypa/gh-action-pypi-publish` step has no `password:` input |
| Installed wheel | `_plugin/skills/*/SKILL.md` | Wheel bundling (Phase 16) + this release's version bump | ✓ WIRED | Confirmed present on disk after a real `pip install` from the public index, not from a local build |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| REL-07 | 17-01 | Version bumped 0.7.0→0.8.0 across all sources, all aligned | ✓ SATISFIED | Independently re-verified via `check_version_alignment.py` against the extracted `v0.8.0` tag tree — 18/18 sources read 0.8.0 |
| REL-08 | 17-02, 17-03, 17-04, 17-05 | 0.8.0 published to production PyPI via OIDC pipeline, post-release smoke confirms import + skills present | ✓ SATISFIED | Independently re-verified via `gh release view`, `gh run view 38824647551`, PyPI JSON API, and a fresh scratch-venv install + functional smoke |

Both requirements are marked `[x]` / `Complete` in `.planning/REQUIREMENTS.md`, consistent with this independent verification. No orphaned requirements found for Phase 17.

### Anti-Patterns Found

Scanned all files changed by this phase (`Cargo.toml`, `pyproject.toml`, `uv.lock`, `python/polars_statistics/__init__.py`, `scripts/release_smoke.py`, `scripts/check_version_alignment.py`, `tests/test_version_alignment.py`, both `getting-started/SKILL.md` copies, `CHANGELOG.md`) for `TBD|FIXME|XXX|TODO|HACK|PLACEHOLDER` and stub patterns.

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `CHANGELOG.md` | 424 | "inference is not yet implemented for WLS" | ℹ️ Info | Pre-existing documented limitation note in historical changelog content (not new debt introduced by this phase's version-bump/release work); not a blocker |
| `plugins/.../getting-started/SKILL.md` (both copies) | 193, 239 | "not available as expression" / "Not available" | ℹ️ Info | Documents real, intentional API limitations (GLMM has no expression form; PLS has no `normalize=True`), not stub/placeholder code |

No 🛑 Blockers found. No `TBD`/`FIXME`/`XXX` debt markers in any file this phase touched.

### Behavioral Spot-Checks / Probe Execution

This phase has no `scripts/*/tests/probe-*.sh` convention — it is a release/publish phase, not a migration/tooling phase. Its "probes" are the real external release artifacts (GitHub tag, release, Actions run, PyPI listing, a live `pip install`), all independently re-executed above rather than through a probe script.

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Tag resolves to the merged commit | `git rev-list -n1 v0.8.0` | `fd52d0781cd9487876c13638b38e9612cd9ee350` | ✓ PASS |
| Version alignment on tagged tree | `git archive v0.8.0 \| tar -x` then `check_version_alignment.py --expect 0.8.0` | `RESULT: PASS` | ✓ PASS |
| GitHub release is live, non-draft | `gh release view v0.8.0 --json ...` | `isDraft=false, isPrerelease=false` | ✓ PASS |
| Publish workflow succeeded via OIDC | `gh run view 38824647551 --json ...` | `conclusion=success`, PyPI step `success`, TestPyPI `skipped` | ✓ PASS |
| PyPI lists the release | `curl pypi.org/pypi/polars-statistics/0.8.0/json` | 6 files, version 0.8.0 | ✓ PASS |
| Fresh install from PyPI works | `pip install polars-statistics==0.8.0` in scratch venv | `__version__ == '0.8.0'`, installed outside repo tree | ✓ PASS |
| Bundled skills present post-install | `find _plugin -name SKILL.md` | 5 files, all `version: "0.8.0"` | ✓ PASS |
| Functional API smoke | `df.group_by(...).agg(ps.ols(...))`, `OLS().fit(...)` | Valid struct output; coefficients produced | ✓ PASS |

### Human Verification Required

None. Every must-have was independently reproducible via programmatic checks (git, `gh`, PyPI JSON API, and a real `pip install`). The one item the 17-05 SUMMARY itself flagged as human-only (visually confirming the rendered PyPI/GitHub release HTML pages, since `curl` hits PyPI's anti-bot challenge) is cosmetic — the underlying facts it would confirm (version, file list, release metadata) are already fully proven here via the JSON API and `gh` CLI, so it is not required to confirm the phase goal.

### Gaps Summary

None. All three ROADMAP success criteria for Phase 17 were independently reproduced against live external systems (GitHub tag/release/Actions, PyPI, and a fresh package install) rather than taken from SUMMARY.md. Both REL-07 and REL-08 are satisfied.

---

_Verified: 2026-10-16T00:24:01Z_
_Verifier: Claude (gsd-verifier)_
