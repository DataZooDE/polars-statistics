---
phase: 17-release-0-8-0
plan: 05
subsystem: release-engineering
tags: [github-release, pypi, oidc, trusted-publishing, post-release-smoke, maturin]

requires:
  - phase: 17-release-0-8-0
    provides: "17-03: PR #48 merged into GitHub main as merge commit fd52d0781cd9487876c13638b38e9612cd9ee350"
  - phase: 17-release-0-8-0
    provides: "17-04: TestPyPI dry run declined (stored-token risk; redundant with production build matrix)"
provides:
  - "Tag v0.8.0 and published GitHub release v0.8.0 at merge commit fd52d0781cd9487876c13638b38e9612cd9ee350"
  - "polars-statistics 0.8.0 live on production PyPI (sdist + 5 platform wheels) via OIDC trusted publishing, publish run 38824647551"
  - "Post-release smoke passed on clean Python 3.12 and 3.9 venvs installed from PyPI: import, version, bundled _plugin skills (5x SKILL.md + manifests at 0.8.0), group_by OLS expression, ttest_ind expression, OLS PyModel fit"
  - "REL-08 fully satisfied: production PyPI publish + post-release smoke complete"
affects: []

actuals:
  tokens: 4200
  tasks: 3
  commits: 0
  plan_head_before: bdaa43d7d5884f7ca59e5fa9ef79a3ae97deff0b

tech-stack:
  added: []
  patterns:
    - "Pre-publish readiness verified against an extracted copy of the tag-target commit (git archive), independent of local working-tree state, mirroring 17-03's main-tree verification pattern"
    - "Immediately-before-acting re-check of absence (tag/release/PyPI) repeated right after the checkpoint, not just at Task 1, to catch any change in the window between readiness report and the user's go-ahead"
    - "Post-release smoke run twice (3.12, 3.9) in throwaway uv venvs seeded fresh from the default PyPI index with --no-cache-dir, asserting the installed package dir resolves outside the repo's python/ source tree (proves the artifact under test, not the source checkout)"

key-files:
  created:
    - target/release-gate/publish-readiness.md (gitignored)
    - target/release-gate/release-notes-0.8.0.md (gitignored, CHANGELOG [0.8.0] section verbatim, 313 lines)
    - target/release-gate/pypi-smoke-3.12.log (gitignored)
    - target/release-gate/pypi-smoke-3.9.log (gitignored)
    - target/release-gate/pypi-0.8.0.json (gitignored)
    - target/release-gate/publish-run-id.txt (gitignored, contents "38824647551")
  modified: []

key-decisions:
  - "User's production-publish approval was obtained in the orchestrator session (\"go and the release is approved.\" / \"do it\" after seeing the full flow) and reconfirmed in this execution's instructions as conditional on all pre-publish readiness checks passing. All checks passed (main CI success, 18/18 version alignment at M, tag/release/PyPI absence, OIDC-only publish-pypi job), so the production publish proceeded per that pre-answered approval."
  - "Release title used the plan's proposed default verbatim: \"v0.8.0 — Claude Code Skills Plugin & Upstream Refresh\" (no custom title supplied)."
  - "Did not wait on or interact with PR #47 (ci/cost-guardrails) or the stale worktree .claude/worktrees/agent-a88ea0e6ef2dbcd4c — both are reported as follow-ups per CONTEXT, not acted on in this plan."

requirements-completed: [REL-08]

coverage:
  - id: D1
    description: "Pre-publish readiness verified before any irreversible action: main CI success on M, 18/18 version alignment at M (extracted tree), tag/release/PyPI absence, publish-pypi job uses OIDC (id-token: write, no stored credential), release notes extracted from CHANGELOG [0.8.0]"
    requirement: "REL-08"
    verification:
      - kind: other
        ref: "Task 1 automated verify chain: echoed PUBLISH_READY"
        status: pass
    human_judgment: false
  - id: D2
    description: "User's explicit, fresh approval for the irreversible production publish, obtained immediately before tagging (not inherited from earlier approvals)"
    requirement: "REL-08"
    verification:
      - kind: manual_procedural
        ref: "User's literal answers in the orchestrator session: \"go and the release is approved.\" then, after being shown the full merge -> skip TestPyPI -> tag/release -> PyPI -> smoke flow, \"do it\" — relayed into this execution as the pre-answered Task 2 checkpoint, conditioned on (and gated by) all Task 1 readiness checks passing, which they did"
        status: pass
    human_judgment: false
  - id: D3
    description: "Tag v0.8.0 created on merge commit M; GitHub release v0.8.0 published (not draft, not prerelease) with CHANGELOG [0.8.0] notes"
    requirement: "REL-08"
    verification:
      - kind: other
        ref: "git rev-list -n1 v0.8.0 == fd52d0781cd9487876c13638b38e9612cd9ee350; gh release view v0.8.0 --json tagName,isDraft,isPrerelease == 'v0.8.0 false false'"
        status: pass
    human_judgment: false
  - id: D4
    description: "Release-event publish.yml run (38824647551) succeeded: all 6 build jobs (sdist + 5 platform wheels) succeeded, Publish to PyPI succeeded via OIDC, Publish to TestPyPI skipped"
    requirement: "REL-08"
    verification:
      - kind: other
        ref: "gh run view 38824647551 --json jobs — all 6 Build* jobs conclusion=success, 'Publish to PyPI' conclusion=success, 'Publish to TestPyPI' conclusion=skipped; gh run view --json conclusion,event,headBranch == 'success release v0.8.0'"
        status: pass
    human_judgment: false
  - id: D5
    description: "PyPI lists exactly the sdist plus 5 platform wheels (manylinux x86_64/aarch64, macOS x86_64/arm64, win_amd64) for 0.8.0 — same 6 file shapes as 0.7.0"
    requirement: "REL-08"
    verification:
      - kind: other
        ref: "https://pypi.org/pypi/polars-statistics/0.8.0/json — info.version=0.8.0, 6 urls, all 6 filename patterns matched"
        status: pass
    human_judgment: false
  - id: D6
    description: "Post-release smoke on clean Python 3.12 and 3.9 venvs installing polars-statistics==0.8.0 from the default PyPI index: import, __version__, bundled _plugin skills (5x SKILL.md + manifests at 0.8.0), group_by OLS expression, ttest_ind expression, OLS PyModel fit"
    requirement: "REL-08"
    verification:
      - kind: other
        ref: "target/release-gate/pypi-smoke-3.12.log and pypi-smoke-3.9.log: both contain 'SMOKE_OK 0.8.0 ' plus a downloaded 'polars_statistics-0.8.0-cp39-abi3-' wheel filename, both end with 'EXIT=0'"
        status: pass
    human_judgment: false
  - id: D7
    description: "Human-observable confirmation of the live PyPI and GitHub release pages (README rendering, release notes content) — left as a judgment check for the user since automated curl to the PyPI HTML page hit PyPI's bot-challenge page"
    verification: []
    human_judgment: true
    rationale: "curl against https://pypi.org/project/polars-statistics/0.8.0/ returned PyPI's 'Client Challenge' anti-bot page rather than rendered HTML; the JSON API (info.version, file list) was used instead and fully verified D5, but visually confirming the rendered README and release notes formatting is a human-only check per the plan's <human-check> block."
    rationale_note: "Programmatic facts (version, file count/shapes, release non-draft/non-prerelease) are fully proven by D3-D5; only the visual rendering is unverified here."

duration: 55min
completed: 2026-10-16
status: complete
---

# Phase 17 Plan 05: Production Publish — Tag, Release, PyPI, Post-release Smoke Summary

**polars-statistics 0.8.0 shipped to production PyPI via the OIDC trusted-publishing pipeline (publish run 38824647551, all 6 build jobs + PyPI upload succeeded); tag/release v0.8.0 pin the exact reviewed merge commit, and clean-venv smoke on Python 3.12 and 3.9 confirms the bundled Claude Code skills plugin installs and works from the real package index.**

## Performance

- **Duration:** ~55 min (dominated by the publish.yml build matrix: ~28 min for all 6 build jobs including the aarch64 cross-build and two macOS targets)
- **Started:** 2026-10-16T00:10:00Z (approx.)
- **Completed:** 2026-10-16T01:05:00Z (approx.)
- **Tasks:** 3 (Task 1 auto — readiness, Task 2 checkpoint:decision — pre-answered via orchestrator, Task 3 auto — publish + verify + smoke)
- **Files modified:** 0 tracked repository files (all artifacts under `target/release-gate/`, gitignored by the repo-root `/target/` ignore rule)

## Accomplishments

- **Task 1 (pre-publish readiness, read-only):** Re-derived M (`fd52d0781cd9487876c13638b38e9612cd9ee350`) from the merged PR #48, confirmed it equals `origin/main`'s tip (nothing landed since 17-03). Waited for main CI run `38821933548` to complete — it had been `queued` since 17-03 but completed with `conclusion: success` during this task (headSha confirmed == M). Extracted M via `git archive` and ran `check_version_alignment.py --expect 0.8.0` against the extracted tree (not the local working copy): `RESULT: PASS — 18 version sources aligned at 0.8.0`. Confirmed tag `v0.8.0`, release `v0.8.0`, and PyPI `0.8.0` were all absent (tag: empty `ls-remote`; release: HTTP 404; PyPI: HTTP 404). Confirmed `publish.yml` at M contains the `release: types: [published]` trigger and the `publish-pypi` job carries `permissions: id-token: write` with no stored-credential input (trusted publishing, no hardcoded token — ROADMAP SC-2). Extracted release notes verbatim from `CHANGELOG.md`'s `[0.8.0] - 2026-10-16` section (313 lines, mentions `_plugin`, dated today). Wrote `target/release-gate/publish-readiness.md` with the full report and the exact Task 3 command. Verify chain echoed `PUBLISH_READY`.
- **Task 2 (checkpoint:decision — production publish, gate=blocking-human):** The user's approval for the production publish was obtained in the orchestrator session before this execution began (and conditioned on all Task 1 readiness checks passing, which they did): "go and the release is approved." / "do it", after being shown the complete flow through tag/release/PyPI/smoke. No GitHub or PyPI write happened before this task's readiness checks were confirmed clean.
- **Task 3 (publish + verify + post-release smoke):**
  - Re-derived M and re-ran the fast absence checks immediately before acting — nothing had changed since Task 1.
  - `gh release create v0.8.0 --target fd52d07... --title "v0.8.0 — Claude Code Skills Plugin & Upstream Refresh" --notes-file target/release-gate/release-notes-0.8.0.md --latest` — succeeded. `git rev-list -n1 v0.8.0` confirmed equal to M after `git fetch --tags`.
  - Found the release-event publish run via `gh run list --workflow publish.yml --event release`: run **38824647551**, `headBranch=v0.8.0`, `headSha=fd52d07...` (matches M exactly).
  - Watched the run across 5 `gh run watch` calls (each timing out at 570s while the build matrix progressed — ubuntu x86_64 wheel finished first at ~9m, macOS arm64 at ~15m, ubuntu aarch64 (QEMU cross-build) last at ~9m into its own stage, then the PyPI upload). Final watch call exited 0 (success).
  - Jobs table: all 6 Build jobs (`sdist`, `ubuntu-latest x86_64`, `ubuntu-latest aarch64`, `macos-latest x86_64`, `macos-14 aarch64`, `windows-latest x86_64`) = `success`; **Publish to PyPI = success**; **Publish to TestPyPI = skipped**.
  - PyPI file list (`pypi.org/pypi/polars-statistics/0.8.0/json`): exactly 6 files — `polars_statistics-0.8.0.tar.gz`, and wheels for manylinux x86_64, manylinux aarch64, macOS x86_64 (`macosx_10_12`), macOS arm64 (`macosx_11_0`), and `win_amd64` — the same 6 shapes as 0.7.0.
  - Post-release smoke, Python 3.12: fresh `uv venv`, `pip install --no-cache-dir polars-statistics==0.8.0 numpy` from the default PyPI index (resolved `polars 2.0.0`, `numpy 2.5.4`), then `scripts/release_smoke.py --expect 0.8.0` — every check (`version`, `install location`, `bundled plugin files`, `bundled plugin versions` x5 SKILL.md, `expression API`, `PyModel API`) passed. Final line: `SMOKE_OK 0.8.0 <venv-site-packages>/polars_statistics`.
  - Post-release smoke, Python 3.9: same sequence (resolved `polars 1.36.1`, `numpy 2.0.2`, same `cp39-abi3` wheel) — identical all-PASS result, `SMOKE_OK 0.8.0 <venv-site-packages>/polars_statistics`.
  - Final automated verify chain across the whole task echoed `RELEASED_OK`.
  - Smoke venvs (`target/release-gate/pypi3.12/`, `pypi3.9/`) removed after the logs were captured.

## Release Record

| Field | Value |
|---|---|
| Tag | [`v0.8.0`](https://github.com/DataZooDE/polars-statistics/releases/tag/v0.8.0) at merge commit `fd52d0781cd9487876c13638b38e9612cd9ee350` |
| GitHub release | **Published**, not draft, not prerelease — "v0.8.0 — Claude Code Skills Plugin & Upstream Refresh" |
| Main CI on M | Run [38821933548](https://github.com/DataZooDE/polars-statistics/actions/runs/38821933548) — `success` |
| Publish run (release event) | Run [38824647551](https://github.com/DataZooDE/polars-statistics/actions/runs/38824647551) — `success` |
| Build jobs (6/6) | sdist, ubuntu x86_64, ubuntu aarch64, macos x86_64, macos-14 arm64, windows x86_64 — all `success` |
| Publish to PyPI | `success` (OIDC trusted publishing, `id-token: write`) |
| Publish to TestPyPI | `skipped` (correct for a release event) |
| PyPI package | [polars-statistics 0.8.0](https://pypi.org/project/polars-statistics/0.8.0/) — 6 files (sdist + 5 wheels) |
| Smoke 3.12 | `SMOKE_OK 0.8.0` — all checks PASS |
| Smoke 3.9 | `SMOKE_OK 0.8.0` — all checks PASS |

## Task Commits

No per-task file-staging commits were made this plan — Task 1/3 produced only gitignored artifacts under `target/release-gate/`, and the GitHub/PyPI-side changes (tag, release, publish run, PyPI upload) are the plan's deliverable, not `task_commit_protocol` staged-file commits.

**Plan metadata:** committed alongside this SUMMARY — `docs(17-05): complete production publish plan`.

## Files Created/Modified

- `target/release-gate/publish-readiness.md` (gitignored) - Full pre-publish readiness report
- `target/release-gate/release-notes-0.8.0.md` (gitignored) - CHANGELOG `[0.8.0]` section, verbatim, used as the release notes
- `target/release-gate/publish-run-id.txt` (gitignored) - `38824647551`
- `target/release-gate/pypi-0.8.0.json` (gitignored) - PyPI JSON API response confirming 6 files at 0.8.0
- `target/release-gate/pypi-smoke-3.12.log`, `pypi-smoke-3.9.log` (gitignored) - Full post-release smoke output, both `SMOKE_OK 0.8.0`

No tracked repository source files were modified by this plan. The only repository-visible changes are the GitHub tag/release and the PyPI upload (all remote, outside git).

## Decisions Made

- Used the plan's proposed default release title verbatim: "v0.8.0 — Claude Code Skills Plugin & Upstream Refresh".
- Proceeded with the production publish strictly per the pre-given approval, conditioned on every Task 1 readiness check passing (it did) — per CONTEXT, earlier approvals (push, merge, TestPyPI) do not themselves authorize this step; the explicit "do it" plus the passing readiness gate together authorize it.
- Did not touch PR #47, the stale worktree, or `conda/meta.yaml` — all deferred to the follow-up list below per CONTEXT and the plan's own scope.

## Deviations from Plan

None — plan executed exactly as written. The main CI run referenced by 17-03 as "queued" completed with `success` during this plan's Task 1, exactly as 17-03 anticipated 17-05 would need to wait for.

## Issues Encountered

- `curl https://pypi.org/project/polars-statistics/0.8.0/` (the HTML page, for the plan's `<human-check>`) returned PyPI's anti-bot "Client Challenge" interstitial rather than rendered HTML. This does not affect the automated verification (which uses the PyPI JSON API and fully confirms version + file list — see D5), but the visual confirmation of the rendered README/release-notes page is left for the user to check manually at the URLs below. Not a release defect.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**REL-08 is now fully satisfied** — production PyPI publish via OIDC trusted publishing, with post-release smoke passing on Python 3.9 and 3.12 (import, version, bundled skills, expression + PyModel calls). This is the last plan in Phase 17 (5/5) and the last phase of the v0.8.0 milestone roadmap (13 → 17) — the milestone is ready to close.

**Links for the user to visually confirm:**
- PyPI: https://pypi.org/project/polars-statistics/0.8.0/
- GitHub release: https://github.com/DataZooDE/polars-statistics/releases/tag/v0.8.0

**Follow-ups (none performed in this plan, all reported per the plan's Task 3 instructions):**
1. Remove the stale local worktree `.claude/worktrees/agent-a88ea0e6ef2dbcd4c` and its branch (confirmed still present, an old Phase 16 attempt, uncommitted — do not merge).
2. Rebase open PR #47 (`ci/cost-guardrails`) onto the new main (confirmed still `OPEN`, untouched throughout this milestone per CONTEXT's instruction).
3. Delete the `release/v0.8.0` branch — it was merged with a merge commit (17-03), so per the plan's guidance it is safe to delete (would need to keep it had it been a squash merge).
4. Open a close-milestone docs PR for the post-merge `.planning/` commits (precedent: PR #46).
5. `conda/meta.yaml` still pins `version = "0.4.0"` (confirmed) — it is not part of the PyPI pipeline and was not touched by this milestone.
6. TestPyPI token status: the stored `TEST_PYPI_API_TOKEN` secret (created 2025-12-09) remains unverified/possibly stale since the 17-04 dry run was declined; consider migrating TestPyPI to OIDC trusted publishing in a future milestone (T-17-15 in 17-04's threat register, disposition "accept").

---
*Phase: 17-release-0-8-0*
*Completed: 2026-10-16*

## Self-Check: PASSED

- `target/release-gate/publish-readiness.md`, `release-notes-0.8.0.md`, `publish-run-id.txt`, `pypi-0.8.0.json`, `pypi-smoke-3.12.log`, `pypi-smoke-3.9.log` all found on disk.
- `git rev-list -n1 v0.8.0` == `fd52d0781cd9487876c13638b38e9612cd9ee350` — re-confirmed.
- `gh release view v0.8.0 --json tagName,isDraft,isPrerelease` == `v0.8.0 false false` — re-confirmed.
- `gh run view 38824647551 --json jobs` — 6/6 Build jobs `success`, `Publish to PyPI` `success`, `Publish to TestPyPI` `skipped` — re-confirmed.
- `curl https://pypi.org/pypi/polars-statistics/0.8.0/json` — `info.version=0.8.0`, 6 files matching all 6 required filename patterns — re-confirmed.
- Both smoke logs contain `SMOKE_OK 0.8.0 ` and end with `EXIT=0` — re-confirmed.
- No commits were expected for this plan's task work (all GitHub/PyPI-side, no tracked file changes); `git rev-list --count bdaa43d..HEAD` == `0` at SUMMARY-write time.
