---
phase: 17-release-0-8-0
plan: 04
subsystem: release-engineering
tags: [github-actions, testpypi, workflow-dispatch, decision-gate]

requires:
  - phase: 17-release-0-8-0
    provides: "17-03: PR #48 merged into GitHub main as merge commit fd52d0781cd9487876c13638b38e9612cd9ee350; main CI run 38821933548 queued for M"
provides:
  - "Recorded decline of the optional TestPyPI dry run for the v0.8.0 release flow"
  - "target/release-gate/testpypi-decision.txt = 'declined' (gitignored)"
  - "Confirmed no publish.yml workflow_dispatch run exists for origin/main tip (= merge commit M, fd52d07) — nothing dispatched before or after the decision"
affects: [17-05-publish]

actuals:
  tokens: 900
  tasks: 2
  commits: 0
  plan_head_before: 824f07d54047a934391fd792983339923c4cd30d

tech-stack:
  added: []
  patterns:
    - "Decline path for an optional checkpoint:decision task records a one-line outcome file under the gitignored target/release-gate/ scratch area, read by the next plan's readiness report — same pattern 17-03 used for merge-readiness.md"

key-files:
  created:
    - target/release-gate/testpypi-decision.txt (gitignored, contents: "declined")
  modified: []

key-decisions:
  - "User declined the TestPyPI dry run. Relayed via the orchestrator session, which had already shown the user the full release flow (push -> PR -> CI -> merge -> skip TestPyPI -> tag/release -> PyPI publish -> smoke on 3.12/3.9) and received \"do it\" as approval for that flow, including the explicit instruction to skip TestPyPI."
  - "Reasons for the decline, as given in the user's approved flow and the plan's own context: TestPyPI publishing authenticates via the stored TEST_PYPI_API_TOKEN secret (created 2025-12-09, last used for TestPyPI's 0.1.0 listing) rather than OIDC trusted publishing, so the rehearsal risks failing on a stale token even if every build succeeds; and the production publish in 17-05 builds and proves all 5 platform wheels (Linux x86_64/aarch64, macOS x86_64/arm64, Windows x86_64) anyway via the same publish.yml matrix, so the dry run's main value (proving the aarch64/macOS cross-builds) is redundant with the irreversible step that follows regardless."
  - "No `gh workflow run` or any other GitHub write was issued before or after recording the decline, per the plan's Task 1 instruction."

requirements-completed: []

coverage:
  - id: D1
    description: "User's explicit decline of the TestPyPI dry run recorded before any workflow dispatch"
    requirement: "REL-08"
    verification:
      - kind: manual_procedural
        ref: "User's literal answers in the orchestrator session: \"go and the release is approved.\" then, after being shown the merge -> skip TestPyPI -> tag/release -> PyPI flow, \"do it\" — relayed into this execution as the pre-answered Task 1 checkpoint, choice = decline"
        status: pass
    human_judgment: false
  - id: D2
    description: "No publish.yml workflow_dispatch run exists for the main commit (origin/main tip == merge commit M, fd52d07); decision recorded in target/release-gate/testpypi-decision.txt"
    requirement: "REL-08"
    verification:
      - kind: other
        ref: "Task 2 automated verify chain: `gh run list --workflow publish.yml --event workflow_dispatch --commit $OM --json databaseId --jq length` == 0, echoed DECLINE_OK"
        status: pass
    human_judgment: false

duration: 5min
completed: 2026-10-16
status: complete
---

# Phase 17 Plan 04: TestPyPI Dry Run Decision — Declined Summary

**User declined the optional TestPyPI dry run (stale stored-token auth risk, redundant with the production build matrix); recorded as `declined` with zero GitHub writes, clearing the way for 17-05's production publish.**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-10-16T00:00:00Z (approx.)
- **Completed:** 2026-10-16T00:05:00Z (approx.)
- **Tasks:** 2 (Task 1 checkpoint:decision — pre-answered via orchestrator, Task 2 auto)
- **Files modified:** 0 tracked repository files (the only artifact, `target/release-gate/testpypi-decision.txt`, is gitignored under `/target/`)

## Accomplishments

- **Task 1 (checkpoint:decision — gate=blocking-human):** The user's choice was obtained in the orchestrator session before this execution began, not invented here. The orchestrator had shown the user the complete release flow — push `release/v0.8.0` + PR, CI, merge commit, **skip TestPyPI**, tag `v0.8.0` + GitHub release, PyPI publish, smoke on Python 3.12/3.9 — and the user answered "do it", with the explicit instruction to skip TestPyPI carried in that approval. No `gh workflow run` or other GitHub write happened before recording this answer.
- **Task 2 (auto — execute the decline path):** Wrote the single line `declined` to `target/release-gate/testpypi-decision.txt`. Confirmed via `git fetch origin` that `origin/main` is still at the merge commit M (`fd52d0781cd9487876c13638b38e9612cd9ee350`) recorded by 17-03 — nothing new landed on main since the merge. Confirmed zero `publish.yml` `workflow_dispatch` runs exist for that commit (`gh run list --workflow publish.yml --event workflow_dispatch --commit fd52d07 --json databaseId --jq length` returned `0`). Nothing was dispatched.

## Decline Record

| Field | Value |
|---|---|
| Decision | **Decline** the TestPyPI dry run |
| Approved via | Orchestrator session: "go and the release is approved." / "do it" (after being shown the full flow including "skip TestPyPI") |
| origin/main tip at decline time | `fd52d0781cd9487876c13638b38e9612cd9ee350` (= merge commit M from 17-03) |
| publish.yml workflow_dispatch runs for that commit | 0 (confirmed via `gh run list`) |
| Reasons recorded | (1) TestPyPI auth uses the stored `TEST_PYPI_API_TOKEN` secret, not OIDC, and that token was created 2025-12-09 with TestPyPI currently listing only 0.1.0 — a real risk of auth failure even on a clean build; (2) the production publish in 17-05 builds and uploads all 5 platform wheels via the same matrix anyway, so the dry run's main value (proving the aarch64/macOS cross-builds) is exercised regardless |

## Task Commits

No per-task file-staging commits were made this plan — the only artifact produced (`target/release-gate/testpypi-decision.txt`) is gitignored. No GitHub writes (workflow dispatches, tags, releases) occurred.

**Plan metadata:** committed alongside this SUMMARY — `docs(17-04): record TestPyPI dry run decline`.

## Files Created/Modified

- `target/release-gate/testpypi-decision.txt` (gitignored) - Single line: `declined`

## Decisions Made

- Declined the TestPyPI dry run per the user's explicit, pre-given instruction in the orchestrator session. See Decline Record above for the full reasoning (stale stored-token auth risk; redundant with the production build matrix in 17-05).

## Deviations from Plan

None - plan executed exactly as written. Task 1's checkpoint:decision was answered by the user in the orchestrator session rather than mid-execution here; this mirrors the identical pattern used in 17-03's Task 2 (merge-method checkpoint), which this plan's `<context>` directly references.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `target/release-gate/testpypi-decision.txt` = `declined`, ready for 17-05's pre-publish readiness report to read and surface the TestPyPI outcome.
- origin/main is unchanged at merge commit `fd52d0781cd9487876c13638b38e9612cd9ee350` — 17-05 tags this exact commit.
- No workflow runs, tags, releases, or PyPI uploads were created by this plan.
- 17-05 proceeds directly to its pre-publish readiness check (main CI on M, version alignment, tag/release/PyPI absence, OIDC proof) followed by the production publish checkpoint.

---
*Phase: 17-release-0-8-0*
*Completed: 2026-10-16*

## Self-Check: PASSED

- `target/release-gate/testpypi-decision.txt` found on disk, contents equal `declined`.
- `gh run list --workflow publish.yml --event workflow_dispatch --commit fd52d0781cd9487876c13638b38e9612cd9ee350 --json databaseId --jq length` == `0` — re-confirmed.
- No commits were expected for this plan's task work (gitignored artifact only); `git rev-list --count 824f07d..HEAD` == `0` at SUMMARY-write time, consistent with "declined, no dispatch" outcome.
