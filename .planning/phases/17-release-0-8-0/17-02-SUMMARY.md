---
phase: 17-release-0-8-0
plan: 02
subsystem: release-engineering
tags: [github-pr, ci-matrix, release-branch, gh-cli, merge-tree-probe]

requires:
  - phase: 17-release-0-8-0
    provides: "17-01: 16/16 CI-equivalent local gate logs bound to gated HEAD 125f738, release wheel+sdist built and smoke-verified on 3.9/3.12"
provides:
  - "origin/release/v0.8.0 (= local main HEAD 869622f), pushed via a single fast-forward refspec, no force, no tags"
  - "Open PR #48 (release/v0.8.0 -> main) with a CHANGELOG-sourced body ending in the Claude Code attribution line"
  - "All 18 CI-workflow jobs green on the PR head (run 38819507686), plus codecov/patch pass — zero CI-fix commits needed"
  - "PR #47 (ci/cost-guardrails) merge-tree probe: clean auto-merge (exit 0, no conflicts) — read-only, untouched"
affects: [17-03-merge, 17-04-testpypi, 17-05-publish]

actuals:
  tokens: 2800
  tasks: 3
  commits: 0
  plan_head_before: 869622fe5865da3eefea950b4d711a64c0c8f0f1

tech-stack:
  added: []
  patterns:
    - "git merge-tree --write-tree --name-only as a read-only pre-merge conflict probe against an unrelated open PR branch, without checking it out or touching it"
    - "git push origin main:refs/heads/<branch> to create a remote branch from local main without ever creating or checking out a local branch of that name — keeps local main the single working branch for any later fast-forward fix commit"

key-files:
  created:
    - target/release-gate/pr-body.md (gitignored, under target/)
    - target/release-gate/post-gate-paths.txt (gitignored)
    - target/release-gate/push-range.diff (gitignored)
  modified: []

key-decisions:
  - "PR #47's expected ci.yml conflict, per CONTEXT's prediction, did NOT materialize: git merge-tree --write-tree --name-only main origin/ci/cost-guardrails exited 0 (clean auto-merge, no conflicted paths). Git's ort strategy combines #47's concurrency: block and per-job timeout-minutes: additions with this milestone's ruff/rust_api/wheel-bundling-drift step insertions because the two sets of edits land on non-overlapping lines within each job block. Verified by diffing the probed merge tree's ci.yml against current main: it contains exactly #47's 6 additions (concurrency block + 5 timeout-minutes lines) and none of this milestone's steps were removed. Recorded as a finding, not treated as a conflict to resolve — #47 was not modified, commented on, or merged."
  - "Did not mark REL-08 complete in REQUIREMENTS.md. The plan's frontmatter lists requirements: [REL-08], but REL-08 explicitly requires production PyPI publish + post-release smoke, which are 17-04/17-05's scope, not this plan's. Marking it complete here would be premature; left as Pending."
  - "Zero CI-fix commits were needed — all 18 jobs passed on the first CI run (38819507686), so the plan's Task 3 fix-iteration protocol (fast-forward fix commits, 3-attempt limit) was never invoked."

requirements-completed: []

coverage:
  - id: D1
    description: "Pre-push readiness proven locally: 16/16 gate logs bound to gated SHA 125f738 with EXIT=0, gated SHA is an ancestor of HEAD with only .planning/ changes after it, working tree clean, .gsd/ and .claude/worktrees untracked, origin/main is an ancestor of local main, no release/v0.8.0 branch or v0.8.0 tag pre-existed on origin, push-range secret scan clean, PR body drafted ending in the attribution line"
    requirement: "REL-08"
    verification:
      - kind: other
        ref: "Task 1 automated verify chain: echoed PREPUSH_OK"
        status: pass
    human_judgment: false
  - id: D2
    description: "User approved publishing the release candidate to GitHub (push release/v0.8.0 + open PR) before any outward write ran"
    requirement: "REL-08"
    verification:
      - kind: manual_procedural
        ref: "User's literal answers in the orchestrator session: \"go and the release is approved.\" then, after being shown the 5-plan release flow, \"do it\" — relayed into this execution as pre-answered Task 2 checkpoint, option approve-push"
        status: pass
    human_judgment: false
  - id: D3
    description: "release/v0.8.0 pushed to origin from local main via a plain fast-forward refspec (no force, no push to main, no tags); PR #48 opened release/v0.8.0 -> main with body ending in the attribution line"
    requirement: "REL-08"
    verification:
      - kind: other
        ref: "git push origin main:refs/heads/release/v0.8.0 (new branch, no +refspec); gh pr create --base main --head release/v0.8.0 -> https://github.com/DataZooDE/polars-statistics/pull/48"
        status: pass
    human_judgment: false
  - id: D4
    description: "All 18 CI-workflow jobs (Lint, Type Stubs, Coverage, 3 Build wheels, 12 Python Tests across ubuntu/macos/windows x 3.9-3.12) pass on the PR head, with zero CI-fix commits required"
    requirement: "REL-08"
    verification:
      - kind: other
        ref: "gh pr checks 48 --json name,workflow,bucket -- 18/18 workflow==CI jobs bucket==pass; run https://github.com/DataZooDE/polars-statistics/actions/runs/38819507686 conclusion=success"
        status: pass
    human_judgment: false
  - id: D5
    description: "PR #47 (ci/cost-guardrails) probed read-only via git merge-tree and reported to the user; not modified, commented on, or merged by this plan"
    requirement: "REL-08"
    verification:
      - kind: other
        ref: "git merge-tree --write-tree --name-only main origin/ci/cost-guardrails exit 0 (clean); gh pr view 47 --json state,comments shows state=OPEN, single pre-existing codecov bot comment dated 2026-10-03 (before this session)"
        status: pass
    human_judgment: false
  - id: D6
    description: "Stale local worktree .claude/worktrees/agent-a88ea0e6ef2dbcd4c / branch worktree-agent-a88ea0e6ef2dbcd4c was not pushed; reported for user cleanup"
    requirement: "REL-08"
    verification:
      - kind: other
        ref: "git ls-remote --heads origin worktree-agent-a88ea0e6ef2dbcd4c (empty, confirmed both before and after push)"
        status: pass
    human_judgment: false

duration: 35min
completed: 2026-10-16
status: complete
---

# Phase 17 Plan 02: Push Release Candidate to GitHub and Drive CI Green Summary

**Pushed `release/v0.8.0` (253 commits / 340 files ahead of `origin/main`) from local main as a plain fast-forward, opened PR #48 into `main`, and all 18 CI-workflow jobs passed on the first run — no fix commits needed; PR #47's predicted `ci.yml` conflict turned out to be a clean git auto-merge instead.**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-10-15T22:50:00Z (approx.)
- **Completed:** 2026-10-15T23:06:00Z
- **Tasks:** 3 (Task 1 auto, Task 2 checkpoint:decision, Task 3 auto)
- **Files modified:** 0 tracked repository files (all outputs under gitignored `target/release-gate/`)

## Accomplishments

- **Task 1 (pre-push readiness, no outward writes):** Verified all 16 `target/release-gate/*.log` gate files are bound to gated SHA `125f738b3ab165c8cb67d9a6f6602eb025529dcd` with `EXIT=0`; confirmed that SHA is an ancestor of HEAD `869622f` and every post-gate commit touches only `.planning/`; confirmed clean working tree and that `.gsd/` / `.claude/worktrees` are untracked; fetched `origin` and confirmed `origin/main` is an ancestor of local `main`; confirmed no `release/v0.8.0` branch or `v0.8.0` tag pre-existed on origin; scanned the full 253-commit push range for credential-shaped strings (clean); drafted `target/release-gate/pr-body.md` ending in the required attribution line. Full automated verify chain printed `PREPUSH_OK`.
- **PR #47 probe:** `git merge-tree --write-tree --name-only main origin/ci/cost-guardrails` exited **0** — a clean auto-merge, not the conflict CONTEXT predicted. Confirmed by diffing the probed merge tree's `ci.yml` against current `main`: it contains exactly #47's `concurrency:` block and 5 `timeout-minutes:` lines, with none of this milestone's ruff/rust_api/wheel-bundling-drift steps removed. #47 was fetched read-only and never modified, commented on, or merged.
- **Task 2 (checkpoint:decision — push + PR):** Pre-answered by the user via the orchestrator session ("go and the release is approved." / "do it", per the 5-plan release flow they were shown). Recorded as `approve-push`. No outward write occurred before this answer.
- **Task 3 (push + PR + CI):**
  - `git push origin main:refs/heads/release/v0.8.0` — new branch created, no `+` refspec, no force, never touched `origin/main`.
  - `gh pr create --base main --head release/v0.8.0 --body-file target/release-gate/pr-body.md` → **PR #48**, head `869622fe5865da3eefea950b4d711a64c0c8f0f1` (= local main HEAD exactly, an ancestor of main by construction).
  - Watched CI via two `timeout 570 gh pr checks 48 --watch --interval 60` calls (first returned exit 124, still running; second completed with exit 0). Total matrix wall time ≈ 17 min (longest job: `Build wheels (windows-latest)` at 16m48s).
  - **All 18 CI-workflow jobs passed on the first run** — zero fix-iteration commits needed.

## CI Results

**PR:** [#48](https://github.com/DataZooDE/polars-statistics/pull/48) `release/v0.8.0 -> main`
**Head SHA:** `869622fe5865da3eefea950b4d711a64c0c8f0f1`
**CI run:** [38819507686](https://github.com/DataZooDE/polars-statistics/actions/runs/38819507686) — workflow `CI`, conclusion `success`

| Job | Result | Duration |
|-----|--------|----------|
| Lint | pass | 5m3s |
| Type Stubs | pass | 6m32s |
| Coverage | pass | 9m15s |
| Build wheels (ubuntu-latest) | pass | 9m36s |
| Build wheels (macos-latest) | pass | 14m1s |
| Build wheels (windows-latest) | pass | 16m48s |
| Python Tests (3.9, ubuntu-latest) | pass | 9m15s |
| Python Tests (3.10, ubuntu-latest) | pass | 8m53s |
| Python Tests (3.11, ubuntu-latest) | pass | 9m0s |
| Python Tests (3.12, ubuntu-latest) | pass | 8m59s |
| Python Tests (3.9, macos-latest) | pass | 15m18s |
| Python Tests (3.10, macos-latest) | pass | 14m32s |
| Python Tests (3.11, macos-latest) | pass | 12m40s |
| Python Tests (3.12, macos-latest) | pass | 15m9s |
| Python Tests (3.9, windows-latest) | pass | 2m5s |
| Python Tests (3.10, windows-latest) | pass | 2m14s |
| Python Tests (3.11, windows-latest) | pass | 1m25s |
| Python Tests (3.12, windows-latest) | pass | 1m30s |

**External status:** `codecov/patch` — pass.

**Lint duration note for PR #47's proposed 15-minute lint timeout:** Lint completed in 5m3s on this PR's head (which already includes the ruff + drift-guard + rust_api steps #47's diff retains timeout-minutes for) — well inside #47's proposed 15-minute budget.

**PR #47 (`ci/cost-guardrails`) disposition:** `gh pr view 47` shows `state: OPEN`, exactly one comment (codecov bot, dated 2026-10-03 — predates this session), confirming no modification, comment, or merge occurred against it from this plan.

**Stale worktree note:** `.claude/worktrees/agent-a88ea0e6ef2dbcd4c` (branch `worktree-agent-a88ea0e6ef2dbcd4c`, an uncommitted old Phase 16 attempt) was never pushed — `git ls-remote --heads origin worktree-agent-a88ea0e6ef2dbcd4c` returned empty both before and after the push. User should clean this up locally at their convenience; it is out of scope for this plan.

## Task Commits

No code-task commits were made this plan — Task 1 only produced gitignored artifacts under `target/release-gate/` (no repo files to stage), Task 2 was a pre-answered checkpoint, and Task 3's push/PR/CI-watch required no fix commits since every CI job passed on the first run.

**Plan metadata:** committed alongside this SUMMARY (see git log after this commit) — `docs(17-02): complete push + PR + CI plan`.

## Files Created/Modified

- `target/release-gate/pr-body.md` (gitignored) - Drafted PR body: summary, local gate evidence table, reviewer notes (merge-commit recommendation, #47 probe result), attribution line
- `target/release-gate/post-gate-paths.txt` (gitignored) - `git diff --name-only GATED HEAD` output, confirms post-gate commits are `.planning/`-only
- `target/release-gate/push-range.diff` (gitignored) - Full `origin/main..main` diff, scanned for credential-shaped strings (clean)

## Decisions Made

- PR #47's predicted `ci.yml` conflict did not occur — documented in `key-decisions` above; treated as a finding to report, not a conflict needing resolution.
- Did not flip REL-08 to complete in REQUIREMENTS.md — it explicitly requires the production PyPI publish + post-release smoke check, which belong to plans 17-04/17-05.
- Recommended merge method in the PR body is a merge commit (not squash/rebase), to preserve individual commit hashes the `.planning/` SUMMARY files cite and keep local `main` in sync with `origin/main` post-merge without a rebase.

## Deviations from Plan

None - plan executed exactly as written. All 18 CI jobs passed on the first run, so the plan's fix-iteration protocol (diagnose → fix commit → re-gate → re-push → re-watch, capped at 3 attempts) was never invoked.

## Issues Encountered

None. Pre-push readiness checks, the push, the PR creation, and the full CI matrix all succeeded on the first attempt.

## User Setup Required

None - no external service configuration required. (Reminder: the stale local worktree noted above is the user's own cleanup item, not a service configuration.)

## Next Phase Readiness

- PR #48 (`release/v0.8.0 -> main`) is open with all 18 CI jobs green and `codecov/patch` passing. Head SHA `869622fe5865da3eefea950b4d711a64c0c8f0f1` is an ancestor of local `main`.
- Nothing has been merged, tagged, or published — plan 17-03 (merge checkpoint) can proceed with this PR as its input.
- PR #47 (`ci/cost-guardrails`) remains open, untouched, and — per the merge-tree probe — will auto-merge cleanly on top of `main` after this PR lands; no rebase conflict is expected, contrary to the CONTEXT's prediction.
- Stale local worktree `.claude/worktrees/agent-a88ea0e6ef2dbcd4c` / branch `worktree-agent-a88ea0e6ef2dbcd4c` still needs the user's manual cleanup; confirmed absent from origin.

---
*Phase: 17-release-0-8-0*
*Completed: 2026-10-16*

## Self-Check: PASSED

- `.planning/phases/17-release-0-8-0/17-02-SUMMARY.md` found on disk.
- `target/release-gate/pr-body.md` found on disk.
- PR #48 confirmed live: `state: OPEN`, `url: https://github.com/DataZooDE/polars-statistics/pull/48`.
- Re-ran `gh pr checks 48 --json name,workflow,bucket`: 18/18 `workflow==CI` jobs in bucket `pass`; `codecov/patch` pass.
- `git ls-remote --heads origin worktree-agent-a88ea0e6ef2dbcd4c` confirmed empty (stale worktree not pushed).
