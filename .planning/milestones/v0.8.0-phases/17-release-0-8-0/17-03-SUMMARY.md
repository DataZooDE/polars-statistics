---
phase: 17-release-0-8-0
plan: 03
subsystem: release-engineering
tags: [github-merge, gh-cli, merge-commit, version-alignment, plugin-manifest, ci]

requires:
  - phase: 17-release-0-8-0
    provides: "17-02: open PR #48 (release/v0.8.0 -> main), head 869622f, all 18 CI jobs green, PR #47 probed clean"
provides:
  - "PR #48 merged into GitHub main as merge commit fd52d0781cd9487876c13638b38e9612cd9ee350"
  - "Merged tree verified: all 18 version sources read 0.8.0 (scripts/check_version_alignment.py)"
  - "Plugin marketplace manifest (.claude-plugin/marketplace.json) confirmed live on GitHub main at the merge commit"
  - "Local main synced with origin/main via a non-destructive git merge (no rebase, no force, no reset)"
  - "Main CI run (queued) and docs.yml GitHub Pages deploy (succeeded) recorded for the merge commit"
affects: [17-04-testpypi, 17-05-publish]

actuals:
  tokens: 2200
  tasks: 3
  commits: 3
  plan_head_before: b905cf0f4cc2bc5424259bcef93ae472de054ba3

tech-stack:
  added: []
  patterns:
    - "gh pr merge --merge --match-head-commit <sha> as a tamper-guard: GitHub rejects the merge if the head moved since the last verification"
    - "git archive of a merge commit extracted to a scratch dir, then scripts/check_version_alignment.py run against that extracted tree (not the live working tree) to prove the merged commit itself is aligned, independent of local state"

key-files:
  created:
    - target/release-gate/merge-readiness.md (gitignored)
    - target/release-gate/main-tree/ (gitignored, extracted merged tree)
    - target/release-gate/main-tree.sha (gitignored)
  modified: []

key-decisions:
  - "Local sync used git merge --no-edit origin/main (not --ff-only) because local main carried one post-push planning commit (b905cf0, 17-02's bookkeeping) that origin/main did not have. The merge was clean (both sides carry identical release content), producing merge commit 0eec6a2 with zero conflicts."
  - "Did not merge, comment on, or modify PR #47 (ci/cost-guardrails) — left open per CONTEXT's explicit instruction and 17-02's probe finding (clean auto-merge expected after this release)."
  - "Did not wait for the main CI run or the docs deploy to finish inside this plan — the plan's contract is to confirm a run exists (17-05 waits for main CI success before tagging). The docs.yml run (GitHub Pages deploy) happened to finish during this plan's execution with conclusion=success; CI (ci.yml) was still queued when last checked."

requirements-completed: []

coverage:
  - id: D1
    description: "Pre-merge re-check confirmed: PR #48 head unchanged (869622fe5865da3eefea950b4d711a64c0c8f0f1), all 18 CI-workflow jobs pass, mergeStateStatus CLEAN/mergeable MERGEABLE, origin/main not moved past the PR base"
    requirement: "REL-08"
    verification:
      - kind: other
        ref: "Task 1 automated verify chain: echoed MERGE_READY"
        status: pass
    human_judgment: false
  - id: D2
    description: "User's explicit merge-method choice (merge commit) obtained via the orchestrator session before any GitHub write"
    requirement: "REL-08"
    verification:
      - kind: manual_procedural
        ref: "User's literal answers in the orchestrator session: \"go and the release is approved.\" then, after being shown the merge -> skip TestPyPI -> tag/release -> PyPI flow, \"do it\" — relayed into this execution as the pre-answered Task 2 checkpoint, merge method = merge commit"
        status: pass
    human_judgment: false
  - id: D3
    description: "PR #48 merged into GitHub main with a merge commit, guarded by --match-head-commit; merge commit fd52d07 is an ancestor of origin/main"
    requirement: "REL-08"
    verification:
      - kind: other
        ref: "gh pr merge 48 --merge --match-head-commit 869622fe5865da3eefea950b4d711a64c0c8f0f1 (exit 0); gh pr view 48 state=MERGED, mergeCommit.oid=fd52d0781cd9487876c13638b38e9612cd9ee350; git merge-base --is-ancestor fd52d07 origin/main"
        status: pass
    human_judgment: false
  - id: D4
    description: "Merged tree verified: all 18 version sources read 0.8.0 via an extracted copy of the merge commit; plugin marketplace.json confirmed live on GitHub main at the merge commit; a main CI run exists for the merge commit"
    requirement: "REL-08"
    verification:
      - kind: other
        ref: "Task 3 automated verify chain: echoed MERGED_OK (version check RESULT: PASS, marketplace.json present via gh api, >=1 ci.yml run for M)"
        status: pass
    human_judgment: false
  - id: D5
    description: "Local main synced to contain origin/main via a non-destructive merge (git merge --no-edit, no force/reset/rebase), with no history rewrite"
    requirement: "REL-08"
    verification:
      - kind: other
        ref: "git merge --no-edit origin/main (clean, merge commit 0eec6a2); git merge-base --is-ancestor origin/main main confirmed afterward"
        status: pass
    human_judgment: false

duration: 20min
completed: 2026-10-16
status: complete
---

# Phase 17 Plan 03: Pre-merge Re-check, Merge PR #48, Verify Merged Commit, Sync Local Main Summary

**PR #48 merged into GitHub main as a merge commit (`fd52d07`) after a clean pre-merge re-check; the merged tree checks out at 0.8.0 across all 18 version sources, the plugin marketplace manifest is live on main, and local main now contains origin/main via a non-destructive merge — nothing rebased, reset, or force-pushed.**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-10-15T23:08:00Z (approx.)
- **Completed:** 2026-10-15T23:12:00Z (approx.)
- **Tasks:** 3 (Task 1 auto, Task 2 checkpoint:decision — pre-answered via orchestrator, Task 3 auto)
- **Files modified:** 0 tracked repository files beyond the merge commits themselves (all plan-produced artifacts are gitignored under `target/release-gate/`)

## Accomplishments

- **Task 1 (pre-merge re-check, read-only):** Re-derived PR #48 from GitHub (not memory); head SHA `869622fe5865da3eefea950b4d711a64c0c8f0f1` matched exactly what 17-02-SUMMARY recorded — unchanged since the PR opened. Confirmed 18/18 `CI`-workflow jobs in bucket `pass`, zero non-pass, `codecov/patch` pass. Confirmed `mergeable=MERGEABLE`, `mergeStateStatus=CLEAN`. Fetched origin and confirmed `origin/main` (`431306d` — the v0.7.0 milestone-close commit) is still an ancestor of the PR head, and the head is an ancestor of local main — nothing landed on GitHub main since the PR opened. Wrote `target/release-gate/merge-readiness.md` with the full readiness report, the PR #47 recommendation (leave open, rebase after release), the expected merge side effects (main CI, docs deploy since `docs/**` changed — 26 files, plugin marketplace going live), and the reminder that merging does not publish to PyPI.
- **Task 2 (checkpoint:decision — merge method):** Pre-answered by the user via the orchestrator session: "go and the release is approved." / "do it", after being shown the full flow (merge with a merge commit, not squash → skip TestPyPI → tag/release → PyPI). Recorded choice: **merge commit** (option `merge-commit`). No GitHub write occurred before this answer.
- **Task 3 (merge, verify, sync):**
  - Re-confirmed immediately before acting: head still `869622f...`, still 18/18 CI green, still MERGEABLE/CLEAN.
  - `gh pr merge 48 --merge --match-head-commit 869622fe5865da3eefea950b4d711a64c0c8f0f1` — exit 0. The `--match-head-commit` guard means GitHub itself would have rejected the merge had the head moved since Task 1's check (T-17-09 mitigation).
  - Merge commit **M = `fd52d0781cd9487876c13638b38e9612cd9ee350`**, confirmed an ancestor of `origin/main` after `git fetch origin`.
  - Extracted M via `git archive` into `target/release-gate/main-tree/`; ran `scripts/check_version_alignment.py --expect 0.8.0` against the extracted copy — **all 18 version sources read 0.8.0**, `RESULT: PASS`.
  - `.claude-plugin/marketplace.json` confirmed present on GitHub main at M via `gh api repos/.../contents/...?ref=M`.
  - Local sync: `git merge --ff-only origin/main` was refused (local main carried 17-02's bookkeeping commit `b905cf0` that origin/main did not have — exactly the expected "planning commits end up on top" case). Fell back to `git merge --no-edit origin/main` — clean, zero conflicts, merge commit `0eec6a2`. Confirmed `origin/main` is now an ancestor of local `main`.
  - Main CI run for M: one run, `databaseId 38821933548`, status `queued` at last check (not awaited here — 17-05 waits for it before tagging).
  - Docs deploy (`docs.yml`) run for M: `databaseId 38821933567`, **completed, conclusion `success`** — the GitHub Pages docs site has redeployed for 0.8.0.
  - Re-confirmed PR #47 (`ci/cost-guardrails`) untouched: still `OPEN`, still exactly 1 pre-existing comment.

## Merge Record

| Field | Value |
|---|---|
| PR | [#48](https://github.com/DataZooDE/polars-statistics/pull/48) `release/v0.8.0 -> main` |
| Merge method | **Merge commit** (`--merge`), user-approved via orchestrator session |
| Verified head (guard) | `869622fe5865da3eefea950b4d711a64c0c8f0f1` |
| Merge commit (M) | `fd52d0781cd9487876c13638b38e9612cd9ee350` |
| Merged at | 2026-10-15T23:10:15Z |
| Main CI run | [38821933548](https://github.com/DataZooDE/polars-statistics/actions/runs/38821933548) — `queued` at last check (not awaited; 17-05 waits before tagging) |
| Docs deploy run | [38821933567](https://github.com/DataZooDE/polars-statistics/actions/runs/38821933567) — `completed` / `success` |
| Version alignment on M | `RESULT: PASS` — 18/18 sources at 0.8.0 |
| Plugin manifest on M | `.claude-plugin/marketplace.json` present (confirmed via `gh api`) |
| Local sync commit | `0eec6a2` (`git merge --no-edit origin/main`, clean, 0 conflicts) |

## Task Commits

No per-task file-staging commits were made this plan — Task 1 produced only gitignored artifacts under `target/release-gate/`, and Task 3's "commits" are the GitHub-side merge commit (`fd52d07`, created by `gh pr merge`) and the local sync merge commit (`0eec6a2`, created by `git merge --no-edit origin/main`). Both are expected git operations specified by the plan, not `task_commit_protocol` staged-file commits.

**Plan metadata:** committed alongside this SUMMARY (see git log after this commit) — `docs(17-03): complete merge + verify + sync plan`.

## Files Created/Modified

- `target/release-gate/merge-readiness.md` (gitignored) - Pre-merge facts: head SHA, 18/18 CI, MERGEABLE/CLEAN, base-not-moved, PR #47 recommendation, expected merge side effects
- `target/release-gate/main-tree/` (gitignored) - Extracted copy of merge commit M, used only for the version-alignment check
- `target/release-gate/main-tree.sha` (gitignored) - Records M (`fd52d0781cd9487876c13638b38e9612cd9ee350`)

No tracked repository source files were modified by this plan. The only repository-visible changes are the two merge commits (`fd52d07` on GitHub main brought in by the PR, `0eec6a2` on local main from the sync) and this SUMMARY + STATE/ROADMAP bookkeeping committed at the end.

## Decisions Made

- Local sync used a clean `git merge --no-edit origin/main` (not `--ff-only`, which was correctly refused) because local main carried one planning commit (`b905cf0`, 17-02's bookkeeping) ahead of `origin/main`. Per the plan's explicit instruction, both sides carry identical release content so the merge was conflict-free.
- PR #47 (`ci/cost-guardrails`) was left entirely untouched — not merged, commented on, or modified. Per CONTEXT and 17-02's finding, it is expected to auto-merge cleanly on top of this release later.
- Did not wait for the main CI run (`ci.yml`) to finish inside this plan. That wait is explicitly 17-05's responsibility, immediately before tagging.

## Deviations from Plan

None — plan executed exactly as written. `git merge --ff-only` being refused and falling through to `git merge --no-edit` is the exact branch the plan itself specifies for this case (local main carries post-push planning commits), not a deviation.

## Note: `.planning/milestone.lock` on GitHub main

`.planning/milestone.lock` (a transient GSD session-lock file, not meant to persist) was committed as part of 17-01's and 17-02's plan-metadata commits (first added in `67a4cd1`, updated in `869622f`) and was therefore included in the PR #48 branch and is now present on GitHub `main` at merge commit M. The plan's rules explicitly forbid rewriting history to remove it — it is left in place and flagged here for the user's awareness. It contains only a session id, PID, and timestamp (no secrets).

## Issues Encountered

None. Pre-merge re-check, the merge, the merged-tree verification, and the local sync all succeeded on the first attempt, with zero conflicts.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- GitHub `main` is now at merge commit `fd52d0781cd9487876c13638b38e9612cd9ee350`, verified: 18/18 version sources at 0.8.0, plugin marketplace manifest live, docs site redeployed successfully.
- Main CI run `38821933548` was queued (not yet resolved) when last checked — **17-05 must wait for this run (or its successor on the same commit) to succeed before tagging v0.8.0.**
- Local `main` contains `origin/main` (merge commit `0eec6a2`); nothing was pushed from local main back to origin in this plan (not requested by the plan).
- PR #47 (`ci/cost-guardrails`) remains open and untouched; recommend rebasing it after the release per CONTEXT.
- `.planning/milestone.lock` is present on GitHub main — see the note above; no action required, just awareness for the user.
- Plan 17-04 (optional TestPyPI dry run) and 17-05 (production publish) can proceed with M as the tag target. Per the user's prior instruction in this session, 17-04 (TestPyPI) is to be **skipped**, proceeding directly to 17-05's tag/release/PyPI-publish flow.

---
*Phase: 17-release-0-8-0*
*Completed: 2026-10-16*

## Self-Check: PASSED

- `target/release-gate/merge-readiness.md` found on disk.
- `target/release-gate/main-tree.sha` found on disk, contents equal `fd52d0781cd9487876c13638b38e9612cd9ee350`.
- PR #48 confirmed `state: MERGED`, `mergeCommit.oid: fd52d0781cd9487876c13638b38e9612cd9ee350`.
- `git merge-base --is-ancestor fd52d0781cd9487876c13638b38e9612cd9ee350 origin/main` — passes.
- `git merge-base --is-ancestor origin/main main` — passes.
- Re-ran `python3 target/release-gate/main-tree/scripts/check_version_alignment.py --expect 0.8.0` → `RESULT: PASS`.
