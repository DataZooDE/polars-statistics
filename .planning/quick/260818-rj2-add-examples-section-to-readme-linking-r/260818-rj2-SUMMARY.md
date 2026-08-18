---
phase: quick-260818-rj2
plan: "01"
subsystem: documentation
status: complete
tags: [readme, examples, discoverability, issue-36]
requirements: [ISSUE-36]
dependency_graph:
  requires: []
  provides: [examples-section-in-readme]
  affects: [README.md]
tech_stack:
  added: []
  patterns: [markdown-tables]
key_files:
  modified:
    - README.md
decisions:
  - Descriptions for the 5 Python scripts copied verbatim from examples/README.md per plan spec
  - rust_wls.rs description written as brief accurate one-liner (not in examples/README.md tables)
  - Cookbook page titles derived from filenames with human-readable casing
metrics:
  duration: 5m
  completed: "2026-08-18"
  tasks: 1
  commits: 1
actuals:
  tokens: 1200
  tasks: 1
  commits: 1
---

# Phase quick-260818-rj2 Plan 01: Add Examples Section to README Summary

## One-liner

New `## Examples` section surfacing 6 runnable scripts and 11 cookbook pages directly from the main README, fixing issue #36 (input DataFrame shape discoverability).

## What Was Built

Inserted a new `## Examples` section into `README.md` immediately before the `## Documentation`
heading (was ~line 354, now ~line 354+33). The section contains:

1. An introductory sentence noting that every example shows the expected input DataFrame shape
   (directly addresses GitHub issue #36 — users could not tell what DataFrame structure each
   method expects).
2. A "Runnable examples" Markdown table linking all 6 files in `examples/`, with descriptions
   copied verbatim from `examples/README.md` for the 5 Python scripts, and a brief accurate
   one-liner for `rust_wls.rs`.
3. A "Cookbook (docs/examples/)" table linking all 11 docs/examples/ pages with human-readable
   titles derived from their filenames.

No other file was touched. CHANGELOG.md, Cargo.toml, and pyproject.toml were left in their
pre-existing uncommitted state — only `README.md` was staged and committed.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add ## Examples section to README.md | b5e8e16 | README.md |

## Verification Results

Plan automated check:
```
bash -c 'grep -q "^## Examples$" README.md && awk "..." README.md && for p in ...; do ...; done && echo OK'
```
Output: `OK`

Additional checks:
- `grep -c "^## Examples$" README.md` → `1` (exactly one heading)
- `## Examples` heading located before `## Documentation` heading (awk order check passed)
- All 6 `examples/` links present (01–05 Python scripts + rust_wls.rs)
- All 11 `docs/examples/` links present
- `git status --porcelain README.md` → ` M README.md` (modified, committed)
- CHANGELOG.md, Cargo.toml, pyproject.toml remained untouched (pre-existing M state preserved)

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None.

## Self-Check: PASSED

- README.md modified and committed at b5e8e16
- `## Examples` section present before `## Documentation`
- All required links verified by automated check (returned OK)
