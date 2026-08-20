---
phase: 5
slug: documentation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-12
---

# Phase 5 — Validation Strategy

> Documentation phase — validation is doc-presence + example-runnability checks.

## Test Infrastructure
| Property | Value |
|----------|-------|
| Framework | grep/doc audit + `python -c` docstring-example spot checks + `mkdocs build` |
| Quick run command | grep for new symbols in docstrings/mkdocs/CHANGELOG |
| Full suite command | `mkdocs build` (no broken refs) + spot-run docstring examples |

## Sampling Rate
- After each doc task: the documented symbols exist in the source and render.
- Before verify: DOCS-01..04 all satisfied (docstrings w/ runnable examples, mkdocs lists new API, Rust /// present, CHANGELOG has 0.6.0).

## Wave 0 Requirements
- Existing docs infra (mkdocs, CHANGELOG) covers this phase; no new framework.

## Validation Sign-Off
- [ ] Every new function/class has a docstring with a runnable example (DOCS-01)
- [ ] mkdocs API pages list all new API; `mkdocs build` clean (DOCS-02)
- [ ] Rust /// doc comments on all new public wrapper fns (DOCS-03)
- [ ] CHANGELOG has a 0.6.0 section covering the new API (DOCS-04)
- [ ] nyquist_compliant: true
