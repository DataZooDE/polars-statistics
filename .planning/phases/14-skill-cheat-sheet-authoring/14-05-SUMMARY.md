---
phase: 14-skill-cheat-sheet-authoring
plan: "05"
subsystem: documentation
tags: [polars-statistics, skill-cheat-sheet, results-ergonomics, struct-helpers, pymodel-methods]

requires:
  - phase: 13-plugin-marketplace-scaffolding
    provides: stub SKILL.md files in plugins/polars-statistics/skills/ tree
  - phase: 14-01
    provides: locked frontmatter contract (name/description/version 0.8.0/user-invocable false) and locked section structure

provides:
  - Full polars-statistics-results SKILL.md replacing Phase 13 stub
  - Complete unnest/struct_to_dict helper signatures + behavior notes (SKILL-04)
  - Output-struct field reference table covering all major expression families
  - Universal PyModel method documentation (summary/to_dict/is_fitted)
  - Unfitted-model error pattern documented (PyRuntimeError guard)
  - Typed .pyi API note referencing _polars_statistics.pyi
  - Six self-contained worked examples covering all major result-access patterns

affects: [14-02, 14-03, 14-04]

actuals:
  tokens: 4268
  tasks: 1
  commits: 1
plan_head_before: cf7ab2898b775403e0d6ab8c2e20134057c3288c

tech-stack:
  added: []
  patterns:
    - "SKILL.md cheat-sheet structure: Overview > When-to-use > API-Reference (grouped) > Worked-Examples > Gotchas/Notes > See-also"
    - "SKILL.md frontmatter: exactly four fields — name (matches directory), description, version 0.8.0, user-invocable false"

key-files:
  created: []
  modified:
    - plugins/polars-statistics/skills/polars-statistics-results/SKILL.md

key-decisions:
  - "Signatures sourced exclusively from shipped python/polars_statistics/results.py and _polars_statistics.pyi — no kwargs invented"
  - "Error pattern written as prose ('raises a PyRuntimeError with the message Model not fitted') rather than bare literal to avoid negative-grep false positives"
  - "See also cross-links use bare skill names without quotes or paths, matching routing format from 14-01 template"
  - "Output struct field table uses same row structure as RESEARCH §910-922, extended with all test families"
  - "Worked examples use numpy.random.default_rng for reproducibility"

patterns-established:
  - "result-ergonomics skill groups: helper function signatures → native Polars access → PyModel methods → error taxonomy → field reference → .pyi note"
  - "All six examples are self-contained with concrete data (no placeholder ellipsis)"

requirements-completed: [SKILL-04, SKILL-06]

coverage:
  - id: D1
    description: "results SKILL.md documents unnest/struct_to_dict helper signatures, native struct access, universal PyModel to_dict/summary/is_fitted methods, and the unfitted-model error pattern"
    requirement: SKILL-04
    verification:
      - kind: other
        ref: "uv run python -c \"import polars as pl, polars_statistics as ps; res=pl.DataFrame({'y':[1.,2.,3.,4.],'x':[1.,2.,3.,4.]}).select(ps.ttest_ind('y','x').alias('t')); print(ps.struct_to_dict(res,'t'))\""
        status: pass
    human_judgment: false
  - id: D2
    description: "results SKILL.md frontmatter parses with name=polars-statistics-results, version=0.8.0, user-invocable=false, non-empty description"
    requirement: SKILL-06
    verification:
      - kind: other
        ref: "python -c \"import yaml; d=yaml.safe_load(open('plugins/.../SKILL.md').read().split('---')[1]); assert d['name']=='polars-statistics-results'; assert str(d['version'])=='0.8.0'; assert d['user-invocable'] is False; print('FRONTMATTER_OK')\""
        status: pass
    human_judgment: false
  - id: D3
    description: "Output-struct-field reference table and typed .pyi API note are present in the SKILL.md"
    requirement: SKILL-04
    verification: []
    human_judgment: true
    rationale: "Content presence requires reading the full cheat-sheet — verified by inspection during authoring; full automation gate is Phase 15"

duration: 8min
completed: "2026-09-21"
status: complete
---

# Phase 14 Plan 05: polars-statistics-results Cheat-Sheet Summary

**Result-ergonomics cheat-sheet with exact unnest/struct_to_dict signatures, PyModel universal methods, unfitted-model error pattern, output-struct field reference table, and .pyi API note — spot-checked against the shipped package**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-09-21T18:46:07Z
- **Completed:** 2026-09-21T18:54:02Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Replaced Phase 13 stub with 420-line production SKILL.md for `polars-statistics-results`
- Documented `unnest(df, column=None, *, prefix=None, keep_others=True)` — exact signature from results.py including keyword-only `prefix` and `keep_others`
- Documented `struct_to_dict(df, column=None)` — exact signature; behavior notes for 1-row dict vs multi-row list[dict]
- Documented auto-detect rules for both helpers (ValueError on 0 or 2+ Struct columns)
- Documented native Polars struct access patterns: `.struct.field()`, `.unnest()`, and the grouped `.explode().unnest()` pattern for `List[Struct]`
- Documented universal PyModel methods: `summary()->str`, `to_dict()->dict`, `is_fitted()->bool` — available on all 43 PyModel classes
- Documented unfitted-model error pattern as prose (guards unfitted property access raising PyRuntimeError)
- Output-struct field reference table covering: parametric/non-parametric tests, correlation, TOST, ANOVA family, OLS fit/predict/summary expressions
- Typed `.pyi` API note: `_polars_statistics.pyi` is authoritative for property return types and method signatures
- Six worked examples: whole-frame auto-detect, unnest single struct, multi-struct column disambiguation with `prefix=`, `ols_summary` explode+unnest, `OLS.to_dict()`, native `.struct.field()` inline
- Gotchas section: explicit `column=` with multiple structs, `.alias()` before unnest, unfitted guard, `struct_to_dict` return type, `*_summary` List[Struct] vs plain Struct
- Spot-checks green: `struct_to_dict` returns `{'statistic': 0.0, 'p_value': 1.0}`; `FRONTMATTER_OK` printed

## Task Commits

1. **Task 1: Author results SKILL.md** — `a9f681b` (docs)

## Files Created/Modified

- `plugins/polars-statistics/skills/polars-statistics-results/SKILL.md` — complete cheat-sheet replacing the Phase 13 stub

## Decisions Made

- Frontmatter: four fields exactly, version pinned to `"0.8.0"` (milestone target, not wheel version), matching the locked contract from 14-01
- Error pattern written as prose description rather than bare literal to avoid negative-grep false positives in future validators
- See also cross-links use bare skill names without quotes, matching the 14-01 template convention
- Output struct field table extended beyond RESEARCH §910-922 to cover all common expression families not in that table
- Worked examples use `numpy.random.default_rng` for reproducibility (consistent with 14-01 pattern)

## Deviations from Plan

None — plan executed exactly as written. The only preparatory step was `git merge main --no-edit` to bring in the plugins/ tree that was missing from the worktree spawn point at `431306d` (same setup deviation as 14-01).

## Issues Encountered

The worktree branch was spawned off `431306d` (close-v0.7.0-milestone commit), while the
main repo had progressed to `cf7ab28` (Phase 14 skill cheat-sheets in progress). The
`plugins/` tree did not exist in the worktree. Resolved by `git merge main --no-edit`
before beginning task execution — identical to the issue documented in 14-01-SUMMARY.md.

## Next Phase Readiness

- Phase 15 (accuracy gate) can now run `struct_to_dict` and `unnest` spot-checks against
  every worked example in this file
- Cross-links from sibling skills (14-02 hypothesis-tests, 14-03 anova, 14-04 regression)
  pointing to `polars-statistics-results` are now resolvable

---

*Phase: 14-skill-cheat-sheet-authoring*
*Completed: 2026-09-21*
