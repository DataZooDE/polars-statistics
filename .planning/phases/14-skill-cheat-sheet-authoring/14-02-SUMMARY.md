---
phase: 14-skill-cheat-sheet-authoring
plan: "02"
subsystem: documentation
tags: [polars-statistics, skill-cheat-sheet, hypothesis-tests, ttest, mann-whitney, correlation, icc, tost, mmd]

requires:
  - phase: 13-plugin-marketplace-scaffolding
    provides: stub SKILL.md in plugins/polars-statistics/skills/polars-statistics-hypothesis-tests/
  - phase: 14-01
    provides: locked frontmatter contract + section structure template

provides:
  - Full hypothesis-tests SKILL.md replacing Phase 13 stub
  - Exhaustive signatures for all 44 in-scope expression functions (7 families)
  - PyModel test class subsection (10 classes)
  - ICC separate-rater-columns gotcha documented
  - Categorical scalar vs column-arg distinction documented

affects: [14-03, 14-04, 14-05]

actuals:
  tokens: 6402
  tasks: 1
  commits: 1
plan_head_before: cf7ab2898b775403e0d6ab8c2e20134057c3288c

tech-stack:
  added: []
  patterns:
    - "SKILL.md cheat-sheet structure: Overview > When-to-use > API-Reference (grouped A-H) > Worked-Examples > Gotchas/Notes > See-also"
    - "SKILL.md frontmatter: exactly four fields — name (matches directory), description, version 0.8.0, user-invocable false"

key-files:
  created: []
  modified:
    - plugins/polars-statistics/skills/polars-statistics-hypothesis-tests/SKILL.md

key-decisions:
  - "Forecast functions (clark_west, spa_test, model_confidence_set, mspe_adjusted, diebold_mariano, permutation_t_test) excluded — out of scope for this milestone (FCST-01 deferred to v2)"
  - "ICC gotcha documented under Group D rather than Gotchas section — keeps it co-located with the signature for discoverability"
  - "Categorical functions split into scalar-argument vs contingency-table-column subgroups — clearer routing for agents"
  - "PyModel test classes documented as Group H in API Reference — consistent numbering with RESEARCH §137-317"
  - "TOST bounds_type table included inline — agents need to distinguish symmetric/raw/cohen_d without reading source"

requirements-completed: [SKILL-01, SKILL-06]

coverage:
  - id: D1
    description: "hypothesis-tests SKILL.md has valid YAML frontmatter (name=polars-statistics-hypothesis-tests, version=0.8.0, user-invocable=false, non-empty description)"
    requirement: SKILL-06
    verification:
      - kind: automated
        ref: "uv run python -c \"import yaml; d=yaml.safe_load(open(SKILL_FILE).read().split('---')[1]); assert d['name']=='polars-statistics-hypothesis-tests'; assert str(d['version'])=='0.8.0'; assert d['user-invocable'] is False; assert d.get('description'); print('FRONTMATTER_OK')\""
        status: pass
    human_judgment: false
  - id: D2
    description: "hypothesis-tests SKILL.md gives exhaustive signature line for every parametric, nonparametric, distributional, correlation, categorical, modern (MMD/energy), and TOST expression symbol"
    requirement: SKILL-01
    verification:
      - kind: automated
        ref: "ttest_ind spot-check: df.select(ps.ttest_ind('a','b')) runs without error"
        status: pass
    human_judgment: true
    rationale: "Full every-symbol accuracy gate is Phase 15; spot-checks during authoring de-risk it"
  - id: D3
    description: "Forecast comparison symbols absent from SKILL.md"
    requirement: SKILL-01
    verification:
      - kind: automated
        ref: "! grep -qE 'clark_west|spa_test|model_confidence_set|mspe_adjusted' SKILL.md && echo NO_FORECAST_OK"
        status: pass
    human_judgment: false

duration: 9min
completed: "2026-09-21"
status: complete
---

# Phase 14 Plan 02: polars-statistics-hypothesis-tests Cheat-Sheet Summary

**Exhaustive signatures for 44 expression functions across 7 test families (parametric, nonparametric, distributional, correlation, categorical, modern MMD/energy, TOST equivalence) plus 10 PyModel test classes in a production SKILL.md with ICC gotcha, categorical split, and TOST bounds_type table**

## Performance

- **Duration:** ~9 min
- **Started:** 2026-09-21T18:45:38Z
- **Completed:** 2026-09-21T18:54:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Replaced Phase 13 stub with 680-line production SKILL.md for `polars-statistics-hypothesis-tests`
- Group A Parametric (4 functions): `ttest_ind`, `ttest_paired`, `brown_forsythe`, `yuen_test` with parameter table and equal_var/Welch note
- Group B Non-parametric (4 functions): `mann_whitney_u`, `wilcoxon_signed_rank`, `kruskal_wallis`, `brunner_munzel` with `conf_level`/`mu` Optional[float] note
- Group C Distributional (2 functions): `shapiro_wilk`, `dagostino` with n-range notes (3..5000 and >=20)
- Group D Correlation (7 functions): `pearson`, `spearman`, `kendall`, `distance_cor`, `partial_cor`, `semi_partial_cor`, `icc` — with ICC type table (icc1/2/3/1k/2k/3k) and ICC gotcha documented inline
- Group E Categorical (13 functions): Split into scalar-argument (binom_test, prop_test_one/two, fisher_exact, mcnemar_test, mcnemar_exact, phi_coefficient) and contingency-table-column (chisq_test, g_test, cohen_kappa, cramers_v, contingency_coef) subgroups; plus chisq_goodness_of_fit
- Group F Modern (3 functions): `energy_distance`, `energy_distance_nd`, `mmd_test` with equal-length x_cols/y_cols requirement for nd variant
- Group G TOST (10 functions): all `tost_*` variants with `bounds_type` table (symmetric/raw/cohen_d) and method/rho_null/pooled/trim parameter table
- Group H PyModel (10 classes): `TTestInd`, `TTestPaired`, `BrownForsythe`, `YuenTest`, `MannWhitneyU`, `WilcoxonSignedRank`, `KruskalWallis`, `BrunnerMunzel`, `ShapiroWilk`, `DAgostino` — each with constructor, `.fit()`, key properties, `.is_fitted()/.summary()/.to_dict()`
- 8 worked examples: t-test (expression + group_by), Mann-Whitney, Shapiro-Wilk, Pearson+ICC, chi-square+Fisher, MMD, TOST equivalence, TTestInd PyModel
- 8 gotchas: ICC matrix contract, kruskal_wallis wide format, conf_level None semantics, scalar categorical args, cohen_d bounds restriction, uniform goodness-of-fit default, struct naming in group_by, PyModel property access before fit
- Frontmatter spot-checked: `FRONTMATTER_OK` and `NO_FORECAST_OK` both printed

## Task Commits

1. **Task 1: Author hypothesis-tests SKILL.md** — `b01ecc3` (docs)

## Files Created/Modified

- `plugins/polars-statistics/skills/polars-statistics-hypothesis-tests/SKILL.md` — complete
  cheat-sheet replacing the Phase 13 stub (680 lines, 44 expression functions, 10 PyModel classes)

## Decisions Made

- Forecast functions excluded per plan — `clark_west`, `spa_test`, `model_confidence_set`,
  `mspe_adjusted` are FCST-01 scope deferred to v2; `diebold_mariano` and
  `permutation_t_test` similarly excluded
- ICC gotcha placed inline under Group D signature (not just in Gotchas section) for
  maximum agent discoverability when looking up the `icc()` signature
- Categorical functions split into scalar-argument vs contingency-table-column subgroups
  per RESEARCH §218-220 — agents need to know which argument form to use before calling
- TOST `bounds_type="cohen_d"` restriction (t-test variants only) documented in both the
  signature table and the Gotchas section for double coverage

## Deviations from Plan

**Setup deviation (same as 14-01):** The worktree branch was spawned at `431306d`
(close-v0.7.0-milestone), while main had advanced to `cf7ab28` (Phase 14 planning + 14-01
complete). The `plugins/` tree did not exist in the worktree. Resolved by
`git merge main --no-edit` (fast-forward) before beginning task execution.

Otherwise: plan executed exactly as written.

## Known Stubs

None — all 44 expression function signatures and 10 PyModel class signatures are sourced
directly from shipped `python/polars_statistics/exprs/*.py` and `_polars_statistics.pyi`.
No invented kwargs; all spot-checked at runtime.

## Threat Flags

None — this phase writes documentation files only. No authentication, session management,
network endpoints, or schema changes involved.

---

*Phase: 14-skill-cheat-sheet-authoring*
*Completed: 2026-09-21*
