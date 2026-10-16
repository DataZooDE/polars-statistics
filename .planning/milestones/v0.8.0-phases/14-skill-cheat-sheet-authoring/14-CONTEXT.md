# Phase 14: Skill Cheat-Sheet Authoring - Context

**Gathered:** 2026-09-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Author the five capability-family `SKILL.md` cheat-sheets inside the plugin's
`plugins/polars-statistics/skills/` tree (created in Phase 13), replacing the stub
placeholders. Each cheat-sheet is a reference a Claude agent reads to apply the full
polars-statistics API. The five skills:
1. `polars-statistics-hypothesis-tests` — expression-API tests (parametric,
   nonparametric, distributional, correlation, categorical, modern MMD/energy,
   TOST/equivalence)
2. `polars-statistics-anova` — one-way, two-way, repeated-measures ANOVA + ICC
3. `polars-statistics-regression` — 40+ PyModel classes, regression expressions,
   R-style formula parser
4. `polars-statistics-results` — `to_dict()`/`summary()`/`__repr__`, unnest/struct
   helpers, contextual errors, typed `.pyi` API
5. `polars-statistics-getting-started` — install, model-selection matrix, sklearn
   migration

Out of scope: the dedicated "every example runs" accuracy gate + description-routing
verification (Phase 15), wheel bundling (Phase 16), release (Phase 17).

</domain>

<decisions>
## Implementation Decisions

### Coverage Depth
- **Exhaustive signatures**: every public function/class in a family gets a signature
  line so an agent can apply any symbol. Worked, runnable examples for representative
  cases per family (not one-example-per-symbol — that dilutes routing and bloats files).
- Regression's 40+ PyModel classes are grouped by family (OLS/Ridge/ElasticNet/GLM/
  Quantile/Huber/PLS/ALM/logistic/etc.) with a signature line each and worked examples
  for the common ones.

### Accuracy While Authoring
- **Spot-check as I write**: import the installed package and run key snippets per skill
  during authoring to catch wrong signatures/kwargs early. The full every-example gate
  is Phase 15 — authoring spot-checks do not replace it, they de-risk it.

### Signature Source of Truth
- The **shipped Python surface** is authoritative: `python/polars_statistics/`
  (`exprs/*.py`, `regression.py`, `results.py`, etc.) plus
  `python/polars_statistics/_polars_statistics.pyi`. **Never invent kwargs** — if a
  parameter is not in the shipped signature, it does not exist. (Carried from STATE.md
  Blockers note for Phases 14/15.)
- Cross-reference `docs/` cookbook and `examples/` scripts for idiomatic usage, but the
  signature itself comes from the Python source / `.pyi`.

### Frontmatter & Structure
- Each `SKILL.md` frontmatter: `name` (matches directory), `description` (task/keyword-
  rich single sentence describing WHAT TASKS route here — Phase 15 verifies routing),
  `version` pinned to the package version (`0.8.0`), `user-invocable: false`
  (auto-surfaced).
- Consistent cheat-sheet structure across all five (e.g. Overview → When to use → API
  reference by group with signatures → Worked examples → Gotchas/notes → Cross-links to
  sibling skills). Mirror the pyfda/fdars SKILL.md conventions where applicable.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Shipped Python API: `python/polars_statistics/exprs/*.py` (parametric, categorical,
  regression, etc.), `regression.py`, `results.py`, `formula/`, and the typed
  `_polars_statistics.pyi`.
- `docs/` (mkdocs cookbook) and `examples/` runnable scripts — idiomatic usage patterns.
- Phase 13 stub `SKILL.md` files (to be replaced) establish the five directory names
  and minimal frontmatter shape.
- pyfda/fdars installed plugin skills as a structural reference for SKILL.md conventions.

### Established Patterns
- The library exposes two API styles: Polars expressions (`df.select(ps.ttest_ind(...))`
  / `group_by().agg()`) and PyModel classes (`from polars_statistics import OLS`). Both
  must be represented where each family uses them.
- Output is struct-typed; `results.py` helpers (unnest/to_dict/summary) are how users
  consume results — the results skill documents this.

### Integration Points
- Skills live in `plugins/polars-statistics/skills/<name>/SKILL.md`; bundled into the
  wheel in Phase 16; verified in Phase 15.

</code_context>

<specifics>
## Specific Ideas

- `version` frontmatter pinned to `0.8.0` (matches plugin.json manifest string).
- `user-invocable: false` on all five (auto-surfaced, per milestone decision).
- Descriptions optimized for routing — Phase 15 asserts each description routes Claude to
  the right cheat-sheet for a representative task.

</specifics>

<deferred>
## Deferred Ideas

- Full "every example runs" execution gate and description-routing verification → Phase 15.
- Any new API surface — this phase documents only the shipped v0.8.0 (== v0.7.0) surface.

</deferred>
