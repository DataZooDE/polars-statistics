# Phase 3: Statistics API Parity - Context

**Gathered:** 2026-08-11
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — 1 grey area, all recommendations accepted

<domain>
## Phase Boundary

Make every unexposed `anofox-statistics` 0.4.2 function identified in the Phase 2 audit
callable from Python via the Polars expression API, following the existing
`#[polars_expr]` + output-struct-schema patterns. The audit's actionable statistics
gaps are:
- **ANOVA family** — `one_way_anova` (STAT-01), `two_way_anova` + `repeated_measures_anova` (STAT-02/03)
- **energy_distance_test nD** overload (STAT-04)
- **ICC** — currently registered but `icc_fit` returns all-NaN (stub) — implement for real (STAT-05)

Scope is IMPLEMENTATION + exposure (expressions, output structs, Python builders,
Rust doc comments inline). Comprehensive R-validated testing is Phase 6; user-facing
docs pages are Phase 5. This phase must produce correct, callable expressions with
sensible output schemas, plus at least a smoke-level correctness check.

</domain>

<decisions>
## Implementation Decisions

### Statistics Parity Scope & Output Shape (accepted from recommendations)
- **STAT-04 (energy distance nD):** Expose the multi-dimensional `energy_distance_test`
  overload as a NEW Polars expression. The existing 1D `energy_distance` expression stays;
  the nD variant is the genuine gap and is what delivers real parity. (Resolves audit Deferred Q1.)
- **ICC:** Implement the real matrix-input ICC (`ICCResult` output with `ICCType` parameter),
  replacing the all-NaN stub in `src/expressions/correlation.rs` (`icc_fit` with `// TODO`).
  A NaN-returning binding is effectively unexposed, so it falls under STAT-05. (Resolves audit Deferred Q3.)
- **ANOVA output structs:** Rich schema — F-statistic, degrees of freedom (between/within),
  p-value, sum-of-squares, mean-squares, and effect size (η²) WHERE the crate's ANOVA result
  types provide them. Mirror the existing `stats_output_dtype`/output-struct conventions
  (`src/expressions/output_types.rs`). Do not invent fields the crate does not return.
- **STAT-05 scope:** "Every remaining unexposed statistics function" = exactly
  {ANOVA family, energy_distance_test nD, ICC}. Internal-only `pub` items (e.g. the LOWESS
  helper) are explicitly EXCLUDED — they are not user-facing capabilities.

### Claude's Discretion
- Exact output-struct field names/ordering (follow existing naming conventions).
- Which existing expression file each new expression lives in (parametric.rs for ANOVA,
  the existing energy/modern file for energy nD, correlation.rs for ICC) — follow the
  established category-to-file mapping.
- How the nD energy expression accepts multi-dimensional input within the Polars expression
  model (e.g. list/array columns) — follow existing multi-input expression patterns.
- The exact matrix-input contract for ICC (how raters/subjects are passed) — follow the
  crate's `ICCResult`/`ICCType` API and the existing correlation-expression patterns.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`02-API-AUDIT.md`** (Phase 2) is the authoritative scope list — per-item rows with target
  surface and notes; consult it directly.
- **`02-RESEARCH.md`** has the crate-source enumeration with exact function/type names and modules.
- Existing expression patterns: `src/expressions/*.rs` (`#[polars_expr]` fns + `pl_*` variants),
  output schemas in `src/expressions/output_types.rs`, Python builders in
  `python/polars_statistics/exprs/*.py`, registration via `src/expressions/mod.rs` + `__init__.py`.

### Established Patterns
- Every statistics expression: `#[polars_expr(output_type_func=...)]` returning a Struct series;
  output schema defined by a `*_output_dtype()` fn; Python builder wraps it as a callable.
- ICC stub lives at `src/expressions/correlation.rs` (`icc_fit`, currently NaN).

### Integration Points
- New expressions must be registered in `src/expressions/mod.rs` and surfaced through the
  Python `__init__.py` / `exprs/*.py` builders to be callable from Python.

</code_context>

<specifics>
## Specific Ideas

- Follow the crate's actual ANOVA result types (from `02-RESEARCH.md`) for the 5 ANOVA result
  structs — do not fabricate fields.
- Keep the 1D `energy_distance` expression untouched; add nD alongside.
- ICC: use the crate's real `ICCResult`/`ICCType` matrix-input API — this is a genuine
  implementation, not just wiring.

</specifics>

<deferred>
## Deferred Ideas

- Comprehensive R-validated tests for the new expressions → Phase 6 (Testing & Validation).
- mkdocs API pages + full user docs → Phase 5 (Documentation).
- Regression-side parity (Gamma/GLMM/PSpline/HC-extend/solvers/diagnostics) → Phase 4.

</deferred>
