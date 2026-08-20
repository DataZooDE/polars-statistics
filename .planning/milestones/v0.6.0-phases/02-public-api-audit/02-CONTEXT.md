# Phase 2: Public API Audit - Context

**Gathered:** 2026-08-11
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — 1 grey area, all recommendations accepted

<domain>
## Phase Boundary

Produce an authoritative, documented gap list enumerating exactly which public
functions/capabilities in each upgraded backing crate (`anofox-statistics` 0.4.2 and
`anofox-regression` 0.5.13) are NOT yet reachable through the wrapper's Polars/Python
API. Each gap entry records the target exposure surface (Polars expression, PyModel
class, or both) so the two parity phases (3 and 4) have unambiguous scope. Confirm the
known candidates (ANOVA family, energy distance, GLMM, P-spline, Gamma GLM, HC
inference, diagnostics) present-or-absent against the ACTUAL upgraded crate surface.

This phase produces a DOCUMENT ONLY — no wrapper code is added or changed here.
Actually exposing the gaps is Phases 3 (statistics) and 4 (regression).

</domain>

<decisions>
## Implementation Decisions

### Audit Approach & Artifact (accepted from recommendations)
- **Artifact location & format:** Write the gap list to `.planning/phases/02-public-api-audit/02-API-AUDIT.md`
  as a planning artifact, with per-crate markdown tables. It feeds Phases 3/4 as their scope.
- **Enumeration method:** Enumerate each crate's public API authoritatively by scanning the
  installed crate source under `~/.cargo/registry/src/*/anofox-regression-0.5.13/` and
  `~/.cargo/registry/src/*/anofox-statistics-0.4.2/` for `pub` items (functions, structs,
  enums, trait methods, regressor types), cross-referenced against the wrapper's existing
  call sites (`src/expressions/*.rs`, `src/pymodels/*.rs`, `python/polars_statistics/`).
  Do NOT trust docs.rs alone; do NOT limit to the known-candidate list.
- **Definition of "exposed":** A capability counts as already-exposed if it is reachable from
  Python via EITHER a Polars expression OR a PyModel class (either surface suffices).
- **Output granularity:** One row per public fn/type — columns: name, kind, exposed? (yes/no),
  target surface (expression / PyModel / both), notes. Group by crate, then by category.

### Claude's Discretion
- Exact table column ordering, section headings, and how to sub-group categories.
- How to handle borderline "public but internal-only" items (e.g. helper traits) — use
  judgment; note them but mark low-priority if they are not meaningful user-facing capabilities.
- Whether to additionally emit a short summary count per crate/category.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Full codebase map at `.planning/codebase/` (STRUCTURE, ARCHITECTURE, CONVENTIONS, INTEGRATIONS).
- Phase 1 RESEARCH.md already inventoried the wrapper's call sites into both crates and confirmed
  the installed crate versions live in the local cargo registry cache — reuse that inventory.

### Established Patterns
- Two exposure surfaces per capability: statistics functions → Polars expressions
  (`src/expressions/*.rs` → `python/polars_statistics/exprs/*.py`); regression models →
  additionally PyModel classes (`src/pymodels/*.rs`).
- `src/expressions/mod.rs` and `src/pymodels/mod.rs` re-export the currently exposed surface —
  useful "what's already exposed" reference points.

### Integration Points
- The audit output (`02-API-AUDIT.md`) is consumed by Phase 3 (statistics parity) and
  Phase 4 (regression parity) planners as their authoritative scope list.

</code_context>

<specifics>
## Specific Ideas

- Known candidates to explicitly confirm present-or-absent in 0.5.13 / 0.4.2:
  ANOVA family (`one_way_anova`, `two_way_anova`, `repeated_measures_anova`), `energy_distance_test`,
  `GlmmRegressor`, `PSplineRegressor`, Gamma GLM, `HcInference`/`HcType`, and the diagnostics suite
  (Cook's distance, VIF, leverage, residual variants, condition diagnostics), plus streaming moment
  fits and robust solvers (Theil–Sen/RANSAC/LOWESS/Bayesian/passive-aggressive/LARS) where applicable.

</specifics>

<deferred>
## Deferred Ideas

- Actually exposing any gap → Phases 3 (statistics) and 4 (regression). This phase only documents.

</deferred>
