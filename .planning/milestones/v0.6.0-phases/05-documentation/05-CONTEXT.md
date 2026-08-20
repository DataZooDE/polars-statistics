# Phase 5: Documentation - Context

**Gathered:** 2026-08-12
**Status:** Ready for planning
**Mode:** Auto-generated (documentation phase — conventions fully determined by existing repo; discretion-based)

<domain>
## Phase Boundary

Document the ENTIRE newly exposed API surface from Phases 3 and 4, for both Python users
and Rust consumers, and describe the 0.6.0 release in the changelog. Four deliverables:
- DOCS-01: every newly exposed function/class has a Python docstring with signature,
  parameters, and a runnable example.
- DOCS-02: the mkdocs API reference pages list all newly exposed API.
- DOCS-03: Rust doc comments (`///`) are present for all new public wrapper functions.
- DOCS-04: the CHANGELOG documents the new API and the 0.6.0 release.

Scope is DOCUMENTATION ONLY — no code behavior changes (docstrings, doc comments, mkdocs
pages, CHANGELOG). Actual version bump + publish is Phase 7; comprehensive tests are Phase 6.

Newly exposed API to document:
- **Statistics (Phase 3):** `one_way_anova`, `two_way_anova`, `repeated_measures_anova`,
  `energy_distance_nd`, and the real matrix-input `icc` (new contract).
- **Regression (Phase 4):** PyModels `Gamma`, `GLMM`, `PSpline`, `TheilSen`, `RANSAC`,
  `BayesianRidge`, `ARD`, `LARS`, `PassiveAggressive`, `MomentAccumulator`; `Ridge`/`WLS`
  `hc_inference` (note WLS raises NotImplementedError — document the limitation);
  `OLS`/`Ridge` `fit_from_accumulator`; 5 GLM-diagnostic expressions
  (`gamma_dispersion_deviance`, `gamma_dispersion_pearson`, `gamma_pearson_chi_squared`,
  `gamma_standardized_pearson_residuals`, `gamma_standardized_deviance_residuals`).
</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion (follow existing repo conventions)
- **Docstring style:** match the existing numpy-style docstrings already used across
  `python/polars_statistics/exprs/*.py` and `src/pymodels/*.rs` PyO3 docstrings (Parameters,
  Returns, Examples sections). Many new functions already carry docstrings from Phases 3–4 —
  audit and fill gaps rather than rewrite; ensure every one has a RUNNABLE example.
- **mkdocs structure:** follow the existing `docs/` layout (`docs/api/`, `docs/api/classes`,
  `docs/api/regression`, `API_REFERENCE.md`, `mkdocs.yml` nav). Add the new
  functions/classes to the appropriate existing reference pages / nav sections; do not
  restructure the docs site.
- **Rust doc comments:** `///` item docs + `//!` module docs consistent with existing
  `src/expressions/*.rs` and `src/pymodels/*.rs` conventions.
- **CHANGELOG:** follow the existing `CHANGELOG.md` format; add a `0.6.0` section listing the
  new statistics + regression API (grouped), the anofox crate bumps, and the column-pivot
  correctness fix inherited in Phase 1. Note the WLS-HC NotImplementedError limitation.
- **Examples must be runnable** — use small inline DataFrames / numpy arrays consistent with
  the smoke tests; do not require external data.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Existing docs site: `docs/` (api/, API_REFERENCE.md, examples/, getting-started.md, index.md),
  `mkdocs.yml`, `CHANGELOG.md`.
- The Phase 3/4 PyModels and expressions already carry numpy-style docstrings written during
  implementation — this phase audits + completes them and surfaces them in mkdocs.
- The new API surface is enumerated in `02-API-AUDIT.md` and the Phase 3/4 SUMMARYs.

### Integration Points
- mkdocs API pages (mkdocstrings or manual) pull from the Python docstrings; ensure the new
  symbols appear in the nav/reference.

</code_context>

<specifics>
## Specific Ideas

- Cross-reference the WLS `hc_inference` NotImplementedError limitation and the ICC contract
  change (new matrix-input signature) explicitly in the docs and CHANGELOG so users aren't
  surprised.

</specifics>

<deferred>
## Deferred Ideas

- Version bump to 0.6.0 in Cargo.toml/pyproject.toml + PyPI publish → Phase 7.
- Comprehensive R-validated tests → Phase 6.
</deferred>
