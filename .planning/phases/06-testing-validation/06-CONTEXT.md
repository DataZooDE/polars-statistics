# Phase 6: Testing & Validation - Context

**Gathered:** 2026-08-12
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — 1 grey area, all recommendations accepted

<domain>
## Phase Boundary

Verify the newly exposed API (Phases 3–4) for correct output SHAPE and VALUES, validate
against references where available, add Rust-side wrapper/schema tests, and confirm the
full CI matrix passes green. Requirements TEST-01..05.

Many SMOKE tests already exist (executors wrote them in Phases 3–4: test_gamma, test_glmm,
test_pspline, test_glm_diagnostics, test_lars, test_pa, test_moments, test_theil_sen,
test_ransac, test_statistics_parity, etc.). This phase UPGRADES coverage from shape-only
smoke to VALUE-correctness, and adds Rust-side tests where missing.

Scope is TESTS ONLY (+ test-only deps) — no library behavior changes. Version bump + PyPI
publish is Phase 7.
</domain>

<decisions>
## Implementation Decisions

### Value-Validation Strategy & CI (accepted from recommendations)
- **TEST-03 value validation:** validate against **scipy/statsmodels** (themselves R-validated)
  where they implement the same method (e.g. one-way ANOVA vs `scipy.stats.f_oneway`; Gamma GLM
  vs `statsmodels`), PLUS **known analytic/textbook constants** where a library equivalent doesn't
  exist (e.g. ICC published examples, energy distance on separated samples). "Validated against R
  references wherever the crates supply them" is read pragmatically: use the STRONGEST available
  reference per method; where none exists, assert analytic/invariant properties and document why.
- **Test-only dependencies:** add `scipy` and `statsmodels` to the TEST/dev extras only (e.g.
  pyproject `[project.optional-dependencies].test` / dev group), NOT runtime deps. Guard imports
  with `pytest.importorskip` so the runtime wheel stays dependency-light.
- **Test depth per symbol:** each newly exposed statistics fn (TEST-01) and regression capability
  (TEST-02) gets a pytest test asserting output SHAPE/schema AND a VALUE-correctness assertion
  (vs reference or analytic). Rust-side tests (TEST-04) cover the new expression `*_fit` wrappers
  and their output-type schemas (extend `tests/rust_api.rs`).
- **TEST-05 full CI matrix:** ensure the LOCAL suite is comprehensive and green (clippy -D warnings,
  fmt, ruff, cargo test, pytest). The full matrix (Python 3.9–3.12 × Linux/macOS/Windows) cannot run
  in this sandbox — it is validated by the existing GH Actions CI on push; Phase 6 makes the local
  suite authoritative and confirms all local gates green. Also run a live `mkdocs build --strict`
  if mkdocs becomes available (deferred DOCS-02 live check).

### Claude's Discretion
- Which reference library per method; exact tolerances (use loose rtol/atol appropriate to each
  statistic); where a value check is analytic vs library-based.
- Grouping of tests into plans (by API area; likely a statistics-values plan, a regression-values
  plan, a Rust-side plan, and a green-gate plan).
- Whether to consolidate/upgrade the existing smoke test files vs add new value tests alongside.
</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- 34 pytest files already exist, incl. the Phase 3/4 smoke suites; `tests/rust_api.rs` has the
  Rust integration tests (15 currently). Baseline: 582 pytest + 15 rust_api green.
- The new-API list is in `02-API-AUDIT.md` and Phase 3/4 SUMMARYs.
- Ops: `maturin develop` must run in the FOREGROUND (background gets killed); pytest runs fine
  backgrounded; sweep `>1200s` orphaned cargo/rustc between heavy runs.

### Integration Points
- `.github/workflows/ci.yml` runs the matrix + clippy/fmt/ruff/coverage — TEST-05 is validated there.
- pyproject test extras is where scipy/statsmodels go.
</code_context>

<specifics>
## Specific Ideas

- Validate one_way_anova against `scipy.stats.f_oneway`; ICC against a published example; energy
  distance separation property; Gamma/GLM coefficients vs statsmodels where feasible.
- Keep the WLS hc_inference NotImplementedError test (already added) and the CR-01/02/03/05 Phase-4
  regression tests.
</specifics>

<deferred>
## Deferred Ideas

- Version bump 0.6.0 + PyPI publish → Phase 7.
- Any brand-new statistical methods → out of scope (parity milestone only).
</deferred>
