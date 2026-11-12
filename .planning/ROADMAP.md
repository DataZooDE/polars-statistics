# Roadmap: polars-statistics

## Milestones

- ✅ **v0.6.0 API Parity & Release** - Phases 1-7 (shipped 2026-08-19)
- ✅ **v0.7.0 Ergonomics, Adoption & Documentation** - Phases 8-12 (shipped 2026-09-20)
- ✅ **v0.8.0 Claude Code Skills Plugin** - Phases 13-17 incl. 16.1-16.4 (shipped 2026-10-16)
- ✅ **0.9.0 bug-fix release** - out of milestone, no phases (shipped 2026-11-11). Fixed GH #51 (the rlib no longer pulls in pyo3; pyo3-polars is optional) and GH #52 (every expression upcasts numeric inputs to Float64/UInt32) via quick tasks 261111-lm9 and 261111-lma. Tag `v0.9.0`.
- 🚧 **v0.10.0 Upstream Wave 3** - Phases 18-25 (in progress)
- 📋 **v2.0 Forecast Integration** - deferred (FCST-01, not yet planned)

## Overview

v0.10.0 moves polars-statistics onto the latest backing crates (anofox-statistics 0.4.7 →
0.10.0, anofox-regression 0.5.22 → 0.12.2) and the current Polars/PyO3 stack (pyo3-polars
0.28 / polars 0.55 / pyo3 0.29 / numpy 0.29) while keeping R parity, exposes the small new
upstream surface, and ships 0.10.0 to PyPI. The upgrade spans 25+ upstream releases with both
numeric and API breaks, so each kind of change gets its own phase. That way a failing test
always has one cause. anofox-statistics goes first because it has no compile breaks and is the
lowest risk (18). anofox-regression is split into the 0.8 numerics step (19) and the 0.9 →
0.12.2 API-break step (20), so an R-fixture failure is never mixed up with a compile error.
The new statistics surface (21) and regression surface (22) are then built on the upgraded
crates. The Polars/PyO3 stack is the last code-migration phase (23). Docs and skills are then
refreshed against the final build (24), and release comes last (25).

**Why the new surface comes before the stack upgrade:**

- **Isolation.** All statistical work, both the upgrades and the R-validation of the new
  surface, happens on the trusted polars 0.52 / pyo3 0.26 stack. Any R mismatch in Phases 18–22
  is therefore the wrapper's fault, not the stack's. This is the same principle behind upgrading
  anofox before the stack.
- **A larger baseline.** STACK-01's "behaves identically" check in Phase 23 then runs against
  the largest possible suite, including every new R-reference test. Any diff in Phase 23 is
  attributable to the stack alone.
- **Risk containment.** The stack is the one change whose size has not been verified: polars
  0.55 was not compiled offline, and a Series→Column change in the plugin signatures is
  possible. Running it last means it cannot block the user-visible surface. It can also fall
  back to stepping pyo3-polars 0.26 → 0.27 → 0.28 without disturbing other phases.
- **Cost.** The new expressions from Phases 21–22 are written against polars 0.52 and must be
  migrated in Phase 23 with the 323 existing `#[polars_expr]` functions. That is a small,
  mechanical addition to a pass that has to happen anyway.

**Behaviour-change log convention:** Phases 18–20 each add their changed outputs, with the R
reference, to the CHANGELOG `[Unreleased]` "Behaviour changes" list as they land. Phase 24
consolidates that list into `[0.10.0]`. Policy: adopt the R-matching output, document it, lock
it with an R-reference test, and add no deprecation warnings.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.
Phase numbering is continuous across milestones. v0.10.0 continues from Phase 18: v0.8.0 ended
at Phase 17, and the 0.9.0 bug-fix release had no phases.

<details>
<summary>✅ v0.6.0 API Parity & Release (Phases 1-7) — SHIPPED 2026-08-19</summary>

- [x] **Phase 1: Dependency Modernization** - Bump both anofox crates, reconcile breakage, keep existing tests green (completed 2026-08-11)
- [x] **Phase 2: Public API Audit** - Enumerate every unexposed function in both crates as an authoritative gap list (completed 2026-08-11)
- [x] **Phase 3: Statistics API Parity** - Expose ANOVA family, energy distance, and all remaining statistics gaps (completed 2026-08-12)
- [x] **Phase 4: Regression API Parity** - Expose GLMM, P-spline, Gamma GLM, HC inference, diagnostics, and remaining regression gaps (completed 2026-08-12)
- [x] **Phase 5: Documentation** - Document all newly exposed API across docstrings, mkdocs, Rust doc comments, and CHANGELOG (completed 2026-08-12)
- [x] **Phase 6: Testing & Validation** - Cover all new API with pytest + Rust tests, validated against R, CI matrix green (completed 2026-08-12)
- [x] **Phase 7: Release 0.6.0** - Bump version, publish wheels to production PyPI, smoke-check the release (completed 2026-08-19)

Full v0.6.0 phase detail is preserved in git history and `.planning/milestones/v0.6.0-phases/`.

</details>

<details>
<summary>✅ v0.7.0 Ergonomics, Adoption & Documentation (Phases 8-12) — SHIPPED 2026-09-20</summary>

- [x] **Phase 8: Type Safety** - Ship `.pyi` stubs + `py.typed` for all Rust-bound classes, expressions, and formula helpers, with a CI drift guard (completed 2026-09-20)
- [x] **Phase 9: Result Ergonomics** - Friendly result access: `.to_dict()`, `.summary()`/`__repr__`, and a one-call struct-unnest helper (completed 2026-09-20)
- [x] **Phase 10: Errors & API Consistency** - Contextual errors plus `with_intercept`→`add_intercept` deprecation and uniform sklearn-style `fit/predict/score` (completed 2026-09-20)
- [x] **Phase 11: Documentation & Adoption** - Cookbook + examples, decision matrix, migration guide, sklearn page, refreshed README (completed 2026-09-20)
- [x] **Phase 12: Release 0.7.0** - Bump 0.6.0→0.7.0 across all version sources and publish to production PyPI (completed 2026-09-20)

Full v0.7.0 phase detail is preserved in git history and `.planning/milestones/v0.7.0-phases/`.

</details>

<details>
<summary>✅ v0.8.0 Claude Code Skills Plugin (Phases 13-17) — SHIPPED 2026-10-16</summary>

- [x] Phase 13: Plugin & Marketplace Scaffolding (1/1 plans) — completed 2026-09-21
- [x] Phase 14: Skill Cheat-Sheet Authoring (5/5 plans) — completed 2026-09-21
- [x] Phase 15: Accuracy Verification Pass (1/1 plans) — completed 2026-09-21
- [x] Phase 16.1: Upstream Crate Refresh (INSERTED) (9/9 plans) — completed 2026-10-09
- [x] Phase 16.2: Penalized GLM and AFT Survival Regression (INSERTED) (8/8 plans) — completed 2026-10-09
- [x] Phase 16.3: Upstream API Wave 2 (INSERTED) (11/11 plans) — completed 2026-10-15
- [x] Phase 16.4: Ruff Cleanup (INSERTED) (3/3 plans) — completed 2026-10-15
- [x] Phase 16: Wheel Bundling (1/1 plans) — completed 2026-10-15
- [x] Phase 17: Release 0.8.0 (5/5 plans) — completed 2026-10-16

Full detail: `.planning/milestones/v0.8.0-ROADMAP.md`, `v0.8.0-REQUIREMENTS.md`, `v0.8.0-MILESTONE-AUDIT.md`, and `.planning/milestones/v0.8.0-phases/`.

</details>

### 🚧 v0.10.0 Upstream Wave 3 (In Progress)

**Milestone Goal:** Move onto anofox-statistics 0.10.0, anofox-regression 0.12.2 and the
pyo3-polars 0.28 / polars 0.55 / pyo3 0.29 stack while keeping R parity, expose the small new
upstream surface, and ship 0.10.0 to PyPI. Large upstream families (FAM-01..08) are deferred.

- [ ] **Phase 18: anofox-statistics 0.10.0 Upgrade** - Upgrade to anofox-statistics 0.10.0, fix the posthoc CI test, and R-validate and document the 0.8.x semantic changes
- [ ] **Phase 19: anofox-regression 0.8 Numerics** - Upgrade to anofox-regression 0.8.x, apply the required wrapper fixes, and re-validate every moved R fixture against R
- [ ] **Phase 20: anofox-regression 0.12.2 API Migration** - Migrate to anofox-regression 0.12.2's API breaks (clippy-clean) and lock the VIF / P-spline AIC semantics
- [ ] **Phase 21: New Statistics Surface** - Expose ttest_1samp, two-sample/k-sample distribution tests, time-series tests, bootstrap methods and the new SPA/MCS/posthoc options
- [ ] **Phase 22: New Regression Surface** - Expose predict_se, cluster/HC vcov, nested F / ANOVA tables, BP/White and influence diagnostics, and the new model kwargs
- [ ] **Phase 23: Polars / PyO3 Stack Upgrade** - Move to pyo3-polars 0.28 / polars 0.55 / pyo3 0.29 / numpy 0.29, drop unused deps, and declare and CI-test the Python polars range
- [ ] **Phase 24: Docs & Skills Refresh** - Document the full v0.10.0 surface and every behaviour change; stubs, mkdocs and skills all verified live
- [ ] **Phase 25: Release 0.10.0** - Bump 0.9.0 → 0.10.0 across all 18 version sources, publish to production PyPI via OIDC, and smoke-check the release

## Phase Details

### Phase 18: anofox-statistics 0.10.0 Upgrade

**Goal**: polars-statistics runs on anofox-statistics 0.10.0. Every statistics output that changed to match R now returns the R value, is locked by an R-reference test, and is documented as a behaviour change.
**Depends on**: Nothing (first phase of v0.10.0; builds on the shipped 0.9.0 baseline)
**Requirements**: UPST-11, UPST-12
**Success Criteria** (what must be TRUE):

  1. The crate builds against `anofox-statistics =0.10.0` and the full pytest and Rust suites pass. `ps.pairwise_t_test` returns real `conf_low`/`conf_high` values instead of NaN. They match the regenerated `tests/fixtures/posthoc_reference.csv`, and the docstring describes them.
  2. SPA and MSPE-adjusted SPA outputs report `p_value_consistent` and `p_value_upper` in the corrected (unswapped) order, and MCS `mcs_p_value`/`included_models` match R `MCSprocedure`. Each is locked by an R-reference test. Inputs that now error upstream (`n_bootstrap=0`, n<2, block length ≤0) give NaN instead of crashing.
  3. ICC(2,1)/ICC(2,k) confidence intervals (the default `icc2`) match `psych::ICC`. For one-sided alternatives, `conf_int()` on the `TTestInd`/`TTestPaired` PyModels returns a one-sided interval with an infinite bound, matching `t.test`.
  4. `semi_partial_cor` with two or more controls matches `ppcor` (joint residualization). Repeated-measures ANOVA `hf_epsilon`/`hf_p_value` match `afex` (Lecoutre HF ε, capped at 1).
  5. Every changed output above is described in its docstring and listed, with its R reference, in the CHANGELOG `[Unreleased]` "Behaviour changes" section.

**Notes**: No compile breaks are expected, because all 68 imported functions keep their signatures. Suggested stepping stones from research: 0.4.11 (posthoc CI only) → 0.7.0 → 0.8.3 (the semantic batch) → 0.10.0 (additive). The new SPA/MCS result fields are exposed in Phase 21, not here.
**Plans**: TBD

### Phase 19: anofox-regression 0.8 Numerics

**Goal**: polars-statistics runs on anofox-regression 0.8.x, and every regression output that moved under the 0.8 numerics is re-validated against R before any API-break migration starts. A fixture failure in this phase is therefore always a numerics question, never a compile error.
**Depends on**: Phase 18 (execution order only; the two anofox crates are independent)
**Requirements**: UPST-13
**Success Criteria** (what must be TRUE):

  1. The crate builds against anofox-regression 0.8.x and the full pytest and Rust suites pass. `ps.augment` still returns its empty-output struct for empty or all-null groups, now built without an `Augment {..}` struct literal.
  2. The penalized GLM (PyModel and expressions) reports `model_info["model_type"] == "glm"`. Its `vcov` applies the dispersion φ only once, so `sqrt(diag(vcov))` equals the reported `std_errors` and matches R for Gamma and quasi-Poisson fits.
  3. `LmDynamic` (PyModel and expression) fits a model with 7 or more predictors under default settings without hitting the `max_models` limit.
  4. Every R-parity fixture that moved under 0.8 passes against a value re-derived from R: elastic-net coordinate descent, Poisson SE, IRLS tolerance, VIF/GVIF/condition labels, ALM, BLS, P-spline, lmDynamic, BayesianRidge/ARD, LARS and GLMM. No tolerance is widened without an R-backed reason recorded in the test.
  5. Each moved output is listed, with old → new value and its R reference, in the CHANGELOG `[Unreleased]` "Behaviour changes" section.

**Notes**: Suggested stepping stones: 0.5.34 → 0.5.38 (the Augment fields) → 0.6/0.7 (basin, ALM) → 0.8.x. Stop before 0.9.0.
**Plans**: TBD

### Phase 20: anofox-regression 0.12.2 API Migration

**Goal**: polars-statistics builds clippy-clean on anofox-regression 0.12.2 with no user-visible API change, apart from two documented 0.11/0.12 semantics: an undefined VIF is now ∞/NaN, and P-spline AIC/BIC are no longer zero.
**Depends on**: Phase 19 (numerics settled on 0.8.x)
**Requirements**: UPST-14
**Success Criteria** (what must be TRUE):

  1. The crate builds against `anofox-regression =0.12.2` with `cargo clippy -D warnings` and `cargo fmt` clean. No deprecated upstream names remain: `LogisticRegression`/`BayesianRidge`/`ArdRegression` and the `.fit_intercept()` builder calls are gone. The `IterativeFit`, `ColRef`, `non_exhaustive` and `RegressionError` call sites are migrated, and every match over a non-exhaustive upstream enum has a fallback arm.
  2. The full pytest and Rust suites from Phase 19 pass with no expected-value edits apart from the two 0.11/0.12 semantics. Python class names, kwargs and getters are unchanged. For example, `LogisticRegression.n_iter` and `PassiveAggressive.n_iter` still work even though upstream removed `n_iter`.
  3. VIF/GVIF return ∞ or NaN (not 1) when undefined, and P-spline AIC/BIC are non-zero and match mgcv. Both are locked by tests, described in the docstrings, and listed in the CHANGELOG `[Unreleased]` "Behaviour changes" section.
  4. Upstream errors (`RegressionError`, the split `NumericalError`, infinite-input errors) reach Python as clear `ValueError`s. Existing `null_policy` behaviour on null/NaN rows is unchanged by upstream's new NaN-dropping default.

**Notes**: Suggested path: 0.9.0 (the API break) → 0.12.2 (additive, apart from the VIF/P-spline semantics). anofox-regression 0.12.2 requires Rust 1.87. The `rust-version` declaration lands in Phase 23 (STACK-02).
**Plans**: TBD

### Phase 21: New Statistics Surface

**Goal**: Users can call the new anofox-statistics 0.10.0 tests and options from Polars expressions and PyModels: the one-sample t-test, two-sample and k-sample distribution tests, time-series diagnostics, bootstrap methods, and the new SPA/MCS/posthoc options. Each works in `group_by`/`over` and is R-validated.
**Depends on**: Phase 18 (uses anofox-statistics 0.10.0 APIs; independent of the regression phases)
**Requirements**: STAT-01, STAT-02, STAT-03, STAT-04
**Success Criteria** (what must be TRUE):

  1. `ps.pairwise_t_test(..., conf_level=)`, `ps.p_adjust(..., n=)` and `ps.ttest_1samp` (with Cohen's d) match R `pairwise.t.test`, `p.adjust(n=)` and `t.test(mu=)`. SPA output carries `p_value_lower`, MCS output carries `mcs_p_values` and `elimination_order`, and `ps.mmd_test` accepts the Gaussian median-heuristic bandwidth.
  2. Bootstrap CI expressions accept `method` (iid, stationary, circular, moving block) and `block_length`. A `MovingBlockBootstrap` class behaves like the existing `StationaryBootstrap` and `CircularBlockBootstrap` classes.
  3. `ps.ks_test_2sample`, `ps.ad_test_ksamp`, `ps.cvm_test_2sample` and `ps.epps_singleton_test` return per-group results in `group_by`/`over`. They match R `ks.test`, `kSamples::ad.test` and `twosamples`, or scipy where R has no reference.
  4. `ps.acf`, `ps.pacf`, `ps.ljung_box`, `ps.box_pierce` and `ps.durbin_watson_test` (exact p-value) match R `acf`, `pacf`, `Box.test` and `lmtest::dwtest`.
  5. Every new builder, kwarg and class has a docstring and a `.pyi` stub entry. All CI gates stay green, including clippy, fmt, ruff and the stub drift guard.

**Notes**: New expressions are written against polars 0.52 and will be migrated in Phase 23 with the rest of the expression set.
**Plans**: TBD

### Phase 22: New Regression Surface

**Goal**: Users can get prediction standard errors, cluster/HC covariance, nested-model tests, heteroskedasticity and influence diagnostics, and the new model kwargs from anofox-regression 0.12.2 through PyModels and expressions. All of it is R-validated.
**Depends on**: Phase 20 (uses anofox-regression 0.12.2 APIs)
**Requirements**: REG-01, REG-02, REG-03
**Success Criteria** (what must be TRUE):

  1. OLS, WLS and GLM models give prediction standard errors (`predict_se`), both as a PyModel method and as an expression option, matching R `predict(..., se.fit = TRUE)`.
  2. `cluster_vcov` and `hc_vcov` match `sandwich::vcovCL`/`vcovHC`, and `nested_f_test`/`anova_table` match R `anova()`.
  3. `breusch_pagan` and `white_test` match `lmtest::bptest`, and `partial_residuals` and `added_variable` match `car`. `detect_separation` flags separated logistic data, and `augment` output includes the LOO/PRESS fields.
  4. The new kwargs and fields are reachable on the PyModels and, where an expression exists, as expression kwargs:
     - Poisson/Tweedie/Gamma `weights`, and Poisson `estimate_dispersion` (quasi-Poisson).
     - Quantile `alpha`, the `HuberScale::Joint` option, ridge `conditioning_aware`, and Theil–Sen CIs.
     - GLMM `satterthwaite`, random-effects SE and `vcov`.
     - Bayesian `credible_intervals`/`predict_std`, and P-spline `lambda`.
  5. `.pyi` stubs cover every new method and kwarg, and all CI gates are green, including the stub drift guard.

**Notes**: New expressions are written against polars 0.52 and will be migrated in Phase 23 with the rest of the expression set.
**Plans**: TBD

### Phase 23: Polars / PyO3 Stack Upgrade

**Goal**: polars-statistics builds on pyo3-polars 0.28 / polars 0.55 / pyo3 0.29 / numpy 0.29 and behaves identically. The unused dependencies are gone, and the Python polars range is declared and tested in CI.
**Depends on**: Phase 22 (migrates the full expression set, including the surface added in Phases 21 and 22)
**Requirements**: STACK-01, STACK-02, STACK-03
**Success Criteria** (what must be TRUE):

  1. The crate builds on pyo3-polars 0.28 / polars 0.55 / pyo3 0.29 / numpy 0.29. The full pytest and Rust suites pass with no expected-value edits, including every R-reference test from Phases 18–22. Plugin expressions and PyModels behave identically.
  2. `Cargo.toml` no longer lists `rand`, `rand_chacha`, `polars-arrow` or `thiserror`, and declares `rust-version = "1.87"`. It keeps faer 0.23 and statrs 0.18, so `cargo tree` shows one faer and one statrs version, shared with the anofox crates. The no-default-features pyo3-absence CI guard stays green.
  3. `pyproject.toml` declares a tested Python `polars` range. A CI job installs the declared minimum polars and runs the test suite green, alongside the existing latest-polars jobs.
  4. The Python 3.9–3.12 × Linux/macOS/Windows CI matrix and the abi3-py39 wheel build stay green on the new stack.

**Notes**: This is the last code-migration phase. The expressions added in Phases 21–22 were written against polars 0.52 and are migrated here with the 323 existing `#[polars_expr]` functions. The known pyo3 edit is `downcast` → `cast` (`py_penalized_glm.rs:31`). If the polars 0.55 changes turn out large (e.g. Series→Column plugin signatures), fall back to stepping pyo3-polars 0.26 → 0.27 → 0.28.
**Plans**: TBD

### Phase 24: Docs & Skills Refresh

**Goal**: Users reading the docs and Claude agents reading the skills see the complete v0.10.0 surface and every behaviour change, all verified against the live package built on the final stack.
**Depends on**: Phase 23 (docs, stubs and skill examples are verified against the final build)
**Requirements**: DOC-01, SKILL-07
**Success Criteria** (what must be TRUE):

  1. mkdocs pages, docstrings and the README cover every builder, class, kwarg and result field added in Phases 21–22, checked against the live import rather than docstrings. `mkdocs build --strict` is green.
  2. The CHANGELOG `[0.10.0]` "Behaviour changes" section lists every changed output from Phases 18–20, each with its R reference:
     - From the anofox-statistics upgrade: the pairwise-t CI, the SPA p-value swap, MCS, the ICC(2) CI, the one-sided t CI, `semi_partial_cor` and the HF ε.
     - From the anofox-regression upgrade: penalized-GLM `model_type`/`vcov`, the 0.8 regression numerics, VIF ∞/NaN and P-spline AIC/BIC.
  3. `.pyi` stubs are regenerated from the final build, and the drift guard is green.
  4. The five Claude Code skills document the new surface and the behaviour changes. `scripts/verify_skill_examples.py` reports 0 FAIL against the live package, and the bundled `_plugin/` copy has no drift.

**Plans**: TBD

### Phase 25: Release 0.10.0

**Goal**: polars-statistics 0.10.0 is live on production PyPI and works from a clean install.
**Depends on**: Phase 24
**Requirements**: REL-09, REL-10
**Success Criteria** (what must be TRUE):

  1. `scripts/check_version_alignment.py --expect 0.10.0` passes across all 18 version sources.
  2. The GitHub release `v0.10.0` publishes wheels for all 5 platforms plus the sdist to production PyPI through the OIDC trusted-publishing pipeline, with no stored tokens.
  3. In a clean venv on Python 3.9 and on 3.12, with the latest polars, `pip install polars-statistics==0.10.0` gives:
     - `ps.__version__ == "0.10.0"`.
     - SPA output with the corrected p-value columns.
     - An OLS fit with working `predict_se`.
     - A working `group_by` aggregation.

**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in order: 18 → 19 → 20 → 21 → 22 → 23 → 24 → 25. Phase 21 depends only on
Phase 18, so it can be pulled forward if needed.

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 18. anofox-statistics 0.10.0 Upgrade | v0.10.0 | 0/TBD | Not started | - |
| 19. anofox-regression 0.8 Numerics | v0.10.0 | 0/TBD | Not started | - |
| 20. anofox-regression 0.12.2 API Migration | v0.10.0 | 0/TBD | Not started | - |
| 21. New Statistics Surface | v0.10.0 | 0/TBD | Not started | - |
| 22. New Regression Surface | v0.10.0 | 0/TBD | Not started | - |
| 23. Polars / PyO3 Stack Upgrade | v0.10.0 | 0/TBD | Not started | - |
| 24. Docs & Skills Refresh | v0.10.0 | 0/TBD | Not started | - |
| 25. Release 0.10.0 | v0.10.0 | 0/TBD | Not started | - |

## Milestone History

| Milestone | Phases | Plans | Status | Shipped |
|-----------|--------|-------|--------|---------|
| v0.6.0 API Parity & Release | 1-7 | 25 | Complete | 2026-08-19 |
| v0.7.0 Ergonomics, Adoption & Documentation | 8-12 | 5 | Complete | 2026-09-20 |
| v0.8.0 Claude Code Skills Plugin | 13-17 (incl. 16.1-16.4) | 44 | Complete | 2026-10-16 |
| 0.9.0 bug-fix release (out of milestone) | — | quick tasks | Complete | 2026-11-11 |
| v0.10.0 Upstream Wave 3 | 18-25 | TBD | In progress | - |
