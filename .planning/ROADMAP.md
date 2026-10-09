# Roadmap: polars-statistics

## Milestones

- ✅ **v0.6.0 API Parity & Release** - Phases 1-7 (shipped 2026-08-19)
- ✅ **v0.7.0 Ergonomics, Adoption & Documentation** - Phases 8-12 (shipped 2026-09-20)
- 🚧 **v0.8.0 Claude Code Skills Plugin** - Phases 13-17 (in progress)
- 📋 **v2.0 Forecast Integration** - deferred (FCST-01, not yet planned)

## Overview

v0.6.0 delivered full API parity with the backing `anofox-*` crates; v0.7.0 made that
surface best-in-class for humans (typed, ergonomic, documented). v0.8.0 shifts the audience
from the human reader to the **AI agent**: it packages the existing v0.7.0 surface as
installable Claude Code skills — a plugin + marketplace mirroring `anofox-forecast` — so
Claude can discover and apply polars-statistics correctly from accurate, version-pinned
cheat-sheets. The work moves scaffolding-first, content-second, verify-third, package-fourth,
release-last: first the plugin/marketplace shell is stood up and installs cleanly, then the
five capability-family `SKILL.md` cheat-sheets are authored (with examples checked live as
they are written), then a dedicated accuracy pass runs every example against the real API and
audits each skill's auto-surfacing description, then the skill files are bundled into the wheel
via maturin package data, and finally 0.8.0 is bumped across all version sources and published
to production PyPI via the existing OIDC pipeline. This milestone adds documentation/tooling,
not statistical capability.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.
Phase numbering is continuous across milestones — v0.8.0 continues from Phase 13.

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

### 🚧 v0.8.0 Claude Code Skills Plugin (In Progress)

**Milestone Goal:** Package the full polars-statistics API as installable Claude Code skills —
a plugin + marketplace mirroring `anofox-forecast` — with content verified accurate against the
real API, bundled into the wheel, and shipped as 0.8.0 on PyPI.

- [x] **Phase 13: Plugin & Marketplace Scaffolding** - Stand up the `.claude-plugin/marketplace.json` + `plugins/polars-statistics/.claude-plugin/plugin.json` shell that installs cleanly in Claude Code (completed 2026-09-21)
- [x] **Phase 14: Skill Cheat-Sheet Authoring** - Author the five capability-family `SKILL.md` cheat-sheets with correct frontmatter and consistent structure, examples checked live while writing (completed 2026-09-21)
- [x] **Phase 15: Accuracy Verification Pass** - Run every skill example against the real API and audit each skill's description so Claude auto-surfaces the right one (completed 2026-09-21)
- [ ] **Phase 16: Wheel Bundling** - Bundle the skill files into the built wheel via maturin package data so they ship on `pip install`
- [ ] **Phase 16.1: Upstream Crate Refresh** (INSERTED) - Bump to anofox-statistics 0.4.6 / anofox-regression 0.5.16, adapt to R-matching fixes, expose new API, update stubs/docs/skills
- [ ] **Phase 16.2: Penalized GLM and AFT Survival Regression** (INSERTED) - Expose anofox-regression 0.5.16's penalized GLM engine and AFT survival regression; add estimate/CI to the Mann-Whitney/Wilcoxon expressions
- [ ] **Phase 17: Release 0.8.0** - Bump 0.7.0→0.8.0 across every version source and publish to production PyPI via the OIDC pipeline

## Phase Details

### Phase 13: Plugin & Marketplace Scaffolding

**Goal**: A valid, discoverable Claude Code plugin shell for polars-statistics exists and installs cleanly — the marketplace lists it and the plugin manifest declares its identity — mirroring the anofox-forecast layout.
**Depends on**: Phase 12 (v0.7.0 surface complete and shipped)
**Requirements**: PLUG-01, PLUG-02, PLUG-03
**Success Criteria** (what must be TRUE):

  1. A root `.claude-plugin/marketplace.json` lists the `polars-statistics` plugin (name, source path, description, keywords) in the anofox-forecast marketplace format
  2. `plugins/polars-statistics/.claude-plugin/plugin.json` declares name, description, version, author, homepage, repository, and license
  3. The plugin installs in Claude Code with valid JSON, resolvable skill directory paths, and is discoverable via the marketplace

**Plans**: 1 plan

- [x] 13-01-PLAN.md — Create both manifests (marketplace.json + plugin.json) and the skills/ tree so the plugin shell installs and resolves end-to-end

### Phase 14: Skill Cheat-Sheet Authoring

**Goal**: All five capability-family cheat-sheets exist as `SKILL.md` files that a Claude agent can read to apply the full polars-statistics API — each covering its surface with accurate signatures and runnable examples, correct pinned frontmatter, and a consistent structure.
**Depends on**: Phase 13 (skills live inside the plugin's `skills/` tree)
**Requirements**: SKILL-01, SKILL-02, SKILL-03, SKILL-04, SKILL-05, SKILL-06
**Success Criteria** (what must be TRUE):

  1. `polars-statistics-hypothesis-tests/SKILL.md` covers the expression-API tests (parametric, nonparametric, distributional, correlation, categorical, modern MMD/energy, TOST/equivalence) with signatures and worked examples
  2. `polars-statistics-anova/SKILL.md` covers one-way, two-way, and repeated-measures ANOVA plus ICC with worked examples
  3. `polars-statistics-regression/SKILL.md` covers the 40+ PyModel classes, the regression expressions, and the R-style formula parser
  4. `polars-statistics-results/SKILL.md` covers `to_dict()`/`summary()`/`__repr__`, unnest/struct helpers, contextual errors, and the typed `.pyi` API; and `polars-statistics-getting-started/SKILL.md` covers install, the model-selection matrix, and sklearn migration
  5. Every `SKILL.md` carries correct frontmatter (`name`, `description`, `version` pinned to the package version, `user-invocable: false`) and a consistent cheat-sheet structure

**Plans**: 2/5 plans executed

- [x] 14-01-PLAN.md — getting-started SKILL.md (tracer: proves frontmatter + structure + spot-check)
- [x] 14-02-PLAN.md — hypothesis-tests SKILL.md (parametric/nonparametric/distributional/correlation/categorical/modern/TOST)
- [x] 14-03-PLAN.md — anova SKILL.md (one-way/two-way/repeated-measures ANOVA + ICC)
- [x] 14-04-PLAN.md — regression SKILL.md (PyModel families, regression expressions, formula parser)
- [x] 14-05-PLAN.md — results SKILL.md (struct helpers, PyModel methods, error taxonomy, .pyi)

### Phase 15: Accuracy Verification Pass

**Goal**: The skills are trustworthy for an agent — every code example provably runs against the real API and every skill's description routes Claude to the right cheat-sheet for a task.
**Depends on**: Phase 14 (verifies the authored content)
**Requirements**: VERIFY-01, VERIFY-02
**Success Criteria** (what must be TRUE):

  1. Every code example in every skill has been executed against the shipped API, with signatures, kwargs, and return shapes confirmed against the real surface (not invented)
  2. Each skill's `description` accurately reflects its covered surface, so a task prompt auto-surfaces the correct skill
  3. Any signature, kwarg, or return-shape drift found during verification is corrected in the skill content before the phase closes

**Plans**: 1 plan

- [x] 15-01-PLAN.md — Build reusable harness executing every skill example against the live API + fix drift in-place (VERIFY-01); document representative task-prompt→skill routing mapping and refine descriptions (VERIFY-02)

### Phase 16: Wheel Bundling

**Goal**: The skill files travel with the package — after installing the wheel, the plugin and all skills are present on disk.
**Depends on**: Phase 15 (verified skills are the artifacts being bundled)
**Requirements**: PKG-01
**Success Criteria** (what must be TRUE):

  1. maturin package-data configuration includes the plugin and skill files so they are packaged into the built wheel
  2. After `pip install polars-statistics` (from the built wheel), the marketplace, `plugin.json`, and every `SKILL.md` are present in the installed package
  3. The bundled-skills packaging does not break the existing wheel build or CI gates

**Plans**: 1 plan

- [ ] 16-01-PLAN.md — Copy the plugin tree under `python/polars_statistics/_plugin/` (maturin python-source auto-include), add a drift + clean-venv install verifier, prove PKG-01 end-to-end

### Phase 16.1: Upstream Crate Refresh (INSERTED)

**Goal**: polars-statistics runs on anofox-statistics 0.4.6 and anofox-regression 0.5.16. Every new upstream capability is exposed through the Polars/Python API, the R-matching behaviour changes are reflected in tests and docs, and the skills that ship in the wheel describe the new surface.
**Depends on**: Phase 15 (updates the verified skills). Phase 16 must re-sync its bundled `_plugin/` copy afterwards; its drift guard enforces that.
**Requirements**: UPST-01, UPST-02, UPST-03, UPST-04
**Success Criteria** (what must be TRUE):

  1. `Cargo.toml`/`Cargo.lock` pin anofox-statistics 0.4.6 and anofox-regression 0.5.16; clippy `-D warnings`, cargo fmt, ruff and the full pytest suite are green
  2. Tests that encoded pre-fix numbers are updated to the R-matching values (Tweedie AIC/BIC NaN for non-closed-form powers, linear-solver AIC/BIC, no-intercept R², Ridge sandwich SEs with None inference for λ>0, RANSAC stats, NegBin SEs/log-likelihood, Gamma/Tweedie dispersion, Shapiro-Wilk n≤11, Kendall ties, binomial/proportion/Fisher intervals, exact Mann-Whitney, NaN/Inf input errors instead of panics), with R references where upstream supplies them
  3. New upstream API is exposed as expressions and/or PyModel options, following the existing patterns: Cohen's d (two-sample/paired/one-sample), rank-biserial effect sizes, Jarque-Bera, bootstrap CIs (`bootstrap_mean_ci`), `fisher_exact_conditional`, `conf_level` for binomial/proportion/Fisher tests, GLMM z/p-values and Wald CIs, Huber and RANSAC inference (`compute_inference`/`confidence_level`), NegBin `estimate_dispersion`, RLS `initial_p_diagonal`
  4. `.pyi` stubs (drift guard green), docstrings, mkdocs pages, CHANGELOG (including a "behaviour changes" section) and the skill cheat-sheets cover the new surface, and the Phase 15 skill-example harness passes

**Plans**: 3/9 plans executed
**Wave 1**

- [x] 16.1-01-PLAN.md — Bump crates (exact 0.4.6 / 0.5.16), fix Tweedie smoke tests, pin every R-matching behaviour change + non-finite handling with R-reference tests (Wave 1)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 16.1-02-PLAN.md — Effect sizes (Cohen's d family, rank-biserial family) + categorical conf_level + fisher_exact_conditional (Wave 2)
- [x] 16.1-03-PLAN.md — NegBin estimate_dispersion/dispersion, RLS initial_p_diagonal (PyModel + all expressions + rlib), GLMM Wald inference (Wave 2)
- [ ] 16.1-09-PLAN.md — anofox-statistics 0.4.5/0.4.6 parity: Mann-Whitney/Wilcoxon Hodges-Lehmann CI vs R (126 cases), exact-distribution gating, mcnemar_exact at large counts (Wave 2)

**Wave 3** *(blocked on Wave 2 completion)*

- [ ] 16.1-04-PLAN.md — Jarque-Bera, bootstrap_mean_ci / bootstrap_ci, rlib smoke for all new/changed statistics *_fit (Wave 3)
- [ ] 16.1-05-PLAN.md — Huber/RANSAC inference, RANSAC.inlier_fit(), TheilSen univariate_pairwise, variance_factor/leverage_new, Ridge inference getters (Wave 3)

**Wave 4** *(blocked on Wave 3 completion)*

- [ ] 16.1-06-PLAN.md — Skill cheat-sheets + ROUTING.md for the new surface, harness green (Wave 4)
- [ ] 16.1-07-PLAN.md — mkdocs API pages (new effect-sizes page, outputs, statistics + regression pages) (Wave 4)
- [ ] 16.1-08-PLAN.md — CHANGELOG [Unreleased] incl. behaviour changes + Rust API, README, API_REFERENCE, validation page (Wave 4)

### Phase 16.2: Penalized GLM and AFT Survival Regression (INSERTED)

**Goal**: The two model families added in anofox-regression 0.5.16 (the penalized GLM engine and accelerated failure time survival regression) are available as Polars expressions and Python model classes. The Mann-Whitney / Wilcoxon expressions also return their Hodges-Lehmann estimate and confidence interval. All of it is documented, R-validated and in the skills.
**Depends on**: Phase 16.1 (crate versions and the refreshed docs/skills baseline)
**Requirements**: UPST-05, UPST-06, UPST-07
**Success Criteria** (what must be TRUE):

  1. Penalized GLM is exposed as a PyModel (family: poisson/binomial/logistic/negative binomial/gamma/tweedie with links; per-coefficient normal/laplace priors; ridge `lambda`; `vcov_type` laplace/sandwich/naive; offsets; inference) and as `#[polars_expr]` expressions for group_by/over. Results match upstream's R references (glm, MASS::glm.nb, negative.binomial(θ)).
  2. AFT survival regression is exposed as a PyModel (Weibull, exponential, log-normal, log-logistic; right censoring via an event indicator; inference; `predict_quantile`/`predict_median`/`predict_survival`/`predict_cdf`) and as expressions. Results match `survival::survreg` references.
  3. `ps.mann_whitney_u` / `ps.wilcoxon_signed_rank` expression outputs gain trailing `estimate`, `ci_lower` and `ci_upper` fields (additive), so `conf_level` takes effect.
  4. `.pyi` stubs (drift guard green), docstrings, mkdocs, CHANGELOG and the regression/hypothesis skill cheat-sheets cover the new surface. The skill-example harness and all CI gates (clippy, fmt, ruff on new files, pytest) are green.

**Plans**: 7 plans
**Wave 1**

- [ ] 16.2-01-PLAN.md — Penalized GLM engine: PyModel + fit expression + formula, R-validated (Wave 1)
- [ ] 16.2-02-PLAN.md — Mann-Whitney/Wilcoxon expression estimate + CI, schema lock updated (Wave 1)

**Wave 2** *(blocked on Wave 1 completion)*

- [ ] 16.2-03-PLAN.md — Penalized GLM summary/predict expressions + formula variants + rlib (Wave 2)

**Wave 3** *(blocked on Wave 2 completion)*

- [ ] 16.2-04-PLAN.md — AFT survival regression: PyModel + expressions vs survreg (Wave 3)

**Wave 4** *(blocked on Wave 3 completion)*

- [ ] 16.2-05-PLAN.md — mkdocs API pages (Wave 4)
- [ ] 16.2-06-PLAN.md — CHANGELOG / README / API_REFERENCE / validation page (Wave 4)
- [ ] 16.2-07-PLAN.md — Skills + ROUTING + full phase gate (Wave 4)

### Phase 17: Release 0.8.0

**Goal**: polars-statistics 0.8.0 — carrying the plugin, the five verified skills, and the wheel bundling — is built, published to production PyPI via the existing OIDC pipeline, and confirmed installable with skills present.
**Depends on**: Phase 16 (release ships the bundled skills)
**Requirements**: REL-07, REL-08
**Success Criteria** (what must be TRUE):

  1. Version reads 0.8.0 across all sources — `Cargo.toml`, `pyproject.toml`, `python/polars_statistics/__init__.py` `__version__`, `plugin.json`, and every `SKILL.md` version field — all aligned (the v0.6.0 stuck-release misaligned-version failure mode does not recur)
  2. sdist and platform wheels are built and published to production PyPI via the GitHub Actions OIDC pipeline (no hardcoded tokens)
  3. A post-release smoke check confirms the published 0.8.0 installs, imports cleanly, and has the skill files present in the installed package

**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in order: 13 → 14 → 15 → 16.1 → 16.2 → 16 → 17 (16.1/16.2 update the skills before 16 bundles them into the wheel)

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Dependency Modernization | v0.6.0 | 2/2 | Complete | 2026-08-11 |
| 2. Public API Audit | v0.6.0 | 1/1 | Complete | 2026-08-11 |
| 3. Statistics API Parity | v0.6.0 | 4/4 | Complete | 2026-08-12 |
| 4. Regression API Parity | v0.6.0 | 6/6 | Complete | 2026-08-12 |
| 5. Documentation | v0.6.0 | 4/4 | Complete | 2026-08-12 |
| 6. Testing & Validation | v0.6.0 | 5/5 | Complete | 2026-08-12 |
| 7. Release 0.6.0 | v0.6.0 | 3/3 | Complete | 2026-08-19 |
| 8. Type Safety | v0.7.0 | 1/1 | Complete | 2026-09-20 |
| 9. Result Ergonomics | v0.7.0 | 1/1 | Complete | 2026-09-20 |
| 10. Errors & API Consistency | v0.7.0 | 1/1 | Complete | 2026-09-20 |
| 11. Documentation & Adoption | v0.7.0 | 1/1 | Complete | 2026-09-20 |
| 12. Release 0.7.0 | v0.7.0 | 1/1 | Complete | 2026-09-20 |
| 13. Plugin & Marketplace Scaffolding | v0.8.0 | 1/1 | Complete    | 2026-09-21 |
| 14. Skill Cheat-Sheet Authoring | v0.8.0 | 5/5 | Complete    | 2026-09-21 |
| 15. Accuracy Verification Pass | v0.8.0 | 1/1 | Complete    | 2026-09-21 |
| 16. Wheel Bundling | v0.8.0 | 0/TBD | Not started | - |
| 17. Release 0.8.0 | v0.8.0 | 0/TBD | Not started | - |
</content>
</invoke>
