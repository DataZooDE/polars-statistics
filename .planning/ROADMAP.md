# Roadmap: polars-statistics

## Milestones

- ✅ **v0.6.0 API Parity & Release** - Phases 1-7 (shipped 2026-08-19)
- ✅ **v0.7.0 Ergonomics, Adoption & Documentation** - Phases 8-12 (shipped 2026-09-20)
- ✅ **v0.8.0 Claude Code Skills Plugin** - Phases 13-17 incl. 16.1-16.4 (shipped 2026-10-16)
- 📋 **v0.9.0 Upstream Wave 3** - candidate (UPST-11, not yet planned; start with `/gsd-new-milestone`)
- 📋 **v2.0 Forecast Integration** - deferred (FCST-01, not yet planned)

## Phases

Phase numbering is continuous across milestones. The next milestone starts at Phase 18.

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

## Progress

| Milestone | Phases | Plans | Status | Shipped |
|-----------|--------|-------|--------|---------|
| v0.6.0 API Parity & Release | 1-7 | 25 | Complete | 2026-08-19 |
| v0.7.0 Ergonomics, Adoption & Documentation | 8-12 | 5 | Complete | 2026-09-20 |
| v0.8.0 Claude Code Skills Plugin | 13-17 (incl. 16.1-16.4) | 44 | Complete | 2026-10-16 |
