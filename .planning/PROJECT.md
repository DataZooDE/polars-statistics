# polars-statistics

## What This Is

A Rust + Python (PyO3) library that exposes comprehensive statistical hypothesis
testing and regression modeling as native Polars expressions and Python classes.
Users write `df.group_by(...).agg(ps.ols("y", "x1", "x2"))` or
`from polars_statistics import OLS` and get R-validated statistics with zero-copy
numpy↔faer data transfer. It is distributed as a PyPI wheel for Python 3.9+.

As of v0.7.0 the library exposes full parity with the backing `anofox-*` crates —
every public statistical capability is reachable, type-safe, ergonomic, documented,
tested, and shipped on PyPI. v0.8.0 added the **AI agent** as an audience. The whole
API surface ships as installable Claude Code skills: a plugin and marketplace that
mirror `anofox-forecast` and are bundled inside the wheel, so Claude can discover and
apply polars-statistics from accurate, version-pinned cheat-sheets. v0.8.0 also brought
the backing crates to anofox-statistics 0.4.7 / anofox-regression 0.5.22. That added
penalized GLM, AFT survival regression, post-hoc tests, augment diagnostics, GLMM
families and EB shrinkage.

## Core Value

Every public statistical capability in the backing `anofox-*` crates is exposed
through the Polars/Python API — correctly (post-bugfix), documented, and tested —
and shipped to users via PyPI.

## Current State

**Shipped:** v0.9.0 bug-fix release (2026-11-11), out of milestone. It fixed GH #51: the
rlib no longer pulls in pyo3, and pyo3-polars is now optional. It also fixed GH #52:
every expression upcasts numeric inputs to Float64/UInt32. 0.9.0 is on PyPI; tag
`v0.9.0`.

- Upstream crates are still frozen at anofox-statistics 0.4.7 / anofox-regression
  0.5.22. Latest upstream (2026-11-06): anofox-statistics 0.10.0 / anofox-regression
  0.12.2.
- Stack: polars 0.52 / pyo3-polars 0.25 / pyo3 0.26 / numpy 0.26. The 0.9.0 wheel was
  verified working with Python polars 1.34, 1.39 and 2.1.
- 18 version sources are aligned at 0.9.0 (`scripts/check_version_alignment.py`).

## Current Milestone: v0.10.0 Upstream Wave 3

**Goal:** Move onto the latest anofox crates (anofox-statistics 0.10.0, anofox-regression
0.12.2) and the current Polars/PyO3 stack while keeping R parity, expose the small new
upstream surface, and ship 0.10.0 to PyPI.

**Target features:**
- anofox-statistics 0.4.7 → 0.10.0. Fix the posthoc test and fixture. Document and
  R-validate the semantic changes: the SPA p-value column swap, MCS, the ICC(2) CI, the
  one-sided t CI, semi_partial_cor and the HF epsilon. Add the cheap API: pairwise
  conf_level, p_adjust_n, ttest_1samp, SPA p_value_lower, the MCS fields, the bootstrap
  method kwarg and the MMD median heuristic.
- anofox-regression 0.5.22 → 0.8.0, the numerics step. Construct Augment without a
  struct literal. Set penalized-GLM model_type to "glm". Remove the double vcov φ
  scaling. Raise the LmDynamic max_models default. Re-validate the R fixtures against
  the 0.8 algorithm changes.
- anofox-regression → 0.9.0 → 0.12.2, the API-break step. Renames, IterativeFit,
  ColRef returns, non_exhaustive enums, error types, and the VIF / P-spline AIC
  semantics.
- Polars/PyO3 stack. pyo3-polars 0.28 / polars 0.55 / pyo3 0.29 / numpy 0.29 (pinned
  by pyo3-polars 0.28). Drop unused deps (rand, rand_chacha, polars-arrow, thiserror).
  MSRV 1.87. Add a tested Python polars range and a min-version CI job. Keep faer 0.23
  and statrs 0.18: both anofox crates still require them.
- Small new upstream surface only: kwargs, result fields and diagnostics. Examples:
  Breusch-Pagan/White, cluster/HC vcov, predict_se, two-sample KS/AD/CvM,
  ACF/PACF/Ljung-Box/Durbin-Watson.
- Docs, skills, CHANGELOG with a Behaviour-changes section, and the v0.10.0 release to
  PyPI.

**Out of scope for this milestone:** large new families (Cox PH, ordinal/multinomial,
GLS, the rms toolkit, drift, PCA/monitoring, classification metrics, survival curves).
They are deferred to later milestones.

**Research:** version-delta analysis in `.planning/research/UPSTREAM-DELTA.md`.

## Requirements

### Validated

<!-- Inferred from the existing codebase (.planning/codebase/) — shipped in v0.5.0. -->

- ✓ Polars expression API for statistical tests across parametric, nonparametric, distributional, correlation, categorical, modern (MMD/energy), forecast, and TOST/equivalence categories — existing
- ✓ 40+ regression & test PyModel classes (OLS, Ridge, ElasticNet, WLS, RLS, BLS, PLS, Quantile, Isotonic, Huber, Logistic, Poisson, NegativeBinomial, Tweedie, Probit, Cloglog, ALM, LmDynamic, Aid, StationaryBootstrap, CircularBlockBootstrap) — existing
- ✓ R-style formula parser (`y ~ x1 + x2 + x1:x2`) with interaction expansion — existing
- ✓ Zero-copy numpy↔faer array conversion (`ToFaer`/`IntoNumpy` traits) — existing
- ✓ Dual-build crate (cdylib for Python + rlib for Rust consumers), `python` feature gate — existing
- ✓ CI matrix (Python 3.9–3.12 × Linux/macOS/Windows), clippy/fmt/ruff gates, coverage — existing
- ✓ GitHub Actions PyPI + TestPyPI publish pipeline with OIDC trusted publishing — existing
- ✓ mkdocs (Material) documentation published to GitHub Pages — existing
- ✓ `anofox-statistics` bumped 0.4.1 → 0.4.2 — Phase 1 (clean build, existing tests green)
- ✓ `anofox-regression` bumped 0.5.4 → 0.5.13, inheriting the column-pivot correctness fix — Phase 1 (zero wrapper reconciliation; 3 new column-pivot tests confirm the fix is active; Rust 15 + pytest 457 passing)
- ✓ Authoritative public-API gap list for both crates (`02-API-AUDIT.md`) — Phase 2 (statistics: 3 ANOVA fns + energy nD + ICC stub; regression: Gamma/GLMM/PSpline/FactorSummary + HC-extend + 6 solvers + minor diagnostics; all 21 known candidates resolved; HC confirmed already-PARTIAL for OLS)
- ✓ All unexposed `anofox-statistics` functions exposed as Polars expressions — Phase 3 (one/two-way + repeated-measures ANOVA, energy_distance_nd, real matrix-input ICC replacing the NaN stub; 479 pytest + 15 rust_api green; 3 critical ANOVA factor-encoding bugs caught in review and fixed with regression tests)
- ✓ All unexposed `anofox-regression` capabilities exposed (full parity) — Phase 4 (10 new PyModels: Gamma, GLMM, PSpline, TheilSen, RANSAC, BayesianRidge, ARD, LARS, PassiveAggressive, MomentAccumulator; Ridge/WLS HC; 5 GLM-diagnostic expressions; OLS/Ridge fit_from_accumulator. 582 pytest + 15 rust_api green; 5 critical + 6 warning review findings fixed. Known limitation: weighted HC for WLS raises NotImplementedError pending crate support.)
- ✓ Full API documentation for the new surface — Phase 5 (Python docstrings w/ runnable examples, mkdocs reference pages, Rust /// comments, CHANGELOG 0.6.0 with known limitations)
- ✓ Value + shape tests for all new API, validated vs scipy/statsmodels/analytic references — Phase 6 (602 pytest + 17 rust_api green; ICC(3,1)=0.715 vs Shrout&Fleiss, GLMM slope vs truth, Gamma vs statsmodels, fit_from_accumulator==batch; full OS/py matrix delegated to GH Actions)
- ✓ Bumped `polars-statistics` 0.5.0 → 0.6.0 and published to production PyPI — Phase 7 + quick task 260819-jkj (0.6.0 live on PyPI; publish had failed on a stale 0.5.0-named wheel because the version bump had only reached Cargo.lock — closed by aligning Cargo.toml/pyproject.toml/`__version__`, re-pointing the tag, and re-running the OIDC pipeline)
- ✓ Type stubs (`.pyi`) + `py.typed` marker for all Rust-bound classes and expression builders, with a CI drift guard — Phase 8 / v0.7.0 (608 pytest green; stubs verified shipped in the wheel)
- ✓ Ergonomic results — `.to_dict()`/`.summary()`/`__repr__` on every model & test class plus `unnest`/`struct_to_dict` helpers for the expression Struct surface — Phase 9 / v0.7.0 (642 pytest green)
- ✓ Contextual, actionable errors — not-fitted names the model + `.fit(...)`; shape/rank-deficient inputs raise clear `ValueError`s — Phase 10 / v0.7.0
- ✓ API consistency — `with_intercept`→`add_intercept` deprecation (back-compat `FutureWarning`) and uniform sklearn-style `fit`/`predict`/`score` — Phase 10 / v0.7.0 (671 pytest green)
- ✓ Documentation & adoption — cookbook + runnable examples for the robust/sparse, GLM/smoother/streaming, and ANOVA families; model-selection matrix; migration guide; sklearn-migration guide; adoption-focused README — Phase 11 / v0.7.0 (`mkdocs build --strict` green)
- ✓ Shipped `polars-statistics` 0.7.0 to production PyPI via the OIDC pipeline — Phase 12 / v0.7.0 (all version sources aligned; publish run green; clean-venv smoke check confirms 0.7.0 + typed API + ergonomics + errors live)
- ✓ Claude Code plugin + marketplace scaffolding (`.claude-plugin/marketplace.json`, `plugins/polars-statistics/.claude-plugin/plugin.json`) — v0.8.0 (PLUG-01..03)
- ✓ 5 version-pinned `SKILL.md` cheat-sheets covering the full API, `user-invocable: false` — v0.8.0 (SKILL-01..06)
- ✓ Skill content verified against the live API (example harness 0 FAIL; ROUTING.md 23/23 prompts route correctly) — v0.8.0 (VERIFY-01/02)
- ✓ Skills bundled into the PyPI wheel at `polars_statistics/_plugin/`, guarded against drift in CI — v0.8.0 (PKG-01)
- ✓ Upstream refresh to anofox-statistics 0.4.7 / anofox-regression 0.5.22: effect sizes, Jarque-Bera, bootstrap CIs, conditional Fisher, Huber/RANSAC/GLMM inference, PenalizedGLM, AFT survival, Hodges-Lehmann CIs, post-hoc tests + p_adjust/ptukey/qtukey, augment, model_info, GLMM NegBin/Gamma/Tweedie + offsets + random slopes, alias_tolerance, eb_shrink. All R-validated — v0.8.0 (UPST-01..10)
- ✓ `ruff check .` green repo-wide and enforced in CI — v0.8.0 (QUAL-01)
- ✓ Bumped 0.7.0 → 0.8.0 across 18 version sources and published to production PyPI via OIDC; post-release smoke run on 3.9/3.12 — v0.8.0 (REL-07/08)

### Active

<!-- v0.10.0 Upstream Wave 3 — detailed REQ-IDs in REQUIREMENTS.md -->

- [ ] UPST-11: upgrade to anofox-statistics 0.10.0 / anofox-regression 0.12.2 and expose the small new surface (v0.10.0, see REQUIREMENTS.md)
- [ ] STACK-01: upgrade to pyo3-polars 0.28 / polars 0.55 / pyo3 0.29 (v0.10.0)
- [ ] FCST-01: evaluate exposing `anofox-forecast` capabilities

### Out of Scope

- New statistical methods not present in the `anofox-*` crates — this milestone is exposure/parity, not novel algorithm work
- `anofox-forecast` integration — separate crate, not requested
- Changes to the FFI architecture or the numpy↔faer transfer strategy — the existing bridge stays
- Reworking already-exposed API surface beyond what the version bump requires
- Slash-command / `user-invocable: true` skills — skills are auto-surfaced reference cheat-sheets, like anofox-forecast's
- Standalone skills repo — skills live in this repo under `plugins/polars-statistics/` and are mirrored into the wheel

## Context

- **Backing crates:** `anofox-regression` and `anofox-statistics` (from github.com/sipemu), both registry deps in `Cargo.toml` (lines 64, 67). The regression jump spans 0.5.5–0.5.13 and includes GLMM (0.5.12), P-spline (0.5.10), streaming moments (0.5.9), and the 0.5.13 correctness fixes.
- **Correctness driver:** anofox-regression 0.5.13 fixes a column-pivot unpermute bug that silently scrambled OLS/WLS/NNLS coefficients on differently-scaled designs. The wrapper currently pins 0.5.4, so it is exposed to this bug today — the bump is a correctness fix, not just a feature add.
- **Codebase map:** Full map exists at `.planning/codebase/` (STACK, ARCHITECTURE, STRUCTURE, CONVENTIONS, TESTING, INTEGRATIONS, CONCERNS).
- **Two exposure surfaces per capability:** most statistics functions are wrapped as Polars expressions (`src/expressions/*.rs` → `python/polars_statistics/exprs/*.py`); regression models are additionally wrapped as PyModel classes (`src/pymodels/*.rs`). New API may need both.
- **Release mechanics:** the version lives in 18 sources: `Cargo.toml`, `pyproject.toml`, `__version__`, `uv.lock`, both plugin/marketplace manifest copies, and every SKILL.md copy. They are all at 0.8.0 and checked by `scripts/check_version_alignment.py` and `tests/test_version_alignment.py`. Publishing is tag/release-triggered via `.github/workflows/publish.yml` using OIDC trusted publishing (no tokens).
- **Skills plugin:** the source of truth is `plugins/polars-statistics/`. `python/polars_statistics/_plugin/` is a byte-identical copy that ships in the wheel, and `scripts/verify_wheel_bundling.py --drift-only` runs in CI. Any API change must also update the skills, plus the harness `scripts/verify_skill_examples.py`, which today runs locally and at release, not in CI.
- **Known tech debt (accepted at v0.8.0 close):** on Python 3.9, `typing.get_type_hints()` resolves for only 11 of 204 public builders after the PEP 604 modernisation (T-16.4-03). `WLS.hc_inference` still raises `NotImplementedError`. `conda/meta.yaml` is still at 0.4.0.

## Constraints

- **Tech stack**: Rust 2021 + PyO3 0.26 + Polars 0.52 (→ PyO3 0.29 + Polars 0.55 in v0.10.0); new API must follow the existing `#[polars_expr]` / PyModel patterns and output-struct schemas — Why: consistency and native Polars integration.
- **Validation**: New tests validated against R where the crates supply references — Why: the library's core promise is "validated against R."
- **CI gates**: clippy `-D warnings`, cargo fmt, ruff (line length 100, py39 target) must stay green — Why: existing merge gates.
- **Compatibility**: Maintain Python 3.9–3.12 support across Linux/macOS/Windows wheels — Why: existing distribution matrix.
- **Release**: Production PyPI publish via the existing GH Actions pipeline (OIDC, no hardcoded tokens) — Why: established, secure release path.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Bump both `anofox-statistics` and `anofox-regression` (not statistics only) | User chose full modernization; regression 0.5.13 also carries a correctness fix and major new API | ✓ Done — Phase 1 (zero reconciliation; column-pivot fix confirmed active by 3 new tests) |
| Expose all unexposed functions (full API parity) | User chose completeness over a targeted subset | ✓ Done — Phases 3+4 (all statistics + regression gaps exposed) |
| Production PyPI publish (v0.6.0) | User wanted a real release, not TestPyPI/prepare-only | ✓ Done — 0.6.0 live on PyPI |
| Target release version 0.6.0 | New API surface was additive → minor bump under semver | ✓ Done |
| v0.7.0 milestone = ergonomics + adoption + docs (not new algorithms) | v0.6.0 delivered full capability parity; the remaining friction is typing, result ergonomics, errors, and docs coverage | ✓ Done — shipped 0.7.0 |
| Target release version 0.7.0 (additive, no breaking changes) | Stubs/helpers/docs are additive; `with_intercept` deprecation stays back-compatible (warning, not removal) → minor bump | ✓ Done — 0.7.0 live on PyPI, no breaking changes |
| Align GSD milestone label to package version (v0.7.0) | STATE previously carried placeholder `v1.0`, diverging from the real 0.x package version | ✓ Done |
| v0.8.0 milestone = package the API as Claude Code skills | User wants an installable plugin mirroring `anofox-forecast` so Claude can apply the library from accurate cheat-sheets | ✓ Good — shipped 0.8.0 |
| 5 skills grouped by capability family | Mirrors anofox-forecast's granularity; coherent cheat-sheets over many tiny files or few giant ones | ✓ Good — 23/23 routing prompts route correctly |
| Plugin + marketplace layout mirrors `anofox-forecast` | Root `.claude-plugin/marketplace.json` + `plugins/polars-statistics/` with `plugin.json` and `skills/`; `user-invocable: false` auto-surfaced skills | ✓ Good (live `/plugin marketplace add` test still manual-only) |
| Bundle skills into the wheel + publish 0.8.0 to PyPI | User chose wheel bundling (maturin package data) over repo-only; additive → minor bump under semver | ✓ Done — maturin python-source copy at `_plugin/` with CI drift guard; 0.8.0 live on PyPI |
| Insert upstream refresh phases 16.1–16.3 into v0.8.0 | New anofox releases carried R-matching fixes and new API; skills had to document the final surface before bundling | ✓ Good — frozen at 0.4.7 / 0.5.22 |
| Defer anofox-statistics 0.4.8+ / anofox-regression 0.5.23+ (UPST-11) to v0.9.0 | Keep v0.8.0 shippable; wave 3 is large (classification, timeseries, survival, CV paths) | ✓ Taken up as v0.10.0. 0.9.0 was used for the #51/#52 bug-fix release |
| v0.10.0 = upgrade + small surface only; large families deferred | Upgrade spans 25+ upstream releases with numeric and API breaks; keep the milestone shippable | — Pending |
| Upgrade anofox before the Polars/PyO3 stack, as separate phases | The anofox crates don't depend on polars/pyo3; isolates statistical changes from stack churn | — Pending |
| Split anofox-regression into a 0.8 numerics step and a 0.9+ API step | Keeps R-fixture failures separate from compile errors | — Pending |
| Repo-wide ruff cleanup as its own phase (16.4), enforced in CI | Phase 16.1 found that ruff had never run in CI (393 findings) | ✓ Good — `ruff check .` green and gated |
| Accept the Python 3.9 `get_type_hints` narrowing (T-16.4-03) | PEP 604 annotations under the future import; runtime is unaffected | ⚠️ Revisit when 3.9 support is dropped |
| Skip the TestPyPI dry run for 0.8.0 | Stale token auth risk, and redundant with the production build matrix | ✓ Good — production publish succeeded on the first run |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-11-11 at v0.10.0 milestone start*
