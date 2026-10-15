# Requirements: polars-statistics

**Defined:** 2026-09-20
**Milestone:** v0.8.0 — Claude Code Skills Plugin
**Core Value:** Every public statistical capability in the backing `anofox-*` crates is exposed through the Polars/Python API — correctly, documented, and tested — and shipped to PyPI.

> Milestone focus: v0.6.0 delivered full API parity; v0.7.0 made that surface
> best-in-class for humans (typed, ergonomic, documented). v0.8.0 packages the
> existing v0.7.0 surface as installable **Claude Code skills** — a plugin +
> marketplace mirroring `anofox-forecast` — bundled into the wheel and shipped as
> 0.8.0 on PyPI. This milestone adds documentation/tooling, not statistical
> capability. Requirements for completed milestones are preserved in git history
> and `.planning/milestones/`.

## v1 Requirements

Requirements for this milestone. Each maps to exactly one roadmap phase.

### Plugin Scaffolding

- [x] **PLUG-01**: Root `.claude-plugin/marketplace.json` lists the `polars-statistics` plugin (name, source path, description, keywords), mirroring the anofox-forecast marketplace format
- [x] **PLUG-02**: `plugins/polars-statistics/.claude-plugin/plugin.json` manifest declares name, description, version, author, homepage, repository, and license
- [x] **PLUG-03**: The plugin installs cleanly in Claude Code — valid JSON, resolvable skill paths, and discoverable via the marketplace

### Skills

- [x] **SKILL-01**: `polars-statistics-hypothesis-tests/SKILL.md` covers the expression-API tests — parametric, nonparametric, distributional, correlation, categorical, modern (MMD/energy), and TOST/equivalence — with accurate signatures and runnable examples
- [x] **SKILL-02**: `polars-statistics-anova/SKILL.md` covers one-way, two-way, and repeated-measures ANOVA plus ICC, with worked examples
- [x] **SKILL-03**: `polars-statistics-regression/SKILL.md` covers the 40+ PyModel classes, the regression expressions, and the R-style formula parser
- [x] **SKILL-04**: `polars-statistics-results/SKILL.md` covers result ergonomics — `to_dict()`/`summary()`/`__repr__`, unnest/struct helpers, contextual errors, and the typed (`.pyi`) API
- [x] **SKILL-05**: `polars-statistics-getting-started/SKILL.md` covers install, the model-selection decision matrix, and sklearn migration
- [x] **SKILL-06**: Every `SKILL.md` carries correct frontmatter (`name`, `description`, `version` pinned to the package version, `user-invocable: false`) and a consistent cheat-sheet structure

### Accuracy

- [x] **VERIFY-01**: Every code example in every skill runs against the shipped 0.8.0 API — signatures, kwargs, and return shapes verified against the real surface, not invented
- [x] **VERIFY-02**: Each skill's `description` accurately reflects its covered surface so Claude auto-surfaces the right skill for a given task

### Packaging

- [x] **PKG-01**: Skill files are bundled into the built wheel via maturin package data and are present after `pip install polars-statistics`

### Release

- [x] **REL-07**: `polars-statistics` is bumped 0.7.0 → 0.8.0 across all version sources (`Cargo.toml`, `pyproject.toml`, `python/polars_statistics/__init__.py` `__version__`) plus the `plugin.json` and every `SKILL.md` version field, all aligned
- [ ] **REL-08**: 0.8.0 is published to production PyPI via the GitHub Actions OIDC pipeline and passes a post-release smoke check confirming import + skills present in the installed package

### Upstream Crate Refresh

- [x] **UPST-01**: Dependencies are bumped to anofox-statistics 0.4.6 and anofox-regression 0.5.16, and all CI gates (clippy, fmt, ruff, pytest) stay green
- [x] **UPST-02**: Tests and docs reflect the upstream R-matching behaviour changes (updated expected values, NaN/Inf now error instead of panicking)
- [x] **UPST-03**: New upstream capabilities are exposed via expressions/PyModels: effect sizes (Cohen's d, rank-biserial), Jarque-Bera, bootstrap CIs, conditional Fisher, conf_level for binomial/proportion/Fisher tests, GLMM Wald inference, Huber/RANSAC inference, NegBin estimate_dispersion, RLS initial_p_diagonal
- [x] **UPST-04**: Stubs, docstrings, mkdocs, CHANGELOG and skill cheat-sheets document the new surface, and the skill-example harness passes

- [x] **UPST-05**: Crates bumped to anofox-statistics 0.4.7 / anofox-regression 0.5.22 with behaviour changes locked; penalized GLM (`solvers::penalized_glm`, incl. estimate_dispersion) is exposed as a PyModel and Polars expressions, validated against R glm / MASS::glm.nb
- [x] **UPST-06**: AFT survival regression (`solvers::aft`) is exposed as a PyModel and Polars expressions, validated against R survival::survreg
- [x] **UPST-07**: Mann-Whitney / Wilcoxon expressions return the Hodges-Lehmann estimate and confidence interval (additive fields), and the new 16.2 surface is documented in stubs, mkdocs, CHANGELOG and skills

- [x] **UPST-08**: Post-hoc tests (Tukey HSD, pairwise t, Dunn), p_adjust and ptukey/qtukey from anofox-statistics 0.4.7 are exposed and match R
- [x] **UPST-09**: anofox-regression 0.5.17–0.5.21 additions (augment diagnostics, ModelInfo, GLM intercept inference, GLMM NegBin/Gamma/Tweedie + offset + per-factor random slopes, alias_tolerance, LARS fit-statistic getters, EB shrinkage) are exposed and match R
- [x] **UPST-10**: The 16.3 surface is documented in stubs, mkdocs, CHANGELOG and skills, with opt-outs recorded in COVERAGE.md

### Code Quality

- [x] **QUAL-01**: `ruff check .` is green repo-wide and enforced in CI

## v2 Requirements

Deferred to future releases.

### Upstream Wave 3 (deferred to v0.9.0, user decision 2026-10-14)

- **UPST-11**: Expose anofox-statistics 0.4.8+ (classification, timeseries incl. Durbin-Watson, survival modules) and anofox-regression 0.5.23+ (regularisation paths + k-fold CV, quantile-regression inference, predict_se, partial residuals, Breusch-Pagan/White tests, AFT survival curves, penalized-GLM augment, GLMM AIC/BIC). v0.8.0 is frozen at anofox-statistics 0.4.7 / anofox-regression 0.5.22.

### Forecast Integration

- **FCST-01**: Evaluate exposing `anofox-forecast` capabilities (separate crate, not in this milestone)

## Out of Scope

| Feature | Reason |
|---------|--------|
| New statistical methods / new API in v0.8.0 | This milestone packages the existing v0.7.0 surface as skills; no new algorithm or API work |
| Slash-command / `user-invocable: true` skills | Skills are auto-surfaced reference cheat-sheets, mirroring anofox-forecast (`user-invocable: false`) |
| Standalone skills repo | Skills live in this repo under `plugins/polars-statistics/`, mirroring anofox-forecast's monorepo layout |
| `anofox-forecast` crate integration | Separate crate, deferred (FCST-01); this milestone only mirrors its *plugin layout* |
| FFI / numpy↔faer architecture changes | Existing bridge is sound and stays as-is |
| Breaking changes to already-exposed API | v0.8.0 is additive under semver; no breaking changes |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PLUG-01 | Phase 13 | Complete |
| PLUG-02 | Phase 13 | Complete |
| PLUG-03 | Phase 13 | Complete |
| SKILL-01 | Phase 14 | Complete |
| SKILL-02 | Phase 14 | Complete |
| SKILL-03 | Phase 14 | Complete |
| SKILL-04 | Phase 14 | Complete |
| SKILL-05 | Phase 14 | Complete |
| SKILL-06 | Phase 14 | Complete |
| VERIFY-01 | Phase 15 | Complete |
| VERIFY-02 | Phase 15 | Complete |
| PKG-01 | Phase 16 | Complete |
| UPST-01 | Phase 16.1 | Complete |
| UPST-02 | Phase 16.1 | Complete |
| UPST-03 | Phase 16.1 | Complete |
| UPST-04 | Phase 16.1 | Complete |
| UPST-05 | Phase 16.2 | Complete |
| UPST-06 | Phase 16.2 | Complete |
| UPST-07 | Phase 16.2 | Complete |
| UPST-08 | Phase 16.3 | Complete |
| UPST-09 | Phase 16.3 | Complete |
| UPST-10 | Phase 16.3 | Complete |
| QUAL-01 | Phase 16.4 | Complete |
| REL-07 | Phase 17 | Complete |
| REL-08 | Phase 17 | Pending |

**Coverage:**

- v1 requirements: 25 total
- Mapped to phases: 25 ✓
- Unmapped: 0 ✓

---
*Requirements defined: 2026-09-20*
</content>
</invoke>
