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

- [ ] **PLUG-01**: Root `.claude-plugin/marketplace.json` lists the `polars-statistics` plugin (name, source path, description, keywords), mirroring the anofox-forecast marketplace format
- [ ] **PLUG-02**: `plugins/polars-statistics/.claude-plugin/plugin.json` manifest declares name, description, version, author, homepage, repository, and license
- [ ] **PLUG-03**: The plugin installs cleanly in Claude Code — valid JSON, resolvable skill paths, and discoverable via the marketplace

### Skills

- [ ] **SKILL-01**: `polars-statistics-hypothesis-tests/SKILL.md` covers the expression-API tests — parametric, nonparametric, distributional, correlation, categorical, modern (MMD/energy), and TOST/equivalence — with accurate signatures and runnable examples
- [ ] **SKILL-02**: `polars-statistics-anova/SKILL.md` covers one-way, two-way, and repeated-measures ANOVA plus ICC, with worked examples
- [ ] **SKILL-03**: `polars-statistics-regression/SKILL.md` covers the 40+ PyModel classes, the regression expressions, and the R-style formula parser
- [ ] **SKILL-04**: `polars-statistics-results/SKILL.md` covers result ergonomics — `to_dict()`/`summary()`/`__repr__`, unnest/struct helpers, contextual errors, and the typed (`.pyi`) API
- [ ] **SKILL-05**: `polars-statistics-getting-started/SKILL.md` covers install, the model-selection decision matrix, and sklearn migration
- [ ] **SKILL-06**: Every `SKILL.md` carries correct frontmatter (`name`, `description`, `version` pinned to the package version, `user-invocable: false`) and a consistent cheat-sheet structure

### Accuracy

- [ ] **VERIFY-01**: Every code example in every skill runs against the shipped 0.8.0 API — signatures, kwargs, and return shapes verified against the real surface, not invented
- [ ] **VERIFY-02**: Each skill's `description` accurately reflects its covered surface so Claude auto-surfaces the right skill for a given task

### Packaging

- [ ] **PKG-01**: Skill files are bundled into the built wheel via maturin package data and are present after `pip install polars-statistics`

### Release

- [ ] **REL-07**: `polars-statistics` is bumped 0.7.0 → 0.8.0 across all version sources (`Cargo.toml`, `pyproject.toml`, `python/polars_statistics/__init__.py` `__version__`) plus the `plugin.json` and every `SKILL.md` version field, all aligned
- [ ] **REL-08**: 0.8.0 is published to production PyPI via the GitHub Actions OIDC pipeline and passes a post-release smoke check confirming import + skills present in the installed package

## v2 Requirements

Deferred to future releases.

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

Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| PLUG-01 | — | Pending |
| PLUG-02 | — | Pending |
| PLUG-03 | — | Pending |
| SKILL-01 | — | Pending |
| SKILL-02 | — | Pending |
| SKILL-03 | — | Pending |
| SKILL-04 | — | Pending |
| SKILL-05 | — | Pending |
| SKILL-06 | — | Pending |
| VERIFY-01 | — | Pending |
| VERIFY-02 | — | Pending |
| PKG-01 | — | Pending |
| REL-07 | — | Pending |
| REL-08 | — | Pending |

**Coverage:**

- v1 requirements: 14 total
- Mapped to phases: 0 (pending roadmap)
- Unmapped: 14 ⚠️

---
*Requirements defined: 2026-09-20*
