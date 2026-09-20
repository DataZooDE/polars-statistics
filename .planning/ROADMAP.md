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

- [ ] **Phase 13: Plugin & Marketplace Scaffolding** - Stand up the `.claude-plugin/marketplace.json` + `plugins/polars-statistics/.claude-plugin/plugin.json` shell that installs cleanly in Claude Code
- [ ] **Phase 14: Skill Cheat-Sheet Authoring** - Author the five capability-family `SKILL.md` cheat-sheets with correct frontmatter and consistent structure, examples checked live while writing
- [ ] **Phase 15: Accuracy Verification Pass** - Run every skill example against the real API and audit each skill's description so Claude auto-surfaces the right one
- [ ] **Phase 16: Wheel Bundling** - Bundle the skill files into the built wheel via maturin package data so they ship on `pip install`
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
- [ ] 13-01-PLAN.md — Create both manifests (marketplace.json + plugin.json) and the skills/ tree so the plugin shell installs and resolves end-to-end

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
**Plans**: TBD

### Phase 15: Accuracy Verification Pass
**Goal**: The skills are trustworthy for an agent — every code example provably runs against the real API and every skill's description routes Claude to the right cheat-sheet for a task.
**Depends on**: Phase 14 (verifies the authored content)
**Requirements**: VERIFY-01, VERIFY-02
**Success Criteria** (what must be TRUE):
  1. Every code example in every skill has been executed against the shipped API, with signatures, kwargs, and return shapes confirmed against the real surface (not invented)
  2. Each skill's `description` accurately reflects its covered surface, so a task prompt auto-surfaces the correct skill
  3. Any signature, kwarg, or return-shape drift found during verification is corrected in the skill content before the phase closes
**Plans**: TBD

### Phase 16: Wheel Bundling
**Goal**: The skill files travel with the package — after installing the wheel, the plugin and all skills are present on disk.
**Depends on**: Phase 15 (verified skills are the artifacts being bundled)
**Requirements**: PKG-01
**Success Criteria** (what must be TRUE):
  1. maturin package-data configuration includes the plugin and skill files so they are packaged into the built wheel
  2. After `pip install polars-statistics` (from the built wheel), the marketplace, `plugin.json`, and every `SKILL.md` are present in the installed package
  3. The bundled-skills packaging does not break the existing wheel build or CI gates
**Plans**: TBD

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
Phases execute in numeric order: 13 → 14 → 15 → 16 → 17

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
| 13. Plugin & Marketplace Scaffolding | v0.8.0 | 0/1 | Planned | - |
| 14. Skill Cheat-Sheet Authoring | v0.8.0 | 0/TBD | Not started | - |
| 15. Accuracy Verification Pass | v0.8.0 | 0/TBD | Not started | - |
| 16. Wheel Bundling | v0.8.0 | 0/TBD | Not started | - |
| 17. Release 0.8.0 | v0.8.0 | 0/TBD | Not started | - |
</content>
</invoke>
