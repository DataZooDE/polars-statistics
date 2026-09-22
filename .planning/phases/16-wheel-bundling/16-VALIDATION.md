---
phase: "16"
slug: "wheel-bundling"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-21"
---

# Phase 16 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | maturin build + `unzip -l` archive inspection + clean-venv install assertion |
| **Config file** | `pyproject.toml` ([tool.maturin]) |
| **Quick run command** | `unzip -l "$(ls -t target/wheels/polars_statistics-*.whl \| head -1)" \| grep -c 'polars_statistics/_plugin/'` |
| **Full suite command** | Build wheel → archive check → clean-venv install → assert files on disk (see below) |
| **Estimated runtime** | ~5 min first build (Rust compile), ~10–30 s incremental |

---

## Sampling Rate

- **After the copy task:** assert the `_plugin/` tree exists in the working tree with marketplace.json (source="."), plugin.json, 5 SKILL.md, ROUTING.md; and that bundled SKILL.md content is IDENTICAL to the root plugin SKILL.md (drift guard).
- **After the build task:** `unzip -l` the built wheel — confirm `polars_statistics/_plugin/` present with marketplace.json, plugin.json, exactly 5 SKILL.md.
- **Phase gate:** clean-venv `pip install <wheel>` then assert marketplace.json + plugin.json + all 5 SKILL.md exist on disk under the installed `polars_statistics/_plugin/`. Existing CI gates (clippy, fmt, ruff, pytest) remain green.

---

## Per-Task Verification Map

| Task | Requirement | Test Type | Automated Command | Status |
|------|-------------|-----------|-------------------|--------|
| copy plugin into python-source | PKG-01 | smoke | `test -f python/polars_statistics/_plugin/marketplace.json && test $(ls python/polars_statistics/_plugin/plugins/polars-statistics/skills/*/SKILL.md 2>/dev/null \| wc -l) -eq 5` | ⬜ pending |
| drift guard (bundled==root) | PKG-01 | smoke | `for s in getting-started hypothesis-tests anova regression results; do diff -q plugins/polars-statistics/skills/polars-statistics-$s/SKILL.md python/polars_statistics/_plugin/plugins/polars-statistics/skills/polars-statistics-$s/SKILL.md || exit 1; done; echo BUNDLE_MATCHES_ROOT` | ⬜ pending |
| build + archive check | PKG-01 | smoke | `unzip -l "$(ls -t target/wheels/polars_statistics-*.whl \| head -1)" \| grep -c 'polars_statistics/_plugin/.*SKILL.md'` (expect 5) | ⬜ pending |
| clean-venv install assert | PKG-01 | behavioral | build → `pip install` into temp venv → python asserts marketplace.json + plugin.json + 5 SKILL.md exist under installed `polars_statistics/_plugin/` | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Recommend a small committed verifier `scripts/verify_wheel_bundling.py` (or inline bash in the plan verify blocks) that: builds/locates the wheel, inspects the archive, and asserts the plugin files are present. No new pytest framework needed. The `_plugin/marketplace.json` must carry `"source": "."` (differs from the root marketplace.json which uses `"./plugins/polars-statistics"`).

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Installed plugin is loadable by Claude Code from the wheel location | PKG-01 | Requires live Claude Code | Optional confidence check; the on-disk-presence assertion is the phase gate |

---

## Validation Sign-Off

- [ ] Bundled `_plugin/` tree present with marketplace.json (source="."), plugin.json, 5 SKILL.md, ROUTING.md
- [ ] Bundled SKILL.md content identical to root (drift guard green)
- [ ] Built wheel archive contains `polars_statistics/_plugin/...` with 5 SKILL.md
- [ ] Clean-venv install places all plugin files on disk
- [ ] Existing wheel build + CI gates unaffected
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
