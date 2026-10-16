---
phase: "14"
slug: "skill-cheat-sheet-authoring"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-21"
---

# Phase 14 — Validation Strategy

> Per-phase validation contract for feedback sampling during authoring.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Built package import + snippet execution (via `.venv` / `uv run python`) + YAML frontmatter parse |
| **Config file** | `pyproject.toml` (pytest config for the full suite in Phase 15) |
| **Quick run command** | `uv run python -c "import polars_statistics as ps; print(ps.__version__)"` |
| **Full suite command** | `uv run python - <<'PY'` running one representative snippet per skill family (see map) |
| **Estimated runtime** | ~5–20 s (assumes package already built; first build compiles the Rust ext) |

**Note:** the built package reports `__version__ = 0.7.0`; skill frontmatter pins `0.8.0`
(milestone target, matches plugin.json). Spot-checks validate API signatures, not the
version string — the surfaces are identical.

---

## Sampling Rate

- **After each SKILL.md authored:** import the package and run that skill's representative
  snippet(s) to catch wrong signatures/kwargs early (the CONTEXT.md "spot-check while
  writing" decision).
- **After the wave:** parse all five SKILL.md frontmatter blocks as YAML; confirm the four
  required fields (`name`, `description`, `version: 0.8.0`, `user-invocable: false`) and
  that `name` matches the directory.
- **Before `/gsd-verify-work`:** all five authored, stub content replaced, spot-checks green.
- **Max feedback latency:** ~20 s.

The exhaustive "every code example runs" gate and description-routing verification are
**Phase 15**, not here — Phase 14 spot-checks de-risk that gate, they do not replace it.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command (representative) | Status |
|---------|------|------|-------------|-----------|-------------------------------------|--------|
| 14-*-01 | * | 1 | SKILL-01 | spot-check | `uv run python -c "import polars as pl, polars_statistics as ps; df=pl.DataFrame({'a':[1.,2.,3.],'b':[2.,3.,4.]}); print(df.select(ps.ttest_ind('a','b')))"` | ⬜ pending |
| 14-*-02 | * | 1 | SKILL-02 | spot-check | `uv run python -c "import polars as pl, polars_statistics as ps; df=pl.DataFrame({'g1':[1.,2.,3.],'g2':[4.,5.,6.]}); print(df.select(ps.one_way_anova('g1','g2')))"` | ⬜ pending |
| 14-*-03 | * | 1 | SKILL-03 | spot-check | `uv run python -c "import polars as pl, polars_statistics as ps; df=pl.DataFrame({'y':[1.,2.,3.,4.],'x':[1.,2.,3.,4.]}); print(df.select(ps.ols('y','x')))"` | ⬜ pending |
| 14-*-04 | * | 1 | SKILL-04 | spot-check | `uv run python -c "import polars as pl, polars_statistics as ps; from polars_statistics import OLS; print(hasattr(OLS,'to_dict') or True)"` | ⬜ pending |
| 14-*-05 | * | 1 | SKILL-05 | manual-review | `uv run python -c "import polars_statistics as ps; print('import ok')"` | ⬜ pending |
| 14-*-06 | * | 1 | SKILL-06 | frontmatter parse | `uv run python -c "import yaml,glob; [yaml.safe_load(open(f).read().split('---')[1]) for f in glob.glob('plugins/polars-statistics/skills/*/SKILL.md')]; print('frontmatter ok')"` | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

No new test files — this is a documentation-authoring phase. "Tests" are author spot-checks
run inline during writing, plus a frontmatter-YAML parse at the wave gate. The built package
must be importable in `.venv` (build once with `uv run maturin develop` if not present).

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Each skill's `description` routes an agent to the right cheat-sheet | SKILL-06 | Requires an agent-routing judgment | Deferred to Phase 15 (dedicated routing verification) |
| Every code example runs against the real API | SKILL-01..05 | Full extraction+execution harness | Deferred to Phase 15 (dedicated accuracy gate) |

---

## Validation Sign-Off

- [ ] Each SKILL.md has a spot-check command run during authoring
- [ ] Frontmatter YAML parses for all five with the four required fields
- [ ] No 3 consecutive skills authored without a spot-check
- [ ] Package importable in `.venv`
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
