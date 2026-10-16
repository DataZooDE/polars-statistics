# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v0.8.0 — Claude Code Skills Plugin

**Shipped:** 2026-10-16
**Phases:** 9 (13, 14, 15, 16.1, 16.2, 16.3, 16.4, 16, 17) | **Plans:** 44 | **Tasks:** 111

### What Was Built
- A Claude Code plugin and marketplace that mirror `anofox-forecast`. It has 5 version-pinned `SKILL.md` cheat-sheets plus `ROUTING.md` with 23 routing prompts.
- `scripts/verify_skill_examples.py`, a harness that executes every fenced example against the live package. Skipped blocks must carry an explicit reason.
- Wheel bundling through a maturin python-source copy at `polars_statistics/_plugin/`, with a CI drift guard.
- An upstream refresh to anofox-statistics 0.4.7 / anofox-regression 0.5.22. New surface: effect sizes, bootstrap CIs, PenalizedGLM, AFT survival, Hodges-Lehmann CIs, post-hoc tests, p_adjust/ptukey/qtukey, augment, model_info, GLMM families, alias_tolerance, EB shrinkage. All of it is R-validated.
- `ruff check .` green repo-wide and gated in CI.
- 0.8.0 released to PyPI through OIDC. 18 version sources are aligned and checked automatically.

### What Worked
- The skill-example harness turned the docs into executable tests. Real drift was caught in Phase 15 and again after each upstream wave.
- Verifiers re-ran gates on their own instead of trusting SUMMARY claims. That surfaced the ruff gap in 16.1 that CI had never enforced.
- The R-reference fixtures from upstream crates were reused byte-identically, which made "validated against R" cheap to prove.
- Release steps were split into separate, approval-gated plans (push, merge, TestPyPI decision, publish). Production published on the first run.

### What Was Inefficient
- Four phases (16.1–16.4) were inserted mid-milestone. That roughly tripled the scope of a milestone planned as "docs/tooling only", and the skills were rewritten after every upstream wave.
- Tracking artifacts drifted: stale ROADMAP checkboxes, REQUIREMENTS marked Pending, a missing `requirements-completed` in 15-01-SUMMARY, and no Progress rows for inserted phases.
- The verification digests went stale after the phases were completed, so the fingerprints had to be refreshed.
- Nyquist VALIDATION.md files were seeded but never reconciled, so all of them are still `draft`.

### Patterns Established
- Keep the skill source of truth in `plugins/polars-statistics/`, mirror it into `_plugin/`, and block divergence with CI.
- Every API-adding phase must also update stubs, mkdocs, CHANGELOG and the skills, then run the harness.
- `scripts/check_version_alignment.py --expect X.Y.Z` is the release gate for every version source.
- Lint-only changes need a behaviour-neutral proof: AST equality plus a public-API fingerprint.

### Key Lessons
1. When a policy says a gate exists (for example "ruff is a merge gate"), check that CI actually runs it. Make it a CI step early, not at release.
2. Upstream crate bumps belong in their own milestone. Mixing them into a docs milestone made the scope unpredictable, so UPST-11 now goes to v0.9.0.
3. Add the skill-example harness to CI so skills cannot drift between releases.

### Cost Observations
- Model mix: not tracked
- Sessions: not tracked
- Notable: the scope expanded through 4 inserted phases. 111 tasks were completed between 2026-09-20 and 2026-10-16.

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v0.8.0 | n/a | 9 | Executable docs (skill harness), inserted upstream-wave phases, a ruff CI gate, an automated version-alignment gate |

### Cumulative Quality

| Milestone | Tests | Coverage | Zero-Dep Additions |
|-----------|-------|----------|-------------------|
| v0.8.0 | 1805 pytest passed (at 16.3) + 22 rust_api | n/a | 0 new runtime deps |

### Top Lessons (Verified Across Milestones)

1. Release version drift has bitten twice: in v0.6.0 the bump only reached Cargo.lock, and in v0.8.0 there were 18 sources. Automate the alignment check.
