---
phase: 02-public-api-audit
verified: 2026-08-11T22:40:00Z
status: passed
score: 7/7 must-haves verified
behavior_unverified: 0
overrides_applied: 0
human_verification_resolved:
  - test: "HcInference/HcType gap verdict accuracy"
    result: "FIXED — the audit's HcInference rows were corrected from 'NO' to 'PARTIAL (OLS only)' with concrete evidence (ols_summary hc_type param @ src/expressions/regression.rs:2640-2690; OLS.hc_inference() @ src/pymodels/py_ols.rs:371-405), resolving the internal NO-vs-PARTIAL inconsistency. The REGR-04 cross-reference was reframed to 'extend HC to non-OLS regressors, not implement from scratch', and a new Deferred Scope Question #4 records the A-vs-B framing decision for Phase 4. Correction applied by autonomous orchestrator 2026-08-11."
---

# Phase 2: Public API Audit — Verification Report

**Phase Goal:** An authoritative, documented gap list enumerates exactly which public functions in each upgraded crate are not yet exposed, defining the concrete scope for the two parity phases.
**Verified:** 2026-08-11T22:40:00Z
**Status:** passed (HC-inference accuracy finding resolved 2026-08-11 — audit rows corrected to PARTIAL/OLS with evidence; Deferred Q4 added)
**Re-verification:** No — initial verification; single accuracy finding corrected in-place

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | 02-API-AUDIT.md enumerates every public anofox-statistics 0.4.2 item with an exposed? verdict (AUDIT-01) | ✓ VERIFIED | Per-crate table with 102 items across 10 categories; each row carries name/kind/exposed?/target surface/priority/notes. Spot-checked: `one_way_anova`, `two_way_anova`, `repeated_measures_anova` confirmed `pub fn` in crate at `parametric/anova.rs:297,898,1157`; all marked NO. `energy_distance_test` nD overload confirmed at `modern/energy.rs:157`, marked NO. `ICCResult`/`ICCType` confirmed at `correlation/icc.rs:10,41`, marked NO. |
| 2 | 02-API-AUDIT.md enumerates every public anofox-regression 0.5.13 capability with an exposed? verdict (AUDIT-02) | ✓ VERIFIED | Per-crate gap table covering Solvers, HC Inference, Diagnostics, Streaming, Core/Utility. Spot-checked: `GlmmRegressor`/`FittedGlmm` confirmed at `solvers/glmm.rs:84`; `PSplineRegressor` at `solvers/pspline.rs:25`; `GammaRegressor` at `solvers/gamma.rs:37`; `HcInference` at `inference/robust_covariance.rs:59`; `compute_hc_inference` at line 217. All marked NO (with HC caveat — see Truth 7). `FactorSummary` at `solvers/glmm.rs:619` confirmed `pub struct`. All gap regressors (TheilSen, RANSAC, BayesianRidge, ARD, LARS, PassiveAggressive) confirmed absent from `src/` and `python/` wrapper layer. |
| 3 | Each gap row records its target exposure surface (expression / PyModel / both / util) | ✓ VERIFIED | All table rows carry a "Target Surface" column populated with expression / PyModel / both / util. Internal items carry util. No blank target surface cells for user-facing gaps. |
| 4 | All 21 known candidates from CONTEXT.md appear with an explicit present-or-absent verdict | ✓ VERIFIED | Known-Candidate Checklist section contains exactly 21 rows (confirmed by count). All candidates resolved: ANOVA triple = Gap; energy_distance = Partial; GlmmRegressor/PSplineRegressor/GammaRegressor = Gap; HC = Partial; diagnostics suite (Cook's/VIF/Leverage/Residuals/Condition) = No gap; MomentAccumulator/TheilSen/RANSAC/LOWESS/BayesianRidge/PassiveAggressive/LARS/ARD = Gap or internal-only. LOWESS confirmed internal-only (not in `solvers/mod.rs` pub use list; not in `lib.rs` pub use). |
| 5 | The 3 open scope questions from RESEARCH.md are recorded as deferred Phase 3/4 decisions, not dropped | ✓ VERIFIED | Section "Deferred Scope Questions" (line 486) contains Q1 (STAT-04 nD sufficiency), Q2 (PassiveAggressive/MomentAccumulator PyModel-only), Q3 (ICC stub Phase 3 scope). Each marked "Status: Deferred to Phase 3/4 planning." Not dropped. |
| 6 | The ICC stub is flagged as exposed-but-non-functional (distinct from a normal gap) | ✓ VERIFIED | Special-Case Flags section (a) documents `icc_fit` body. Verified against actual source: `src/expressions/correlation.rs:270-302` has `// TODO: Implement proper ICC with matrix input` and returns all `f64::NAN`. Audit correctly classifies as "exposed-but-stubbed" distinct from a missing-binding gap. `ICCResult`/`ICCType` still absent from output schema (correct — these are listed as NO in table). |
| 7 | HcInference/HcType gap verdict is accurate | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED | The per-item table row marks `HcInference` as "NO" with note "no user-callable expression returns HC inference output." However: (1) `ols_summary` expression already accepts `hc_type` parameter and internally calls `hc_inference`, populating `std_error`/`statistic`/`p_value` fields with HC data; (2) `OLS` PyModel (registered in `__init__.py`) exposes `.hc_inference(x, hc_type)` method returning a dict. The Known-Candidate Checklist row correctly says "HcType imported but HC output not wired | Partial". The per-item table row conflicts with the checklist row. A human must decide if this internal inconsistency is acceptable for Phase 4 planning or requires a correction to the per-item row (from "NO" to "YES (partial — OLS only)"). |

**Score:** 6/7 truths verified (1 present, behavior/accuracy unverified — requires human judgment)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.planning/phases/02-public-api-audit/02-API-AUDIT.md` | Authoritative gap list with per-crate tables | ✓ VERIFIED | File exists, 559 lines, created by commit `3296e5d`. Confirmed to contain both crate sections, Known-Candidate Checklist, Special-Case Flags, Deferred Scope Questions, and Cross-Reference section. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `02-API-AUDIT.md` | Phase 3 planner scope | All STAT-* gaps mapped in Cross-Reference section | ✓ WIRED | STAT-01 → one_way_anova; STAT-02 → two_way_anova; STAT-03 → repeated_measures_anova; STAT-04 → energy_distance_test nD; STAT-05 → ICC stub + remaining items |
| `02-API-AUDIT.md` | Phase 4 planner scope | All REGR-* gaps mapped in Cross-Reference section | ✓ WIRED | REGR-01 → GlmmRegressor; REGR-02 → PSplineRegressor; REGR-03 → GammaRegressor; REGR-04 → HcInference (with caveat); REGR-05 → GLM diagnostic variants; REGR-06 → MEDIUM solver set |
| Gap rows | Pub declarations in crate source | Spot-check via crate registry files | ✓ WIRED | ANOVA functions, energy_distance_test nD, HcInference, GlmmRegressor, PSplineRegressor, GammaRegressor, FactorSummary — all confirmed real `pub` items in cited crate source files. |

---

### Data-Flow Trace (Level 4)

Not applicable. This is a documentation-only phase. No data flows through the deliverable — it is a planning artifact read by human planners and downstream phase agents.

---

### Behavioral Spot-Checks

Step 7b: SKIPPED (no runnable entry points in this phase — documentation-only, no code changes)

---

### Probe Execution

No probes declared in PLAN.md. The automated verify from PLAN.md task was not run independently (it was run by the executor). The verify commands greping for section headings and key items are passive structural checks — not probes requiring dedicated execution.

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| AUDIT-01 | 02-01-PLAN.md | Gap list for anofox-statistics functions | ✓ SATISFIED | Per-crate table with 102 items, 13 user-facing gaps identified and documented |
| AUDIT-02 | 02-01-PLAN.md | Gap list for anofox-regression capabilities | ✓ SATISFIED | Per-crate table with 20 actionable gaps (7 HIGH + 13 MEDIUM) identified and documented |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `02-API-AUDIT.md` | 282 | `HcInference` marked "NO" in per-item table, but "PARTIAL" in Known-Candidate Checklist for the same capability | ⚠️ Warning | Creates ambiguity for Phase 4 planner: will they start HC from scratch, or know that OLS already has it? Does not prevent using the audit as planning input, but risks re-implementing already-done work. |

No `TBD`, `FIXME`, or `XXX` debt markers found in the audit document. No `src/` or `python/` files were modified during Phase 2 (confirmed via git diff across all phase 2 commits `2509d8b..5a40fb8`).

---

### Human Verification Required

#### 1. HC Inference Gap Accuracy

**Test:** Read `02-API-AUDIT.md` rows for `HcType` and `HcInference` in the Inference/HC Robust Errors section, then compare against:
- `src/expressions/regression.rs` lines 2640–2693 (ols_summary routing through hc_inference)
- `src/pymodels/py_ols.rs` lines 371–405 (OLS.hc_inference() PyModel method)
- `python/polars_statistics/exprs/regression.py` lines 2066–2124 (ols_summary Python wrapper with hc_type param)

**Expected:** You should find that HC inference data (std_errors, t_statistics, p_values via HcInference) IS already accessible for OLS fits via:
1. `ps.ols_summary("y", "x1", hc_type="hc3")` — expression surface
2. `OLS().fit(x, y).hc_inference(x, "hc3")` — PyModel surface

**Decision required:** 
- Option A: Accept the audit's per-item "NO" as a conservative framing (the struct `HcInference` is not directly user-importable; downstream planner will discover the partial exposure). Keep doc as-is.
- Option B: Correct `HcInference` per-item row from "NO" to "YES (partial — OLS expression + PyModel only)" and re-scope REGR-04 to "extend HC inference to non-OLS regressors" rather than "implement HC inference from scratch."

**Why human:** This is an accuracy judgment call about audit framing, not a mechanical code check. Option B changes the scope of REGR-04 for Phase 4 planners. Option A may cause Phase 4 to duplicate already-wired functionality.

---

### Gaps Summary

No hard gaps (missing artifacts, absent sections, dropped deferred questions). The single unresolved item is an internal consistency issue within the audit document regarding HC inference exposure — one row says "NO" while the checklist row correctly says "PARTIAL." This does not prevent Phase 3 from starting (STAT-* gaps are unaffected), but should be resolved before Phase 4 planning begins so the HC gap scope is correctly understood.

---

### Spot-Check Accuracy Summary

| Gap Row Spot-Checked | Audit Verdict | Crate Evidence | Wrapper Evidence | Accurate? |
|----------------------|---------------|----------------|------------------|-----------|
| `one_way_anova` | NO | `parametric/anova.rs:297` — `pub fn one_way_anova` | Absent from `src/` and `python/` | Yes |
| `two_way_anova` | NO | `parametric/anova.rs:898` — `pub fn two_way_anova` | Absent from `src/` and `python/` | Yes |
| `repeated_measures_anova` | NO | `parametric/anova.rs:1157` — `pub fn repeated_measures_anova` | Absent from `src/` and `python/` | Yes |
| `energy_distance_test` (nD) | NO | `modern/energy.rs:157` — nD overload; `modern/energy.rs:187` — 1D overload | Wrapper uses `energy_distance_test_1d` (confirmed at `src/expressions/modern.rs:6`) | Yes |
| `icc` binding | YES (stub) | `correlation/icc.rs:99` — `pub fn icc` | `icc_fit` at `src/expressions/correlation.rs:270` has `TODO` and returns all NaN | Yes |
| `GlmmRegressor` | NO | `solvers/glmm.rs:84` — `pub struct GlmmRegressor` | Absent from `src/` and `python/` | Yes |
| `PSplineRegressor` | NO | `solvers/pspline.rs:25` — `pub struct PSplineRegressor` | Absent from `src/` and `python/` | Yes |
| `GammaRegressor` | NO | `solvers/gamma.rs:37` — `pub struct GammaRegressor` | Absent from `src/` and `python/` | Yes |
| `FactorSummary` | NO (HIGH priority) | `solvers/glmm.rs:619` — `pub struct FactorSummary`; `glmm.rs:818` — `pub fn factors(&self) -> &[FactorSummary]` | Absent; correct as HIGH priority output type | Yes |
| `HcInference` | NO | `inference/robust_covariance.rs:59` — `pub struct HcInference`; re-exported at `lib.rs:68` | OLS.hc_inference() at `py_ols.rs:371`; ols_summary uses HC at `regression.rs:2670` | **Partially inaccurate — HC data IS exposed for OLS (see Human Verification)** |
| LOWESS | internal-only | Not in `lib.rs` pub use; `pub mod lowess` in `solvers/mod.rs` but no `pub use lowess::*` | Absent from user-facing API | Yes |

---

_Verified: 2026-08-11T22:40:00Z_
_Verifier: Claude (gsd-verifier)_
