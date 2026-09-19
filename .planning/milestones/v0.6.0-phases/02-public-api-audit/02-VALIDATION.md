---
phase: 2
slug: public-api-audit
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-11
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Document verification (grep/manual cross-check — this phase produces a markdown artifact, not code) |
| **Config file** | none |
| **Quick run command** | `grep`-based cross-check of gap claims against crate sources |
| **Full suite command** | Manual review of 02-API-AUDIT.md against RESEARCH.md enumeration |
| **Estimated runtime** | ~1 min |

---

## Sampling Rate

- **After every task commit:** Spot-check that each documented gap/exposed claim traces to a real `pub` item or wrapper call site
- **Before `/gsd-verify-work`:** Both AUDIT-01 and AUDIT-02 gap lists complete; all 21 known candidates resolved
- **Max feedback latency:** ~60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 2-01-01 | 01 | 1 | AUDIT-01 / AUDIT-02 | — | N/A (doc artifact) | doc-check | grep-based cross-check | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

*Existing infrastructure covers all phase requirements — this phase produces a documentation artifact; no test framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Gap list completeness & accuracy | AUDIT-01, AUDIT-02 | Correctness of an enumeration is a human/semantic judgment | Cross-check every 02-API-AUDIT.md row against the crate source `pub` items and the wrapper's exposed surface; confirm all 21 known candidates are marked present/absent |

---

## Validation Sign-Off

- [ ] Every documented gap maps to a real `pub` item in the crate source
- [ ] Every "already exposed" claim maps to a concrete wrapper call site
- [ ] Known-candidate checklist (21 items) fully resolved
- [ ] Target exposure surface recorded for each gap
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
