---
phase: "13"
slug: "plugin-marketplace-scaffolding"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-20"
---

# Phase 13 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Python stdlib `json` (no test framework needed — JSON-validity smoke checks) |
| **Config file** | none — infrastructure phase, files created by the phase itself |
| **Quick run command** | `python3 -m json.tool .claude-plugin/marketplace.json && python3 -m json.tool plugins/polars-statistics/.claude-plugin/plugin.json` |
| **Full suite command** | `python3 -c "import json,os; d=json.load(open('plugins/polars-statistics/.claude-plugin/plugin.json')); assert os.path.isdir('plugins/polars-statistics/skills'), 'skills dir missing'; m=json.load(open('.claude-plugin/marketplace.json')); assert any(p['name']=='polars-statistics' for p in m['plugins']), 'plugin not listed'"` |
| **Estimated runtime** | ~1 second |

---

## Sampling Rate

- **After every task commit:** Run the quick run command (JSON validity on both manifests)
- **After every plan wave:** Run the full suite command (path resolution + marketplace listing)
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 2 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 13-01-01 | 01 | 1 | PLUG-01 | — | N/A | smoke | `python3 -m json.tool .claude-plugin/marketplace.json` | ❌ W0 | ⬜ pending |
| 13-01-02 | 01 | 1 | PLUG-02 | — | N/A | smoke | `python3 -m json.tool plugins/polars-statistics/.claude-plugin/plugin.json` | ❌ W0 | ⬜ pending |
| 13-01-03 | 01 | 1 | PLUG-03 | — | N/A | smoke | `python3 -c "import json,os; d=json.load(open('plugins/polars-statistics/.claude-plugin/plugin.json')); assert os.path.isdir('plugins/polars-statistics/skills')"` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements — the JSON smoke commands above use only Python stdlib and need no framework install. No dedicated pytest file is required for this infrastructure-only phase.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Plugin installs and is discoverable via marketplace in Claude Code | PLUG-03 | Requires a live Claude Code `/plugin marketplace add` + install action | Add the local marketplace, install `polars-statistics`, confirm it appears and resolves skill paths |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 2s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
