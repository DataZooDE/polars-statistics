---
phase: 16-wheel-bundling
verified: 2026-10-15T22:10:00Z
status: passed
score: 5/5 must-haves verified
covered_files:
  - ".github/workflows/ci.yml"
  - ".planning/phases/16-wheel-bundling/16-01-PLAN.md"
  - ".planning/phases/16-wheel-bundling/16-01-SUMMARY.md"
  - ".planning/phases/16-wheel-bundling/16-CONTEXT.md"
  - ".planning/phases/16-wheel-bundling/16-RESEARCH.md"
  - ".planning/phases/16-wheel-bundling/16-VALIDATION.md"
  - "python/polars_statistics/_plugin/.claude-plugin/plugin.json"
  - "python/polars_statistics/_plugin/marketplace.json"
  - "python/polars_statistics/_plugin/skills/ROUTING.md"
  - "python/polars_statistics/_plugin/skills/polars-statistics-anova/SKILL.md"
  - "python/polars_statistics/_plugin/skills/polars-statistics-getting-started/SKILL.md"
  - "python/polars_statistics/_plugin/skills/polars-statistics-hypothesis-tests/SKILL.md"
  - "python/polars_statistics/_plugin/skills/polars-statistics-regression/SKILL.md"
  - "python/polars_statistics/_plugin/skills/polars-statistics-results/SKILL.md"
  - "scripts/verify_wheel_bundling.py"
covered_digest: "v1:sha256:e6aa4b55e99452616dca5fd13e13cdda772d050d6b9a871f23354176a2d9ed35"
behavior_unverified: 0
overrides_applied: 0
---

# Phase 16: Wheel Bundling Verification Report

**Phase Goal:** The skill files travel with the package — after installing the wheel, the plugin and all skills are present on disk.
**Verified:** 2026-10-15T22:10:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | maturin package-data configuration includes the plugin and skill files so they are packaged into the built wheel | ✓ VERIFIED | No `pyproject.toml`/`[tool.maturin]` change was made or needed; maturin's `python-source = "python"` auto-include carries everything under `python/polars_statistics/_plugin/` into the wheel. Confirmed by `unzip -l target/wheels/polars_statistics-0.7.0-cp39-abi3-manylinux_2_39_x86_64.whl \| grep _plugin` listing all 8 files (marketplace.json, `.claude-plugin/plugin.json`, ROUTING.md, 5× SKILL.md) under `polars_statistics/_plugin/`. |
| 2 | After `pip install polars-statistics` (from the built wheel), the marketplace, `plugin.json`, and every `SKILL.md` are present in the installed package | ✓ VERIFIED | Independently re-ran `uv run python scripts/verify_wheel_bundling.py` (full, both halves) against the current wheel: `Half B: behavioral clean-venv install check → PASS clean-venv install places all plugin files on disk`, exit 0. This spins up a throwaway `python3 -m venv`, `pip install`s the newest wheel, and asserts via `pathlib.Path(polars_statistics.__file__).parent / "_plugin"` that all 7 required paths exist on disk. |
| 3 | The bundled-skills packaging does not break the existing wheel build or CI gates | ✓ VERIFIED | `uv run ruff check .` → "All checks passed!" (exit 0, independently re-run). No `Cargo.toml`/Rust source changes (`git diff --stat <plan-head>..HEAD -- Cargo.toml src/` empty — confirmed no Rust/pyproject diffs beyond the new files). CI `lint` job gained one cheap `--drift-only` step (stdlib-only, no wheel build) placed correctly right after `ruff check .` and before the Rust toolchain steps. |
| 4 | Each bundled SKILL.md is byte-identical to its root `plugins/polars-statistics` counterpart (drift guard) | ✓ VERIFIED | `diff -q` against all 5 skills + ROUTING.md + `plugin.json` returned no differences; only `marketplace.json`'s `source` field differs (`"./plugins/polars-statistics"` → `"."`), which is the single intentional, documented change. Independently re-ran `uv run python scripts/verify_wheel_bundling.py --drift-only` → PASS. |
| 5 | The repo-root plugin (`.claude-plugin/`, `plugins/`) is unchanged and still usable for local `/plugin marketplace add` | ✓ VERIFIED | `git diff --stat <plan_head_before=6eb2300>..HEAD -- .claude-plugin/ plugins/` is empty; `git log --oneline -1 -- .claude-plugin/ plugins/` shows the last touch to those paths predates this phase (`f0ee6f0`, phase 16.3). |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `python/polars_statistics/_plugin/marketplace.json` | `source = "."` | ✓ VERIFIED | Present, parses as valid JSON, `plugins[0].source == "."` confirmed via script and manual `diff`. |
| `python/polars_statistics/_plugin/.claude-plugin/plugin.json` | byte-identical to root | ✓ VERIFIED | `diff` against `plugins/polars-statistics/.claude-plugin/plugin.json` — no difference. |
| `python/polars_statistics/_plugin/skills/ROUTING.md` | byte-identical to root | ✓ VERIFIED | `diff -q` clean. |
| 5× `python/polars_statistics/_plugin/skills/polars-statistics-*/SKILL.md` | byte-identical to root, exactly 5 | ✓ VERIFIED | `diff -q` clean for all 5; `find ... -name SKILL.md \| wc -l` = 5; no `.gitkeep`/placeholder present. |
| `scripts/verify_wheel_bundling.py` | stdlib-only drift + behavioral verifier | ✓ VERIFIED | Imports are `argparse, glob, json, os, shutil, subprocess, sys, tempfile, venv` — all stdlib, no third-party. `ruff check scripts/verify_wheel_bundling.py` clean. Both halves (`--drift-only` and full) independently re-run, both exit 0. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `python/polars_statistics/_plugin/` tree | built wheel archive | maturin `python-source` auto-include, no pyproject.toml change | ✓ WIRED | `unzip -l` on the newest wheel lists `polars_statistics/_plugin/...` for all 8 files with correct byte sizes matching source files (spot-checked `marketplace.json` and `polars-statistics-anova/SKILL.md` via `unzip -p` + `diff` — content matches exactly). |
| `_plugin/marketplace.json` (`source: "."`) | `_plugin/.claude-plugin/plugin.json` | relative path resolution, plugin dir == marketplace dir | ✓ WIRED | Single changed field verified programmatically (JSON parse + equality check) in both the drift guard and by manual `diff`. |
| `scripts/verify_wheel_bundling.py --drift-only` | CI `lint` job | new step in `.github/workflows/ci.yml`, after `ruff check .`, before Rust toolchain setup | ✓ WIRED | Confirmed step present at the correct position (lines 36-37), correct job (`lint`), no wheel build or Rust toolchain dependency — stays cheap as intended. |

### Data-Flow Trace (Level 4)

Not applicable in the usual dynamic-rendering sense — this phase bundles static files. The equivalent trace is "root file → wheel archive entry", traced above via direct `unzip -p` content comparison (not merely archive listing), which confirms the wheel's bundled content is not a stale/static artifact but the actual current source content.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Drift guard fails loudly on real drift (adversarial check, not in SUMMARY) | Copied `plugins/` + `python/polars_statistics/_plugin/` + the verifier script into an isolated scratch dir (never touching tracked files), appended one byte to the scratch copy of `polars-statistics-getting-started/SKILL.md`, re-ran the drift-only check | Baseline run on the unmutated scratch copy: `RESULT: PASS`, exit 0. After the single-byte mutation: `FAIL  drift: .../python/.../SKILL.md != .../plugins/.../SKILL.md` then `RESULT: FAIL — 1 check(s) failed`, exit 1 | ✓ PASS |
| Drift guard passes on real, current repo state | `uv run python scripts/verify_wheel_bundling.py --drift-only` | `PASS  all bundled files byte-identical to root...`, exit 0 | ✓ PASS |
| Behavioral install check (PKG-01 proper) | `uv run python scripts/verify_wheel_bundling.py` (both halves) | `PASS  clean-venv install places all plugin files on disk`, exit 0 | ✓ PASS |
| ruff clean | `uv run ruff check .` | `All checks passed!`, exit 0 | ✓ PASS |
| Wheel archive contains exactly 5 SKILL.md + manifests | `unzip -l` on newest wheel | 5 `SKILL.md` entries + `marketplace.json` + `.claude-plugin/plugin.json`, all matching source byte sizes | ✓ PASS |
| Wheel content (not just filename) matches current source | `unzip -p` + `diff` on `marketplace.json` and `polars-statistics-anova/SKILL.md` | No diff output | ✓ PASS |
| Root plugin untouched | `git diff --stat <plan-head>..HEAD -- .claude-plugin/ plugins/` | Empty | ✓ PASS |

All checks completed well under the 10-second-per-check guidance except the venv install (Half B), which is a deliberate, already-scoped behavioral check (not a server start, no tracked-file mutation).

### Probe Execution

No `scripts/*/tests/probe-*.sh` convention is used in this phase; `scripts/verify_wheel_bundling.py` is itself the phase-specific verifier and was executed directly above (both `--drift-only` and full modes), not via the probe convention. N/A.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|--------------|-------------|--------------|--------|----------|
| PKG-01 | 16-01-PLAN.md | Skill files are bundled into the built wheel via maturin package data and are present after `pip install polars-statistics` | ✓ SATISFIED | Truths 1-5 above; wheel archive + clean-venv install both independently re-verified. |

No orphaned requirements: `.planning/REQUIREMENTS.md` maps only PKG-01 to Phase 16, and the plan claims exactly PKG-01.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | None found | — | `grep -nE "TBD\|FIXME\|XXX\|HACK\|PLACEHOLDER"` over `scripts/verify_wheel_bundling.py` and the CI diff returned nothing. No stub return values, no empty handlers, no hardcoded empty-data patterns relevant to this phase's static-file-bundling scope. |

### Human Verification Required

None. All must-haves are deterministically checkable (file byte-identity, archive listing, JSON parsing, subprocess exit codes) and were independently re-executed rather than taken from SUMMARY.md.

### Gaps Summary

None. All 5 must-have truths verified, all artifacts present/substantive/wired, the one key link (maturin python-source → wheel archive) independently confirmed by content comparison (not just filename presence), the drift guard was adversarially proven to actually fail on real drift (not just claimed to), and root-plugin non-regression and CI-gate non-regression were both independently confirmed.

---

_Verified: 2026-10-15T22:10:00Z_
_Verifier: Claude (gsd-verifier)_
