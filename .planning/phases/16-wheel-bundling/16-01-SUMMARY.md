---
phase: 16-wheel-bundling
plan: 01
subsystem: infra
tags: [maturin, pyo3, packaging, claude-code-plugin, wheel, ci]

requires:
  - phase: 13-plugin-marketplace-scaffolding
    provides: "Repo-root .claude-plugin/marketplace.json + plugins/polars-statistics/ plugin tree"
  - phase: 14-skill-cheat-sheet-authoring
    provides: "Five SKILL.md cheat-sheets + ROUTING.md"
  - phase: 16.4-ruff-cleanup
    provides: "Pinned ruff==0.16.8 CI gate; python/ and scripts/ must stay ruff-clean"
provides:
  - "python/polars_statistics/_plugin/ tree bundled into the wheel via maturin python-source auto-include"
  - "scripts/verify_wheel_bundling.py drift guard + behavioral clean-venv install verifier"
  - "CI drift-only gate preventing bundled/root skill divergence from shipping silently"
affects: [17-release-0.8.0]

actuals:
  tokens: 55110
  tasks: 3
  commits: 3
  plan_head_before: 6eb2300cc69493b7236d083ab899442dc059d2b0

tech-stack:
  added: []
  patterns:
    - "Static data bundled into a PyO3/maturin wheel by physical copy under python/<pkg>/_plugin/ (python-source auto-include), not pyproject.toml [tool.maturin] include globs"
    - "Drift guard pattern: a stdlib-only script asserts byte-identity between a repo-root source of truth and its bundled copy, runnable standalone in CI (--drift-only) or in full (behavioral install check) locally"

key-files:
  created:
    - python/polars_statistics/_plugin/marketplace.json
    - python/polars_statistics/_plugin/.claude-plugin/plugin.json
    - python/polars_statistics/_plugin/skills/ROUTING.md
    - python/polars_statistics/_plugin/skills/polars-statistics-anova/SKILL.md
    - python/polars_statistics/_plugin/skills/polars-statistics-getting-started/SKILL.md
    - python/polars_statistics/_plugin/skills/polars-statistics-hypothesis-tests/SKILL.md
    - python/polars_statistics/_plugin/skills/polars-statistics-regression/SKILL.md
    - python/polars_statistics/_plugin/skills/polars-statistics-results/SKILL.md
    - scripts/verify_wheel_bundling.py
  modified:
    - .github/workflows/ci.yml

key-decisions:
  - "Copied the plugin tree under python/polars_statistics/_plugin/ (maturin python-source auto-include) rather than pyproject.toml [tool.maturin] include globs — the latter lands files outside the package namespace (site-packages/plugins/...), which is not discoverable via polars_statistics.__file__."
  - "_plugin/marketplace.json is the only file NOT byte-identical to its root counterpart: source changed from './plugins/polars-statistics' to '.' since the plugin dir and marketplace.json are now siblings inside _plugin/."
  - "[Rule 2] Wired scripts/verify_wheel_bundling.py --drift-only into the CI lint job (no wheel build required, stdlib-only) so a future skill edit under plugins/ that isn't re-synced into _plugin/ fails CI before release, instead of shipping stale silently."

patterns-established:
  - "Drift-guard verifier with a --drift-only flag: cheap structural check usable in CI without a full release-wheel build, plus a full behavioral mode (clean-venv pip install + on-disk assertion) for local/phase-gate use."

requirements-completed: [PKG-01]

coverage:
  - id: D1
    description: "After pip install of the built wheel into a clean venv, marketplace.json, plugin.json, and all five SKILL.md exist on disk under the installed polars_statistics/_plugin/"
    requirement: PKG-01
    verification:
      - kind: other
        ref: "uv run python scripts/verify_wheel_bundling.py (Half B: clean-venv install assertion) -> PKG01_BEHAVIORAL_OK"
        status: pass
    human_judgment: false
  - id: D2
    description: "Built wheel archive contains polars_statistics/_plugin/ with exactly five SKILL.md"
    requirement: PKG-01
    verification:
      - kind: other
        ref: "unzip -l target/wheels/polars_statistics-*.whl | grep 'polars_statistics/_plugin/.*SKILL.md$' | wc -l -> 5"
        status: pass
    human_judgment: false
  - id: D3
    description: "Every bundled SKILL.md + ROUTING.md is byte-identical to its root plugins/polars-statistics counterpart (drift guard)"
    verification:
      - kind: other
        ref: "uv run python scripts/verify_wheel_bundling.py --drift-only -> PASS"
        status: pass
    human_judgment: false
  - id: D4
    description: "Repo-root plugin (.claude-plugin/, plugins/) unchanged; existing CI gates (ruff, pytest) stay green"
    verification:
      - kind: other
        ref: "git diff --stat <plan-head-before>..HEAD -- .claude-plugin/ plugins/ (empty); uv run ruff check . (All checks passed); uv run pytest tests/ -q (1805 passed, 6 skipped)"
        status: pass
    human_judgment: false
  - id: D5
    description: "CI lint job gains a cheap drift-only gate so stale bundled skills fail CI before release (orchestrator-added deviation)"
    verification:
      - kind: other
        ref: ".github/workflows/ci.yml lint job: 'Verify wheel-bundled plugin is not stale (drift-only)' step runs python scripts/verify_wheel_bundling.py --drift-only"
        status: pass
    human_judgment: false

duration: 15min
completed: 2026-10-15
status: complete
---

# Phase 16 Plan 01: Wheel Bundling Summary

**Claude Code plugin (marketplace + plugin manifest + 5 skill cheat-sheets) now ships inside the built `polars-statistics` wheel via a maturin python-source copy at `python/polars_statistics/_plugin/`, proven by a clean-venv install and gated against drift by a new CI lint step.**

## Performance

- **Duration:** ~15 min (includes two `maturin build --release --strip` runs, ~5.5 min first Rust compile)
- **Tasks:** 3 (3 task-scoped commits; Task 3 added no new files of its own but added a Rule 2 CI wiring deviation)
- **Files modified:** 10 (9 created, 1 modified)

## Accomplishments

- Bundled the full Claude Code plugin tree — `marketplace.json` (adjusted `source: "."`), `.claude-plugin/plugin.json`, `ROUTING.md`, and all five `SKILL.md` cheat-sheets — under `python/polars_statistics/_plugin/`, byte-identical to their `plugins/polars-statistics/` root counterparts.
- Proved end-to-end via the tracer task that maturin's python-source auto-include carries `_plugin/` (including the hidden `.claude-plugin/` dir) into the wheel archive with zero `pyproject.toml` changes.
- Added `scripts/verify_wheel_bundling.py`, a stdlib-only, ruff-clean verifier with two independent halves: Half A (drift guard — byte-identity + exactly-five-SKILL.md + marketplace.json source check, no wheel build needed) and Half B (behavioral — clean-venv `pip install` of the newest wheel + on-disk assertion of all seven required files).
- Rebuilt the wheel with the full five-skill tree and confirmed both halves pass: `PKG01_BEHAVIORAL_OK`.
- [Rule 2 deviation] Wired `scripts/verify_wheel_bundling.py --drift-only` into the CI lint job so a future skill edit under `plugins/` that isn't re-synced into `_plugin/` fails CI before release, rather than shipping stale silently.

## Task Commits

1. **Task 1: End-to-end "plugin lands in wheel" tracer** — `85f5239` (feat) — one manifest pair + one skill, confirmed present in the built wheel archive (`TRACER_OK`).
2. **Task 2: Copy remaining four skills + ROUTING.md, add drift/behavioral verifier** — `e10e763` (feat) — `COPY_AND_GUARD_OK`.
3. **Task 3 + Rule 2 deviation: full behavioral gate + CI wiring** — `73f7922` (feat) — rebuilt wheel, confirmed `PKG01_BEHAVIORAL_OK` with no verifier changes needed; added the CI drift-only step.

**Plan metadata:** (this commit, below)

## Files Created/Modified

- `python/polars_statistics/_plugin/marketplace.json` - adjusted copy of root marketplace.json (`source: "."`)
- `python/polars_statistics/_plugin/.claude-plugin/plugin.json` - byte-identical copy of root plugin manifest
- `python/polars_statistics/_plugin/skills/ROUTING.md` - byte-identical copy
- `python/polars_statistics/_plugin/skills/polars-statistics-{anova,getting-started,hypothesis-tests,regression,results}/SKILL.md` - byte-identical copies of all five skill cheat-sheets
- `scripts/verify_wheel_bundling.py` - new stdlib-only drift + behavioral verifier (`--drift-only` flag)
- `.github/workflows/ci.yml` - added drift-only verification step to the `lint` job (Rule 2 deviation)

## Decisions Made

- Copy-under-python-source (not `[tool.maturin] include`) is the only approach that lands files inside the `polars_statistics` package namespace, matching RESEARCH.md's HIGH-confidence, experimentally-verified recommendation.
- `_plugin/marketplace.json`'s `source` field is the single intentional content difference from root (`"./plugins/polars-statistics"` → `"."`), since `_plugin/` itself is now the plugin directory.
- Did not add an `__init__.py` under `_plugin/` (data directory, not a Python submodule) and did not copy the git `.gitkeep` placeholder — matches RESEARCH.md guidance exactly.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Wired the drift guard into CI**
- **Found during:** Task 3 (after the plan's three tasks completed with the gate green)
- **Issue:** The plan committed `scripts/verify_wheel_bundling.py` "for CI reuse" but never added a CI step invoking it. Without this, a later skill edit under `plugins/polars-statistics/skills/` that is not re-synced into `python/polars_statistics/_plugin/skills/` would pass CI silently and only be caught at release time (or never).
- **Fix:** Added a `Verify wheel-bundled plugin is not stale (drift-only)` step to the `lint` job in `.github/workflows/ci.yml`, immediately after `ruff check .`, running `python scripts/verify_wheel_bundling.py --drift-only` with the job's existing Python 3.12 setup (no wheel build, stdlib-only — cheap).
- **Files modified:** `.github/workflows/ci.yml`
- **Verification:** YAML parses (`yaml.safe_load`); the drift-only command itself passes locally (`uv run python scripts/verify_wheel_bundling.py --drift-only` → PASS).
- **Committed in:** `73f7922` (part of Task 3 commit)

---

**Total deviations:** 1 auto-fixed (Rule 2 — missing critical CI wiring, explicitly called out in `<context_since_planning>`).
**Impact on plan:** Closes a gap the orchestrator flagged; no scope creep beyond what was requested. All other tasks executed exactly as planned.

## Issues Encountered

None. All three tasks' automated `<verify>` blocks passed on the first attempt (`TRACER_OK`, `COPY_AND_GUARD_OK`, `PKG01_BEHAVIORAL_OK`).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- PKG-01 is fully satisfied: the plugin ships inside the wheel, proven by a clean-venv install.
- Repo-root plugin (`.claude-plugin/`, `plugins/`) is byte-for-byte unchanged and still usable for local `/plugin marketplace add`.
- No `pyproject.toml`, Cargo, or Rust source changes — existing CI gates (clippy, fmt, ruff, pytest) remain green; `ruff check .` and `pytest tests/ -q` (1805 passed, 6 skipped) both confirmed green at phase-gate time.
- Phase 17 (Release 0.8.0) can proceed: the wheel-bundling mechanism and its CI drift gate are in place before the production PyPI publish.

---
*Phase: 16-wheel-bundling*
*Completed: 2026-10-15*

## Self-Check: PASSED

All 10 claimed files found on disk; all 3 task commits (`85f5239`, `e10e763`, `73f7922`) found in git log.
