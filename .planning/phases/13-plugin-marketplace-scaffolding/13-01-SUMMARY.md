---
phase: 13-plugin-marketplace-scaffolding
plan: "01"
subsystem: infra
tags: [claude-code-plugin, marketplace, plugin-manifest, skills]

# Dependency graph
requires: []
provides:
  - Root .claude-plugin/marketplace.json registering the polars-statistics plugin (PLUG-01)
  - plugins/polars-statistics/.claude-plugin/plugin.json with v0.8.0 identity (PLUG-02)
  - Five placeholder skill directories with stub SKILL.md files (PLUG-03)
  - Full Claude Code plugin discovery chain: marketplace -> source -> plugin.json -> skills
affects:
  - 14-skill-cheat-sheets
  - 16-wheel-packaging

# Actuals (#2632)
actuals:
  tokens: 1246
  tasks: 2
  commits: 1
  plan_head_before: 431306d73485f54d1c278fab2de4fdb951700ba8

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Claude Code plugin layout: root .claude-plugin/marketplace.json + plugins/<name>/.claude-plugin/plugin.json + skills/"
    - "Skill stub: minimal YAML frontmatter (name, description) + one-line body noting Phase 14 ownership"

key-files:
  created:
    - .claude-plugin/marketplace.json
    - plugins/polars-statistics/.claude-plugin/plugin.json
    - plugins/polars-statistics/skills/polars-statistics-hypothesis-tests/SKILL.md
    - plugins/polars-statistics/skills/polars-statistics-anova/SKILL.md
    - plugins/polars-statistics/skills/polars-statistics-regression/SKILL.md
    - plugins/polars-statistics/skills/polars-statistics-results/SKILL.md
    - plugins/polars-statistics/skills/polars-statistics-getting-started/SKILL.md
  modified: []

key-decisions:
  - "Mirrored pyfda/fdars layout exactly: source path is plugin root (not .claude-plugin subdir), skills field is ./skills"
  - "Version set to 0.8.0 (milestone target) in both manifests; pyproject.toml/Cargo.toml left at 0.7.0 per scope guardrail"
  - "Added repository field to plugin.json as innocuous extension (not in pyfda minimal schema but requested in CONTEXT.md)"
  - "Created five stub SKILL.md files with minimal frontmatter to satisfy PLUG-03 path resolution at install time"

patterns-established:
  - "Pattern 1: marketplace.json source path -> plugin ROOT (Claude Code appends /.claude-plugin/plugin.json internally)"
  - "Pattern 2: All manifest paths are relative (./plugins/polars-statistics, ./skills) -- never absolute"
  - "Pattern 3: Stub SKILL.md carries only name + description frontmatter; no user-invocable or version fields (Phase 14 owns those)"

requirements-completed: [PLUG-01, PLUG-02, PLUG-03]

coverage:
  - id: D1
    description: "marketplace.json is valid JSON listing the polars-statistics plugin with name, source, description, version, homepage, license, and keywords"
    requirement: PLUG-01
    verification:
      - kind: other
        ref: "python3 -m json.tool .claude-plugin/marketplace.json > /dev/null && python3 -c \"import json; m=json.load(open('.claude-plugin/marketplace.json')); assert any(e['name']=='polars-statistics' for e in m['plugins'])\""
        status: pass
    human_judgment: false
  - id: D2
    description: "plugin.json is valid JSON declaring name, description, version 0.8.0, author, homepage, repository, license, and skills path"
    requirement: PLUG-02
    verification:
      - kind: other
        ref: "python3 -m json.tool plugins/polars-statistics/.claude-plugin/plugin.json > /dev/null"
        status: pass
    human_judgment: false
  - id: D3
    description: "Full discovery chain resolves end-to-end: marketplace -> source -> plugin.json -> skills dir; both versions read 0.8.0"
    requirement: PLUG-03
    verification:
      - kind: other
        ref: "python3 -c \"import json,os; m=json.load(open('.claude-plugin/marketplace.json')); p=json.load(open('plugins/polars-statistics/.claude-plugin/plugin.json')); assert any(e['name']=='polars-statistics' for e in m['plugins']); src=m['plugins'][0]['source'].lstrip('./'); assert os.path.isfile(os.path.join(src,'.claude-plugin','plugin.json')); assert os.path.isdir(os.path.join('plugins/polars-statistics', p['skills'].lstrip('./'))); assert m['plugins'][0]['version']=='0.8.0' and p['version']=='0.8.0'; print('CHAIN_RESOLVES')\""
        status: pass
    human_judgment: false
  - id: D4
    description: "Plugin installs and is discoverable via marketplace in live Claude Code (manual only)"
    requirement: PLUG-03
    verification: []
    human_judgment: true
    rationale: "Requires a live Claude Code /add-marketplace + install action; cannot be automated in CI"

# Metrics
duration: 2min
completed: "2026-09-20"
status: complete
---

# Phase 13 Plan 01: Plugin & Marketplace Scaffolding Summary

**Root marketplace.json + plugin.json identity manifest + five stub SKILL.md files forming an installable Claude Code plugin shell for polars-statistics v0.8.0**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-09-20T21:13:55Z
- **Completed:** 2026-09-20T21:15:11Z
- **Tasks:** 2 (1 tracer + 1 validation-only)
- **Files modified:** 7 created, 0 modified

## Accomplishments

- Created `.claude-plugin/marketplace.json` registering the polars-statistics plugin (PLUG-01)
- Created `plugins/polars-statistics/.claude-plugin/plugin.json` with full identity (name, description, v0.8.0, author, homepage, repository, license, skills pointer) (PLUG-02)
- Created five placeholder skill directories under `plugins/polars-statistics/skills/`, each with a stub SKILL.md that satisfies path resolution at install time (PLUG-03)
- Full discovery chain validated: `CHAIN_RESOLVES` -- marketplace -> source -> plugin.json -> skills all resolve; both manifests carry version 0.8.0

## Task Commits

1. **Task 1: Create both manifests + skills tree** - `3bccc03` (feat)
2. **Task 2: Validate full discovery chain** - validation-only, no new commit (CHAIN_RESOLVES confirmed)

**Plan metadata:** committed with docs commit below

## Files Created/Modified

- `.claude-plugin/marketplace.json` - Root marketplace registration (PLUG-01); lists plugin name, source `./plugins/polars-statistics`, description, v0.8.0, homepage, license, keywords
- `plugins/polars-statistics/.claude-plugin/plugin.json` - Plugin identity manifest (PLUG-02); declares name, description, v0.8.0, author, homepage, repository, license, skills `./skills`
- `plugins/polars-statistics/skills/polars-statistics-hypothesis-tests/SKILL.md` - Phase 14 placeholder
- `plugins/polars-statistics/skills/polars-statistics-anova/SKILL.md` - Phase 14 placeholder
- `plugins/polars-statistics/skills/polars-statistics-regression/SKILL.md` - Phase 14 placeholder
- `plugins/polars-statistics/skills/polars-statistics-results/SKILL.md` - Phase 14 placeholder
- `plugins/polars-statistics/skills/polars-statistics-getting-started/SKILL.md` - Phase 14 placeholder

## Decisions Made

- Mirrored pyfda/fdars layout exactly: `source` in marketplace.json points to the plugin root (`./plugins/polars-statistics`), not the `.claude-plugin` subdir. Claude Code appends `/.claude-plugin/plugin.json` internally.
- Version set to `0.8.0` (milestone target) in both manifests. Source files (`pyproject.toml`, `Cargo.toml`, `__init__.py`) remain at `0.7.0` -- the real version bump is Phase 17 / REL-07.
- `repository` field added to plugin.json as a harmless extension (not in pyfda minimal schema but specified in CONTEXT.md). Per RESEARCH Assumption A1, Claude Code ignores unrecognized fields.
- Five stub SKILL.md files created with only `name` and `description` frontmatter; no `user-invocable`, `version`, or content (Phase 14 owns those).

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None -- no external service configuration required. The plugin manifests are ready for local installation via Claude Code `/add-marketplace` (manual verification, out-of-band per VALIDATION.md).

## Known Stubs

| File | Description |
|------|-------------|
| `plugins/polars-statistics/skills/polars-statistics-hypothesis-tests/SKILL.md` | Placeholder only; real cheat-sheet authored in Phase 14 (SKILL-01) |
| `plugins/polars-statistics/skills/polars-statistics-anova/SKILL.md` | Placeholder only; real cheat-sheet authored in Phase 14 (SKILL-02) |
| `plugins/polars-statistics/skills/polars-statistics-regression/SKILL.md` | Placeholder only; real cheat-sheet authored in Phase 14 (SKILL-03) |
| `plugins/polars-statistics/skills/polars-statistics-results/SKILL.md` | Placeholder only; real cheat-sheet authored in Phase 14 (SKILL-04) |
| `plugins/polars-statistics/skills/polars-statistics-getting-started/SKILL.md` | Placeholder only; real cheat-sheet authored in Phase 14 (SKILL-05) |

These stubs are intentional -- Phase 13 scope is the installable shell only. Phase 14 (SKILL-01..06) replaces the placeholder content with real cheat-sheets.

## Threat Flags

No new security-relevant surface introduced. Both manifests contain only public package metadata (name, email, URLs already in pyproject.toml). T-13-02 mitigated: JSON validity verified by `python3 -m json.tool` gate on both files before commit.

## Next Phase Readiness

- Phase 14 (skill cheat-sheets): Plugin shell is ready. Phase 14 writes real SKILL.md content directly into the five skill subdirectories already created here.
- Phase 16 (wheel packaging): `plugins/polars-statistics/` directory exists and can be declared as package data in maturin configuration.
- Phase 17 (release): plugin.json `version` field pre-declares `0.8.0`; when REL-07 runs the source version bump, no conflict since both will read `0.8.0`.

---
*Phase: 13-plugin-marketplace-scaffolding*
*Completed: 2026-09-20*
