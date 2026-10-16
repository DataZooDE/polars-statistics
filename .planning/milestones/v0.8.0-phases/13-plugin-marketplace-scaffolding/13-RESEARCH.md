# Phase 13: Plugin & Marketplace Scaffolding - Research

**Researched:** 2026-09-20
**Domain:** Claude Code Plugin system — marketplace.json + plugin.json + skills directory layout
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- None explicit — see Claude's Discretion.

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure phase.
The layout is fixed by the milestone decision (mirror anofox-forecast): root
`.claude-plugin/marketplace.json` + `plugins/polars-statistics/` with `plugin.json`
and a `skills/` tree. Use the anofox-forecast repo as the authoritative format
reference. Version in `plugin.json` should track the package version line (0.8.0).

### Deferred Ideas (OUT OF SCOPE)
None — infrastructure phase.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PLUG-01 | Root `.claude-plugin/marketplace.json` lists the `polars-statistics` plugin (name, source path, description, keywords), mirroring the anofox-forecast marketplace format | Exact field names captured from live pyfda reference; format verified |
| PLUG-02 | `plugins/polars-statistics/.claude-plugin/plugin.json` manifest declares name, description, version, author, homepage, repository, and license | plugin.json schema verified from live cache; all field values derived from pyproject.toml |
| PLUG-03 | The plugin installs cleanly in Claude Code — valid JSON, resolvable skill paths, and discoverable via the marketplace | Install layout verified from cached plugin at /home/simonm/.claude/plugins/cache/pyfda/fdars/0.13.0/ |
</phase_requirements>

---

## Summary

This phase creates two JSON manifest files and an empty `skills/` directory tree that form the Claude Code plugin shell for polars-statistics. No Rust, no Python, no tests — just JSON and directory structure.

The authoritative format reference is the `fdars` plugin from the `pyfda` repository, which is already installed on this machine at `/home/simonm/.claude/plugins/marketplaces/pyfda/`. Both the source layout (`pyfda/.claude-plugin/marketplace.json`, `pyfda/.claude/.claude-plugin/plugin.json`, `pyfda/.claude/skills/`) and the installed cache layout (`~/.claude/plugins/cache/pyfda/fdars/0.13.0/`) were read directly this session.

The key insight: the `plugins/polars-statistics/` directory in the repo root maps to the `.claude` source directory in pyfda. When Claude Code installs the plugin from the marketplace, it reads the `source` path from marketplace.json, navigates into that directory, reads `.claude-plugin/plugin.json`, then loads the skills subdirectory declared by the `skills` field in plugin.json.

**Primary recommendation:** Create exactly three artifacts — `.claude-plugin/marketplace.json` (root), `plugins/polars-statistics/.claude-plugin/plugin.json`, and `plugins/polars-statistics/skills/` (empty directory placeholder). The version in plugin.json must be `0.8.0` (the milestone target), not `0.7.0` (the current shipped version).

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Marketplace discovery | Root `.claude-plugin/` | — | Claude Code reads this to list the plugin in the marketplace |
| Plugin identity declaration | `plugins/polars-statistics/.claude-plugin/plugin.json` | — | Resolved from `source` path in marketplace.json |
| Skills surface | `plugins/polars-statistics/skills/` | — | Populated in Phase 14; the `skills` field in plugin.json points here |
| Package metadata (version, author, homepage) | `pyproject.toml` (source of truth) | `Cargo.toml` (mirrors same values) | These are the inputs; plugin.json copies from them |

---

## Standard Stack

This phase creates only JSON files and empty directories. There are no package installations.

### Core
| Artifact | Purpose | Source of Truth |
|----------|---------|-----------------|
| `.claude-plugin/marketplace.json` | Registers the plugin in the Claude Code marketplace | pyfda format [VERIFIED: /home/simonm/.claude/plugins/marketplaces/pyfda/.claude-plugin/marketplace.json] |
| `plugins/polars-statistics/.claude-plugin/plugin.json` | Declares plugin identity and points to skills dir | pyfda format [VERIFIED: /home/simonm/.claude/plugins/cache/pyfda/fdars/0.13.0/.claude-plugin/plugin.json] |
| `plugins/polars-statistics/skills/` | Empty directory; populated by Phase 14 | pyfda layout [VERIFIED: /home/simonm/.claude/plugins/cache/pyfda/fdars/0.13.0/skills/] |

**Installation:** No packages to install. Files are created with the Write tool.

---

## Package Legitimacy Audit

> Not applicable — this phase installs no external packages.

---

## Architecture Patterns

### System Architecture Diagram

```
polars-statistics/ (repo root)
│
├── .claude-plugin/
│   └── marketplace.json          ← registers this repo as a marketplace
│
└── plugins/
    └── polars-statistics/        ← plugin source directory (= "source" value in marketplace.json)
        ├── .claude-plugin/
        │   └── plugin.json       ← plugin identity + skills pointer
        └── skills/               ← populated in Phase 14
            ├── polars-statistics-hypothesis-tests/
            │   └── SKILL.md
            ├── polars-statistics-anova/
            │   └── SKILL.md
            ├── polars-statistics-regression/
            │   └── SKILL.md
            ├── polars-statistics-results/
            │   └── SKILL.md
            └── polars-statistics-getting-started/
                └── SKILL.md
```

When Claude Code installs the plugin:
1. Reads `marketplace.json` → finds plugin entry with `"source": "./plugins/polars-statistics"`
2. Navigates to `plugins/polars-statistics/`
3. Reads `.claude-plugin/plugin.json` → finds `"skills": "./skills"`
4. Loads each `skills/<name>/SKILL.md` file

### Recommended Project Structure
```
.claude-plugin/
└── marketplace.json

plugins/
└── polars-statistics/
    ├── .claude-plugin/
    │   └── plugin.json
    └── skills/                   ← empty directory (Phase 14 adds SKILL.md files)
```

### Pattern 1: marketplace.json Schema

**What:** Root-level marketplace registration file.
**When to use:** Every plugin marketplace has exactly one of these at the repo root under `.claude-plugin/`.
**Verbatim schema** (extracted from pyfda [VERIFIED: /home/simonm/.claude/plugins/marketplaces/pyfda/.claude-plugin/marketplace.json:1-20]):

```json
{
  "name": "pyfda",
  "description": "Claude Code plugin marketplace for fdars — functional data analysis for Python (Rust-backed). Ships the fdars-capabilities and fdars-advisor Agent Skills.",
  "owner": {
    "name": "Simon Müller",
    "email": "sm@data-zoo.de",
    "url": "https://github.com/sipemu/pyfda"
  },
  "plugins": [
    {
      "name": "fdars",
      "source": "./.claude",
      "description": "fdars functional data analysis Agent Skills: ...",
      "version": "0.13.0",
      "homepage": "https://sipemu.github.io/pyfda/",
      "license": "MIT",
      "keywords": ["functional-data-analysis", "fdars", "statistics", "skills"]
    }
  ]
}
```

**Top-level fields:** `name`, `description`, `owner` (object with `name`, `email`, `url`), `plugins` (array).
**Per-plugin entry fields:** `name`, `source` (relative path to plugin root), `description`, `version`, `homepage`, `license`, `keywords` (array of strings).

Note: The official `claude-plugins-official` marketplace.json also includes `"$schema"`, `category`, `author` (vs `owner`), and optional `tags` per plugin entry — these are used only in the official registry and are not present in the pyfda pattern being mirrored.

### Pattern 2: plugin.json Schema

**What:** Plugin identity manifest read from inside the plugin source directory.
**Location:** `<source>/.claude-plugin/plugin.json` where `<source>` is the value from marketplace.json.
**Verbatim schema** (extracted from pyfda cache [VERIFIED: /home/simonm/.claude/plugins/cache/pyfda/fdars/0.13.0/.claude-plugin/plugin.json:1-13]):

```json
{
  "name": "fdars",
  "description": "fdars functional data analysis Agent Skills (capability discovery + grounded parameter advisor)",
  "version": "0.13.0",
  "author": {
    "name": "Simon Müller",
    "email": "sm@data-zoo.de"
  },
  "homepage": "https://sipemu.github.io/pyfda/",
  "license": "MIT",
  "skills": "./skills"
}
```

**Required fields:** `name`, `description`, `version`, `author` (object with `name` and `email`), `homepage`, `license`, `skills` (relative path to skills directory).
**Note:** The CONTEXT.md also requests `repository` — this field does not appear in the pyfda reference. It may be added safely as an extension; it is not part of the verified minimal schema.

### Pattern 3: Skills Directory Layout

**What:** One subdirectory per skill under the path declared by `skills` in plugin.json.
**Convention from pyfda** [VERIFIED: /home/simonm/.claude/plugins/cache/pyfda/fdars/0.13.0/skills/]:
```
skills/
├── fdars-capabilities/
│   └── SKILL.md
└── fdars-advisor/
    ├── SKILL.md
    └── scripts/
```

Each skill subdirectory name becomes the skill name. Only `SKILL.md` is required per Phase 13; additional files (scripts, etc.) are optional.

For Phase 13 (empty shell), `skills/` must exist as a directory. The five skill subdirectories can be created with placeholder content or left for Phase 14.

### Pattern 4: SKILL.md Frontmatter

**What:** Required YAML frontmatter in each skill file.
**From pyfda fdars-capabilities SKILL.md** [VERIFIED: /home/simonm/.claude/plugins/marketplaces/pyfda/.claude/skills/fdars-capabilities/SKILL.md:1-7]:

```yaml
---
name: fdars-capabilities
description: >
  Discover the full fdars functional data analysis capability surface...
  [multi-line description]
compatibility: >
  Requires fdars installed (Python 3.9+). ...
allowed-tools: Bash Read WebFetch
---
```

**Note:** `user-invocable` is not present in the pyfda SKILL.md frontmatter — it is mentioned in `anofox-forecast/.claude/skills/forecast.md` as `user_invocable: true`. The CONTEXT.md specifies `user-invocable: false` for polars-statistics skills, mirroring the milestone decision. This field is for Phase 14; Phase 13 creates empty skill directories only.

### Anti-Patterns to Avoid

- **Placing `plugin.json` at the plugin root instead of `.claude-plugin/plugin.json`:** The system expects `<source>/.claude-plugin/plugin.json`, not `<source>/plugin.json`.
- **Using absolute paths in `source` or `skills` fields:** All paths in marketplace.json and plugin.json must be relative (`./plugins/polars-statistics`, `./skills`).
- **Setting `version` to the current shipped `0.7.0`:** This milestone ships 0.8.0 — plugin.json version must be `0.8.0` to align with REL-07.
- **Forgetting the skills directory entirely:** `plugin.json` declares `"skills": "./skills"` — if that directory does not exist, Claude Code cannot resolve skill paths (PLUG-03 fails).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Plugin format schema | Custom JSON structure | Mirror pyfda exactly — verified format |
| Skills loading | Custom loader | Claude Code reads SKILL.md natively from the declared `skills` path |

---

## Runtime State Inventory

> Not applicable — this is a greenfield phase creating new files only. No renaming, refactoring, or migration of existing state.

Nothing found in any category — verified by inspection of the polars-statistics repo root (no `.claude-plugin/` or `plugins/` directory exists yet [VERIFIED: shell ls check this session]).

---

## Common Pitfalls

### Pitfall 1: Source Path Must Resolve to Plugin Root, Not `.claude-plugin/`
**What goes wrong:** Setting `"source": "./plugins/polars-statistics/.claude-plugin"` in marketplace.json.
**Why it happens:** Confusion between the plugin root and where plugin.json lives.
**How to avoid:** Set `"source": "./plugins/polars-statistics"`. Claude Code appends `/.claude-plugin/plugin.json` internally when resolving.
**Warning signs:** Plugin fails to install; Claude Code reports "plugin.json not found."

### Pitfall 2: Version Drift Between plugin.json and Package Metadata
**What goes wrong:** plugin.json carries `0.7.0` while the milestone is bumping to `0.8.0`.
**Why it happens:** Copying the current pyproject.toml version without updating it.
**How to avoid:** REL-07 requires all version sources aligned; set plugin.json version to `0.8.0` now, matching the milestone target.
**Warning signs:** Version mismatch flagged in REL-07 checklist.

### Pitfall 3: Missing `skills/` Directory
**What goes wrong:** plugin.json has `"skills": "./skills"` but no `skills/` directory exists.
**Why it happens:** Forgetting to create the empty directory when there are no SKILL.md files yet.
**How to avoid:** Create `plugins/polars-statistics/skills/.gitkeep` (or any placeholder) to ensure the directory is tracked in git.
**Warning signs:** PLUG-03 validation fails; path resolution error when loading plugin.

### Pitfall 4: Invalid JSON in Either Manifest
**What goes wrong:** Trailing commas, comments, or unquoted strings cause parse failure.
**Why it happens:** Hand-editing JSON.
**How to avoid:** Validate with `python3 -m json.tool <file>` before committing.
**Warning signs:** Claude Code silently skips the plugin or reports parse error.

---

## Code Examples

### Exact marketplace.json for polars-statistics

```json
{
  "name": "polars-statistics",
  "description": "Claude Code plugin marketplace for polars-statistics — statistical tests and regression for Polars DataFrames (Rust-backed). Ships five Agent Skills covering hypothesis tests, ANOVA, regression, result ergonomics, and getting started.",
  "owner": {
    "name": "Simon Müller",
    "email": "sm@data-zoo.de",
    "url": "https://github.com/DataZooDE/polars-statistics"
  },
  "plugins": [
    {
      "name": "polars-statistics",
      "source": "./plugins/polars-statistics",
      "description": "polars-statistics Agent Skills: hypothesis tests, ANOVA, regression, result ergonomics, and getting started cheat-sheets for the Polars statistical plugin.",
      "version": "0.8.0",
      "homepage": "https://datazoode.github.io/polars-statistics/",
      "license": "MIT",
      "keywords": ["polars", "statistics", "hypothesis-testing", "regression", "anova", "skills"]
    }
  ]
}
```

Field values sourced from:
- `owner.name`, `owner.email`: `pyproject.toml` authors [VERIFIED: /home/simonm/projects/rust/polars-statistics/pyproject.toml:12]
- `owner.url`: `pyproject.toml` project.urls.Repository [VERIFIED: /home/simonm/projects/rust/polars-statistics/pyproject.toml:90]
- `plugins[0].homepage`: `mkdocs.yml` site_url [VERIFIED: /home/simonm/projects/rust/polars-statistics/mkdocs.yml:2]
- `plugins[0].license`: `pyproject.toml` license [VERIFIED: /home/simonm/projects/rust/polars-statistics/pyproject.toml:10]
- `plugins[0].version`: `0.8.0` — milestone target (current is `0.7.0` per pyproject.toml [VERIFIED: /home/simonm/projects/rust/polars-statistics/pyproject.toml:7])
- `plugins[0].keywords`: subset of pyproject.toml keywords [VERIFIED: /home/simonm/projects/rust/polars-statistics/pyproject.toml:14-57]

### Exact plugin.json for polars-statistics

```json
{
  "name": "polars-statistics",
  "description": "Statistical hypothesis tests, ANOVA, and regression for Polars DataFrames — Agent Skills for the polars-statistics library",
  "version": "0.8.0",
  "author": {
    "name": "Simon Müller",
    "email": "sm@data-zoo.de"
  },
  "homepage": "https://datazoode.github.io/polars-statistics/",
  "repository": "https://github.com/DataZooDE/polars-statistics",
  "license": "MIT",
  "skills": "./skills"
}
```

Note: `repository` is not in the pyfda reference schema [ASSUMED] but is an innocuous extension; it appears in the CONTEXT.md spec list. If it causes install errors, remove it — `homepage` alone is sufficient per the verified schema.

---

## Package Metadata Reference (pyproject.toml and Cargo.toml)

All values below are read directly from source files this session:

| Field | Value | Source |
|-------|-------|--------|
| Package name | `polars-statistics` | `pyproject.toml:6` [VERIFIED] |
| Current version | `0.7.0` | `pyproject.toml:7` [VERIFIED] |
| Plugin version to use | `0.8.0` | Milestone target — not yet in files |
| Description (pyproject) | `"Statistical tests, regression, and machine learning for Polars - t-tests, ANOVA, chi-square, correlation, OLS, GLM, quantile regression, and more"` | `pyproject.toml:8` [VERIFIED] |
| License | `MIT` | `pyproject.toml:10` [VERIFIED] |
| Author name | `Simon Müller` | `pyproject.toml:12` [VERIFIED] |
| Author email | `sm@data-zoo.de` | `pyproject.toml:12` [VERIFIED] |
| Homepage (docs) | `https://datazoode.github.io/polars-statistics/` | `mkdocs.yml:2` [VERIFIED] |
| Repository | `https://github.com/DataZooDE/polars-statistics` | `pyproject.toml:90`, `Cargo.toml:8` [VERIFIED] |

---

## Existing State Verification

- **`.claude-plugin/` directory:** Does not exist in polars-statistics root [VERIFIED: shell check this session]
- **`plugins/` directory:** Does not exist in polars-statistics root [VERIFIED: shell check this session]
- **Version sources for 0.7.0 (current):** `pyproject.toml:7`, `Cargo.toml:4`, `python/polars_statistics/__init__.py` [VERIFIED]
- **No existing plugin or skills infrastructure to migrate or conflict with**

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Skills in `.claude/skills/` (direct, no marketplace) | Skills in `plugins/<name>/skills/` with marketplace.json | Installable via Claude Code `/add-marketplace`; distributable to users |
| `user_invocable: true` (anofox-forecast uses this) | `user-invocable: false` (auto-surfaced) | Milestone decision — polars-statistics skills are auto-surfaced cheat-sheets, not slash commands |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `repository` field in plugin.json is a harmless extension that Claude Code ignores if unrecognized | Code Examples | Low — worst case it causes a warning; remove field if install fails |
| A2 | Creating an empty `skills/` directory (with a `.gitkeep` placeholder) satisfies PLUG-03 validation | Architecture Patterns | Medium — if Claude Code validates that skills exist at install time, need at least stub SKILL.md files |

---

## Open Questions

1. **Does PLUG-03 require skill files to exist at install time, or is an empty `skills/` directory acceptable?**
   - What we know: The pyfda reference has two SKILL.md files populated; we have no example of an empty skills dir being installed
   - What's unclear: Whether Claude Code validates skill file existence during install vs. lazy-loads
   - Recommendation: Create the five empty skill subdirectories with minimal placeholder SKILL.md files (just frontmatter) to be safe. Phase 14 replaces their content.

---

## Environment Availability

> Not applicable — this phase creates only JSON files and directories. No external tools, services, or runtimes are required beyond a text editor / Write tool.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 7.0+ (configured in pyproject.toml) |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `python3 -m json.tool .claude-plugin/marketplace.json && python3 -m json.tool plugins/polars-statistics/.claude-plugin/plugin.json` |
| Full suite command | `python3 -m pytest tests/ -v --tb=short` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PLUG-01 | marketplace.json is valid JSON with required fields | smoke | `python3 -m json.tool .claude-plugin/marketplace.json` | ❌ Wave 0 |
| PLUG-02 | plugin.json is valid JSON with required fields | smoke | `python3 -m json.tool plugins/polars-statistics/.claude-plugin/plugin.json` | ❌ Wave 0 |
| PLUG-03 | Skills path in plugin.json resolves to existing directory | smoke | `python3 -c "import json,os; d=json.load(open('plugins/polars-statistics/.claude-plugin/plugin.json')); assert os.path.isdir('plugins/polars-statistics/'+d['skills'].lstrip('./')), 'skills dir missing'"` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** JSON validity check (`python3 -m json.tool` on both files)
- **Per wave merge:** All three smoke checks above
- **Phase gate:** All checks green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_plugin_manifest.py` — covers PLUG-01, PLUG-02, PLUG-03 JSON validation and path resolution

*(Alternatively, the JSON smoke commands above are sufficient for this infrastructure-only phase and a dedicated test file may be overkill.)*

---

## Security Domain

> `security_enforcement: true` in config.json; ASVS level 1.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | — |
| V3 Session Management | No | — |
| V4 Access Control | No | — |
| V5 Input Validation | No | JSON files have no user input |
| V6 Cryptography | No | — |

**Security assessment:** This phase creates static JSON configuration files with no code execution, no user input handling, no network calls, and no secrets. ASVS categories do not apply. The only security consideration is ensuring the JSON files do not accidentally embed secrets (author email is public metadata; no API keys or tokens are involved).

---

## Sources

### Primary (HIGH confidence)
- `/home/simonm/.claude/plugins/marketplaces/pyfda/.claude-plugin/marketplace.json` — marketplace.json schema (read this session)
- `/home/simonm/.claude/plugins/cache/pyfda/fdars/0.13.0/.claude-plugin/plugin.json` — plugin.json schema (read this session)
- `/home/simonm/.claude/plugins/cache/pyfda/fdars/0.13.0/skills/` — skills directory layout (read this session)
- `/home/simonm/projects/rust/polars-statistics/pyproject.toml` — package metadata source of truth (read this session)
- `/home/simonm/projects/rust/polars-statistics/Cargo.toml` — confirms name, version, repository (read this session)
- `/home/simonm/projects/rust/polars-statistics/mkdocs.yml` — homepage URL (read this session)

### Secondary (MEDIUM confidence)
- `/home/simonm/.claude/plugins/marketplaces/claude-plugins-official/.claude-plugin/marketplace.json` — official marketplace format for comparison (read this session)
- `/home/simonm/.claude/plugins/known_marketplaces.json` — marketplace registration format (read this session)
- `/home/simonm/.claude/plugins/installed_plugins.json` — install cache path pattern (read this session)

---

## Metadata

**Confidence breakdown:**
- File layout and schema: HIGH — read live files from installed pyfda plugin this session
- Package metadata values: HIGH — read directly from pyproject.toml, Cargo.toml, mkdocs.yml this session
- `repository` field in plugin.json: LOW [ASSUMED] — not in verified schema; safe extension
- PLUG-03 empty-skills behavior: LOW [ASSUMED] — no empty-skills install example to verify against

**Research date:** 2026-09-20
**Valid until:** Stable (Claude Code plugin format changes infrequently; re-verify if `/add-marketplace` behavior changes)
