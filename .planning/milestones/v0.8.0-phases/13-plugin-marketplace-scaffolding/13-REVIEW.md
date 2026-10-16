---
phase: 13-plugin-marketplace-scaffolding
reviewed: 2026-09-20T23:30:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - .claude-plugin/marketplace.json
  - plugins/polars-statistics/.claude-plugin/plugin.json
  - plugins/polars-statistics/skills/polars-statistics-hypothesis-tests/SKILL.md
  - plugins/polars-statistics/skills/polars-statistics-anova/SKILL.md
  - plugins/polars-statistics/skills/polars-statistics-regression/SKILL.md
  - plugins/polars-statistics/skills/polars-statistics-results/SKILL.md
  - plugins/polars-statistics/skills/polars-statistics-getting-started/SKILL.md
findings:
  critical: 0
  warning: 1
  info: 1
  total: 2
status: resolved
dispositions:
  - id: WR-01
    disposition: rejected
    rationale: >-
      `repository` is a REQUIRED field per PLUG-02 success criterion ("plugin.json
      declares name, description, version, author, homepage, repository, and license")
      and was explicitly requested in CONTEXT.md. Claude Code's plugin manifest
      tolerates extra descriptive fields; the explicit phase requirement outweighs the
      unverified-field concern. Field retained.
  - id: IN-01
    disposition: fixed
    rationale: >-
      Added plugins/polars-statistics/skills/.gitkeep as a safeguard so the skills
      directory survives if stub SKILL.md files are bulk-replaced during Phase 14.
---

# Phase 13: Code Review Report

**Reviewed:** 2026-09-20T23:30:00Z
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Seven files were reviewed: two JSON manifests (`marketplace.json`, `plugin.json`) and five SKILL.md stub placeholders. The JSON is syntactically valid (confirmed with `python3 -m json.tool`), has no BOM or CRLF issues, and is structurally consistent — name, version (`0.8.0`), homepage, and license fields all match between `marketplace.json` and `plugin.json`. All five skill subdirectory names match their SKILL.md `name:` frontmatter values exactly, and the `./skills` path in `plugin.json` resolves correctly. The plugin layout correctly mirrors the pyfda reference pattern (`<source>/.claude-plugin/plugin.json`).

One warning is raised for an `extra field` in `plugin.json` that has no precedent in the verified reference schema and carries a non-trivial installation risk. One info item notes there is no `.gitkeep` in the `skills/` parent directory — all directories are currently tracked only because they each contain a `SKILL.md`, which is fine for Phase 13 but worth documenting.

---

## Warnings

### WR-01: `repository` Field in plugin.json Has No Reference Precedent — Risk of Install Failure

**File:** `plugins/polars-statistics/.claude-plugin/plugin.json:10`
**Issue:** The `repository` field is present in `plugin.json` but does not appear in the verified pyfda reference schema (`["author", "description", "homepage", "license", "name", "skills", "version"]`). The research document (13-RESEARCH.md) explicitly flags this as an unverified assumption (A1): "not part of the verified minimal schema … if it causes install errors, remove it." An unknown field in a JSON manifest that is parsed by a closed-source runtime (Claude Code's plugin installer) may cause a strict-schema parse rejection, a warning that silently aborts installation, or be harmlessly ignored. There is no test coverage for this field's behavior. Since PLUG-03 (clean installation) is the phase gate, this unknown field is a live risk until verified.

**Fix:** Remove the `repository` field from `plugin.json` for Phase 13. The information is not lost — it is already present in `marketplace.json` via `owner.url`, in the `homepage` field, and in `pyproject.toml`. If Phase 14 or a later phase confirms Claude Code accepts extension fields, re-add it then.

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
  "license": "MIT",
  "skills": "./skills"
}
```

---

## Info

### IN-01: No `.gitkeep` in `plugins/polars-statistics/skills/` — Relies on SKILL.md Content for Git Tracking

**File:** `plugins/polars-statistics/skills/` (directory)
**Issue:** The `skills/` directory and its five subdirectories are currently tracked in git only because each contains a `SKILL.md` file. This is intentional and correct for Phase 13 (the stubs serve as placeholders). However, the research document (13-RESEARCH.md, Pitfall 3) called out a `.gitkeep` in `skills/` as a safeguard against the directory disappearing if SKILL.md files are later removed or replaced during Phase 14 authoring. No `.gitkeep` was created anywhere in the `plugins/` tree. This is a low-severity consistency gap — if a Phase 14 edit tool deletes and rewrites a SKILL.md (rather than updating in place), git will track the change correctly. But if Phase 14 bulk-deletes the stubs before writing new content, the directories could temporarily fall out of the index.
**Fix:** Add a `.gitkeep` to the `skills/` parent directory to ensure the directory tree is tracked independent of file content changes:

```bash
touch plugins/polars-statistics/skills/.gitkeep
git add plugins/polars-statistics/skills/.gitkeep
```

This is an optional safeguard, not a correctness defect. No action is required before Phase 13 is closed.

---

_Reviewed: 2026-09-20T23:30:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
