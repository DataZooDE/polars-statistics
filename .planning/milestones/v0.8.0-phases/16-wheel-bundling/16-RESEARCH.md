# Phase 16: Wheel Bundling - Research

**Researched:** 2026-09-22
**Domain:** Python packaging (maturin wheel bundling, pyproject.toml)
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
None locked — pure infrastructure phase at Claude's discretion.

### Claude's Discretion
- Determine the maturin mechanism to bundle the root-level plugin tree into the wheel (`[tool.maturin] include` / copy under python-source), and WHERE the files land in the installed package.
- Mirror the pyfda/fdars approach where applicable.
- Decide whether to use `include` (root-relative paths) or copy under `python/polars_statistics/` so maturin picks them up automatically.
- Prefer the least-disruptive option that keeps the repo-root plugin working for local marketplace add AND lands the files in the wheel.
- Must not break the existing wheel build or CI gates (clippy, fmt, ruff, pytest).

### Deferred Ideas (OUT OF SCOPE)
None — infrastructure phase.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PKG-01 | Skill files are bundled into the built wheel via maturin package data and are present after `pip install polars-statistics` | Copy plugin tree under `python/polars_statistics/_plugin/` — maturin auto-includes all files under python-source; behavioral check via `unzip -l` + clean-venv install |
</phase_requirements>

---

## Summary

The recommended approach is to copy the plugin tree into `python/polars_statistics/_plugin/`, which maturin automatically includes in the wheel at `polars_statistics/_plugin/...` (the `python/` prefix is stripped). This requires **zero changes to `pyproject.toml`** and zero changes to any existing plugin file except creating a new `_plugin/marketplace.json` with an adjusted `source` path.

The alternative — using `[tool.maturin] include` with project-root-relative globs — was experimentally confirmed to work, but files land at `site-packages/plugins/...` and `site-packages/.claude-plugin/...` (not inside the `polars_statistics` package namespace). That layout is incorrect for a wheel-bundled asset: loose top-level files in site-packages are fragile (collide across packages, not namespaced, harder to locate programmatically).

The `python/polars_statistics/_plugin/` layout was experimentally verified by building a test wheel and inspecting its archive with `unzip -l`. The `.claude-plugin/` hidden directory inside the python-source tree is picked up by maturin without any special configuration.

**Primary recommendation:** Copy the plugin tree under `python/polars_statistics/_plugin/` with a new `_plugin/marketplace.json` (source path adjusted to `"."`). No `pyproject.toml` changes. No changes to existing plugin files.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Plugin tree bundling | Build / packaging (pyproject.toml / file layout) | — | Files must be present on disk after pip install; maturin controls wheel contents |
| Plugin discovery (post-install) | Python package layer (`polars_statistics.__file__`) | — | `pathlib.Path(polars_statistics.__file__).parent / '_plugin'` is the canonical locator |
| Local marketplace add (dev) | Repo root `.claude-plugin/marketplace.json` | — | Unchanged — repo root plugin keeps pointing at `plugins/polars-statistics/` |
| Wheel content verification | CI / build verification script | — | `unzip -l` fast-check + clean-venv behavioral assert |

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| maturin | 1.13.3 [VERIFIED: `uv run maturin --version`] | Build Python wheel from Rust | Already the project's build backend |
| Python stdlib `pathlib` | 3.9+ [VERIFIED: pyproject.toml:11] | Locate `_plugin/` from installed package | No extra deps; `Path(__file__).parent / '_plugin'` |
| Python stdlib `unzip` | system | Inspect wheel archive without install | Fast verification — no Rust compile needed |

### No New Dependencies

This phase requires no new Python or Rust dependencies. The change is:
1. Copy files into the source tree under `python/polars_statistics/_plugin/`
2. maturin picks them up automatically at build time
3. No `pyproject.toml` changes required

---

## Package Legitimacy Audit

No new packages are introduced in this phase.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| (none) | — | — | — | — | — | — |

---

## Architecture Patterns

### System Architecture Diagram

```
Repo root
├── .claude-plugin/marketplace.json        ← LOCAL DEV only (unchanged)
│   "source": "./plugins/polars-statistics"
└── plugins/polars-statistics/
    ├── .claude-plugin/plugin.json          ← source of truth for plugin manifest
    └── skills/*/SKILL.md                   ← source of truth for skills

    COPY (phase 16 task)
    ▼
python/polars_statistics/_plugin/           ← maturin python-source tree
├── marketplace.json                        ← NEW: source = "." (adjusted)
├── .claude-plugin/plugin.json              ← COPY: unchanged
└── skills/
    ├── ROUTING.md                          ← COPY: unchanged
    ├── polars-statistics-anova/SKILL.md
    ├── polars-statistics-getting-started/SKILL.md
    ├── polars-statistics-hypothesis-tests/SKILL.md
    ├── polars-statistics-regression/SKILL.md
    └── polars-statistics-results/SKILL.md

    maturin build --release --strip
    ▼
target/wheels/polars_statistics-0.7.0-*.whl
    polars_statistics/_plugin/              ← python/ prefix stripped automatically
    ├── marketplace.json
    ├── .claude-plugin/plugin.json
    └── skills/*/SKILL.md

    pip install <wheel>
    ▼
site-packages/polars_statistics/_plugin/    ← discoverable as:
    marketplace.json                         pathlib.Path(polars_statistics.__file__).parent / '_plugin'
    .claude-plugin/plugin.json
    skills/*/SKILL.md
```

### Recommended Project Structure (added files only)

```
python/polars_statistics/
└── _plugin/                  # bundled Claude Code plugin tree
    ├── marketplace.json      # adjusted: "source": "."
    ├── .claude-plugin/
    │   └── plugin.json       # copy of plugins/polars-statistics/.claude-plugin/plugin.json
    └── skills/
        ├── ROUTING.md
        ├── polars-statistics-anova/SKILL.md
        ├── polars-statistics-getting-started/SKILL.md
        ├── polars-statistics-hypothesis-tests/SKILL.md
        ├── polars-statistics-regression/SKILL.md
        └── polars-statistics-results/SKILL.md
```

### Pattern 1: maturin python-source auto-include

**What:** maturin includes ALL files under the `python-source` tree in the wheel, stripping the `python/` prefix. Non-Python files (`.json`, `.md`), hidden directories (`.claude-plugin/`), and arbitrary subdirectories are all included without any explicit `include` config.

**When to use:** Any time you need to bundle static assets alongside a PyO3 wheel. Files go under `python/<package_name>/<subdir>/` and land at `<package_name>/<subdir>/` in the wheel.

**Experimentally verified** by:
1. Creating `python/polars_statistics/_plugin/.claude-plugin/plugin.json`, `_plugin/marketplace.json`, `_plugin/skills/test-skill/SKILL.md`
2. Running `uv run maturin build --release --strip`
3. Checking `unzip -l target/wheels/polars_statistics-0.7.0-*.whl | grep _plugin`:

```
15  1980-01-01 00:00   polars_statistics/_plugin/.claude-plugin/plugin.json
15  1980-01-01 00:00   polars_statistics/_plugin/marketplace.json
 7  1980-01-01 00:00   polars_statistics/_plugin/skills/test-skill/SKILL.md
```

[VERIFIED: local build experiment, this session]

**Source:** `[VERIFIED: maturin 1.13.3 build experiment]`

### Pattern 2: pyproject.toml `include` (rejected for this phase)

**What:** `[tool.maturin] include = [{ path = "plugins/**/*", format = "wheel" }]` pulls files from outside the python-source tree into the wheel.

**Critical limitation verified:** Files land at their project-root-relative path, NOT inside the package namespace:
- `plugins/polars-statistics/.claude-plugin/plugin.json` → `plugins/polars-statistics/.claude-plugin/plugin.json` in site-packages
- `.claude-plugin/marketplace.json` → `.claude-plugin/marketplace.json` in site-packages

These would be loose top-level files outside `polars_statistics/` — not discoverable via `polars_statistics.__path__` and likely to collide across packages.

[VERIFIED: local build experiment, `unzip -l` confirmed path, this session]

**Reject: do not use** for this phase.

### Pattern 3: `marketplace.json` path adjustment

The root `.claude-plugin/marketplace.json` has `"source": "./plugins/polars-statistics"` — pointing from the repo root to the plugin directory. The bundled `_plugin/marketplace.json` needs `"source": "."` because `marketplace.json` will be at the same directory level as `.claude-plugin/` (the plugin's manifest directory).

The `plugin.json` `"skills": "./skills"` path is unchanged — `skills/` is adjacent to `.claude-plugin/` in both the repo layout and the `_plugin/` layout.

[VERIFIED: file structure inspection, this session — see `.claude-plugin/marketplace.json:10` and `plugins/polars-statistics/.claude-plugin/plugin.json:12`]

### Anti-Patterns to Avoid

- **Using `include` globs for non-python-source files:** Files land outside the package namespace at top-level site-packages paths. Fragile, not namespaced, breaks the `Path(__file__).parent / '_plugin'` discovery pattern.
- **Symlinking `plugins/` into `python/polars_statistics/`:** Symlinks are not followed by maturin's file walker in all cases and break on Windows. Use real file copies.
- **Adding `__init__.py` to `_plugin/`:** Not needed — `_plugin/` is a data directory, not a Python package. Adding `__init__.py` would expose it as a Python submodule unnecessarily.
- **Modifying `plugin.json` `"skills"` path:** It already points to `"./skills"` relative to `.claude-plugin/`, which resolves to `_plugin/skills/` in both repo and wheel contexts. No change needed.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| File path discovery post-install | Custom env-var lookup | `pathlib.Path(polars_statistics.__file__).parent / '_plugin'` | Works for editable installs and regular installs; `__file__` is always correct |
| Wheel content verification | Writing a custom wheel parser | `unzip -l <wheel> \| grep _plugin` | Wheels are ZIP archives; `unzip` is sufficient for content listing |
| Bundling logic | Custom build script / Makefile | Copy files into python-source tree; maturin auto-includes | maturin handles everything without configuration |

---

## Runtime State Inventory

Not applicable — this is a packaging configuration phase, not a rename/refactor phase.

---

## Common Pitfalls

### Pitfall 1: Stale compiled wheel in `target/wheels/`

**What goes wrong:** `unzip -l` shows the old wheel (without `_plugin/`) because `maturin build` uses a cached artifact and does not rebuild when only Python/data files change.

**Why it happens:** The Rust compilation result is cached; adding new Python-side files triggers a new wheel package but requires `maturin build` (not `maturin develop`) to produce the actual `.whl` file.

**How to avoid:** Always run `uv run maturin build --release --strip` (not `maturin develop`) to produce the `.whl` artifact, then inspect `target/wheels/polars_statistics-*.whl` (the newest one by version/mtime).

**Warning signs:** `unzip -l` shows a wheel with a timestamp older than the file copy step.

### Pitfall 2: `.claude-plugin/` hidden directory silently excluded

**What goes wrong:** maturin might skip hidden directories (dot-prefixed) under python-source.

**Why it happens:** Some packaging tools skip `.`-prefixed directories by default.

**How to avoid:** This was experimentally verified NOT to be an issue for maturin 1.13.3 — `.claude-plugin/plugin.json` inside `_plugin/` was confirmed present in the wheel. The `unzip -l` check in the verification procedure catches this if behavior changes.

**Warning signs:** `unzip -l | grep '.claude-plugin'` returns empty.

### Pitfall 3: Wheel inspect passes but install check fails

**What goes wrong:** `unzip -l` shows files, but `Path(polars_statistics.__file__).parent / '_plugin'` doesn't exist after pip install.

**Why it happens:** The wheel was inspected but the install was done into a different environment, or the `maturin develop` path (in-place `.pth` file) is being used instead of the wheel.

**How to avoid:** Use the clean-venv install in the behavioral verification step. The install check in the behavioral script explicitly uses `sys.prefix` from the venv, not the dev environment.

### Pitfall 4: `include` globs with `plugins/**/*` pull in `.gitkeep`

**What goes wrong:** The `.gitkeep` placeholder in `plugins/polars-statistics/skills/` would be bundled if using the `include` approach.

**Why it happens:** `**/*` matches all files including empty placeholders.

**How to avoid:** Not applicable for the recommended approach (copy under python-source); the `.gitkeep` is not copied as part of the task. If the copy script uses `cp -r`, explicitly exclude `.gitkeep`.

---

## Code Examples

### Verified: pyproject.toml (no changes needed)

```toml
# Source: this project's pyproject.toml — NO CHANGES REQUIRED
[tool.maturin]
python-source = "python"
module-name = "polars_statistics._polars_statistics"
bindings = "pyo3"
features = ["pyo3/extension-module", "python"]
strip = true
# No `include` directive needed — all files under python/polars_statistics/ are auto-included
```

[VERIFIED: maturin 1.13.3 build experiment, this session]

### New file: `python/polars_statistics/_plugin/marketplace.json`

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
      "source": ".",
      "description": "polars-statistics Agent Skills: hypothesis tests, ANOVA, regression, result ergonomics, and getting started cheat-sheets for the Polars statistical plugin.",
      "version": "0.8.0",
      "homepage": "https://datazoode.github.io/polars-statistics/",
      "license": "MIT",
      "keywords": ["polars", "statistics", "hypothesis-testing", "regression", "anova", "skills"]
    }
  ]
}
```

Key difference from root `.claude-plugin/marketplace.json`: `"source": "."` (plugin dir is the same directory as marketplace.json, not `"./plugins/polars-statistics"`).

[VERIFIED: derived from `.claude-plugin/marketplace.json:10` + path analysis, this session]

### Locating the plugin post-install

```python
# Source: Python stdlib pathlib — standard pattern
import pathlib
import polars_statistics

plugin_dir = pathlib.Path(polars_statistics.__file__).parent / "_plugin"
marketplace = plugin_dir / "marketplace.json"
plugin_manifest = plugin_dir / ".claude-plugin" / "plugin.json"
```

### File copy commands (bash)

```bash
# Create _plugin tree under python-source
REPO_ROOT="/path/to/polars-statistics"  # or $(git rev-parse --show-toplevel)
DEST="$REPO_ROOT/python/polars_statistics/_plugin"

mkdir -p "$DEST/.claude-plugin"
mkdir -p "$DEST/skills"

# Copy plugin manifest (unchanged)
cp "$REPO_ROOT/plugins/polars-statistics/.claude-plugin/plugin.json" \
   "$DEST/.claude-plugin/plugin.json"

# Copy skills (rsync or cp -r, exclude .gitkeep)
cp -r "$REPO_ROOT/plugins/polars-statistics/skills/polars-statistics-anova"       "$DEST/skills/"
cp -r "$REPO_ROOT/plugins/polars-statistics/skills/polars-statistics-getting-started" "$DEST/skills/"
cp -r "$REPO_ROOT/plugins/polars-statistics/skills/polars-statistics-hypothesis-tests" "$DEST/skills/"
cp -r "$REPO_ROOT/plugins/polars-statistics/skills/polars-statistics-regression"   "$DEST/skills/"
cp -r "$REPO_ROOT/plugins/polars-statistics/skills/polars-statistics-results"      "$DEST/skills/"
cp "$REPO_ROOT/plugins/polars-statistics/skills/ROUTING.md" "$DEST/skills/"

# Create adjusted marketplace.json (source = ".")
# Note: do NOT copy root .claude-plugin/marketplace.json directly (source path differs)
# Write _plugin/marketplace.json with source = "." — see template above
```

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (existing) + bash verification script |
| Config file | `pyproject.toml` lines 103-105 |
| Quick run command | `unzip -l target/wheels/polars_statistics-*.whl \| grep -c _plugin` |
| Full suite command | See behavioral check below |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PKG-01 | Plugin files present after pip install | behavioral | `python scripts/verify_wheel_bundling.py` (Wave 0 gap) | ❌ Wave 0 |
| PKG-01 | Wheel archive contains `_plugin/` | smoke | `unzip -l target/wheels/polars_statistics-*.whl \| grep -c 'polars_statistics/_plugin/'` | ❌ inline cmd |

### Concrete Verification Procedure

**Step 1 — Build the wheel** (required; compiles Rust, ~5 minutes first time, ~10 seconds with cache):

```bash
uv run maturin build --release --strip
WHEEL=$(ls -t target/wheels/polars_statistics-*.whl | head -1)
echo "Checking: $WHEEL"
```

**Step 2 — Fast check: inspect archive contents** (no install, < 1 second):

```bash
echo "=== Fast archive check ==="
unzip -l "$WHEEL" | grep "_plugin/"
unzip -l "$WHEEL" | grep "marketplace.json"       && echo "OK: marketplace.json present"
unzip -l "$WHEEL" | grep ".claude-plugin/plugin.json" && echo "OK: plugin.json present"
COUNT=$(unzip -l "$WHEEL" | grep -c "SKILL.md")
[ "$COUNT" -eq 5 ] && echo "OK: 5 SKILL.md files" || echo "FAIL: expected 5 SKILL.md, got $COUNT"
```

**Step 3 — Behavioral check: clean-venv install + disk assertion** (behavioral, ~30 seconds):

```bash
echo "=== Behavioral check ==="
VENV=$(mktemp -d)/venv
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet "$WHEEL"
"$VENV/bin/python" - << 'PY'
import pathlib, sys
import polars_statistics
pkg = pathlib.Path(polars_statistics.__file__).parent
required = [
    "_plugin/marketplace.json",
    "_plugin/.claude-plugin/plugin.json",
    "_plugin/skills/polars-statistics-anova/SKILL.md",
    "_plugin/skills/polars-statistics-getting-started/SKILL.md",
    "_plugin/skills/polars-statistics-hypothesis-tests/SKILL.md",
    "_plugin/skills/polars-statistics-regression/SKILL.md",
    "_plugin/skills/polars-statistics-results/SKILL.md",
]
missing = [p for p in required if not (pkg / p).exists()]
if missing:
    print("FAIL — missing files:", missing)
    sys.exit(1)
print("OK — all plugin files present under", pkg / "_plugin")
PY
rm -rf "$VENV"
```

**Why two steps:** Step 2 is fast (< 1 s, catches most issues early without waiting for venv creation). Step 3 is the behavioral criterion required by the CONTEXT.md spec and verifies the exact on-disk presence after a real `pip install`. Both steps together constitute the PKG-01 verification.

### Sampling Rate

- **Per task commit:** Step 2 fast check (`unzip -l` grep count)
- **Per wave merge:** Steps 2 + 3 (full behavioral)
- **Phase gate:** Steps 2 + 3 must both pass before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `scripts/verify_wheel_bundling.py` — covers PKG-01 behavioral check (wraps Step 3 above as a standalone script)

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| maturin | Wheel build | ✓ | 1.13.3 [VERIFIED: `uv run maturin --version`] | — |
| uv | Build runner | ✓ | present in PATH [VERIFIED: this session] | `pip install maturin` directly |
| unzip | Fast archive check | ✓ | system [VERIFIED: used in session] | `python -c "import zipfile; ..."` |
| python3 venv | Behavioral install check | ✓ | Python 3.14 on dev machine [VERIFIED: this session] | Any Python ≥ 3.9 |
| Rust stable toolchain | Wheel compilation | ✓ | present (build succeeded) [VERIFIED: this session] | — |

---

## Security Domain

Not applicable — this phase adds static JSON and Markdown data files to the wheel. No authentication, cryptography, session management, or user input processing involved.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual MANIFEST.in for sdist data files | `[tool.maturin] include` or python-source auto-include | maturin 0.x → 1.x | Simpler, no separate MANIFEST.in needed |
| `setup.py` `package_data` for Python package assets | Files under python-source tree auto-included by maturin | maturin adoption | Zero config for in-package assets |
| `include` with `to` for path remapping | `to` field only works with `from = "out-dir"` (build artifacts) | maturin 1.12.1 | Path remapping for regular source files is NOT supported via `include`; use python-source placement instead |

**Key state-of-art finding:** The `to` field in `[tool.maturin] include` is exclusively for OUT_DIR build artifacts (introduced in maturin 1.12.1). There is no `to` equivalent for source file `include` entries. The only path-remapping mechanism for source files is to physically place them at the desired path within the python-source tree.

[VERIFIED: maturin GlobPattern source `WithOutDir` struct analysis; maturin changelog 1.12.1]

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The `_plugin/` directory name is a good choice — no conflict with existing polars_statistics namespace or future submodule | Architecture Patterns | Low: rename `_plugin` if a clash emerges. No functional dependency. |
| A2 | `plugin.json` `"skills": "./skills"` path resolves correctly when plugin.json is at `_plugin/.claude-plugin/plugin.json` and skills are at `_plugin/skills/` | Architecture Patterns | Medium: if Claude Code resolves `"./skills"` relative to `plugin.json`'s directory (`.claude-plugin/`), then `../skills` would be needed — but this matches the existing repo layout where it works, so assumption is LOW risk. |

**On A2:** The existing repo has `plugins/polars-statistics/.claude-plugin/plugin.json` with `"skills": "./skills"`. The skills are at `plugins/polars-statistics/skills/`. This means `./skills` is resolved relative to the PLUGIN DIRECTORY (`plugins/polars-statistics/`), not relative to `.claude-plugin/`. In the `_plugin/` layout, the plugin directory is `_plugin/` and skills are at `_plugin/skills/` — same relative structure. The `marketplace.json` `"source": "."` points to `_plugin/` as the plugin root. So `./skills` from the plugin root = `_plugin/skills/`. This is correct.

---

## Open Questions

1. **`.gitkeep` in skills directory**
   - What we know: `plugins/polars-statistics/skills/.gitkeep` exists as a git placeholder
   - What's unclear: Whether to copy it into `_plugin/skills/.gitkeep` (unnecessary in python-source tree)
   - Recommendation: Do NOT copy `.gitkeep` — it's a git artifact, not needed in the wheel

2. **`ROUTING.md` in skills directory**
   - What we know: `plugins/polars-statistics/skills/ROUTING.md` exists
   - What's unclear: Whether it needs to be bundled (it's a GSD routing artifact, not a user-facing skill)
   - Recommendation: Copy it — `plugins/**/*` includes it in the current `include` test; keeping parity is safe and it's tiny

---

## Sources

### Primary (HIGH confidence)

- Local build experiment (this session): `maturin build --release --strip` + `unzip -l` confirmed file paths in wheel — no citations needed, direct observation
- `python/polars_statistics/__init__.py` (read this session) — confirms `polars_statistics.__file__` is the correct locator for the package directory
- `plugins/polars-statistics/.claude-plugin/plugin.json` (read this session, line 12: `"skills": "./skills"`) — confirmed path semantics
- `.claude-plugin/marketplace.json` (read this session, line 10: `"source": "./plugins/polars-statistics"`) — confirmed path needing adjustment

### Secondary (MEDIUM confidence)

- [maturin Configuration Guide](https://www.maturin.rs/config) — `include`/`exclude` semantics, glob resolution
- [PR #2973 PyO3/maturin](https://github.com/pyo3/maturin/pull/2973) — confirmed include patterns resolve relative to project root (not python-source)
- maturin changelog — `WithOutDir` `to` field added in 1.12.1, limited to OUT_DIR artifacts

### Tertiary (LOW confidence)

- None.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — maturin is the existing build backend, no new dependencies
- Architecture (copy under python-source): HIGH — experimentally verified by building a test wheel this session
- `include` glob behavior: HIGH — experimentally verified (files land outside package namespace)
- `to` field limitation: HIGH — confirmed via maturin source struct analysis and changelog

**Research date:** 2026-09-22
**Valid until:** Stable for maturin 1.x series; re-verify if maturin major version changes

## Project Constraints (from CLAUDE.md)

| Directive | Impact on Phase 16 |
|-----------|-------------------|
| Tech stack: Rust 2021 + PyO3 0.26 + Polars 0.52 | No impact — packaging only |
| CI gates: clippy `-D warnings`, cargo fmt, ruff, pytest | No Rust or Python code added → no clippy/fmt/ruff impact; no pytest tests added (Phase 16 adds `scripts/verify_wheel_bundling.py`) |
| `python-source = "python"` convention (existing) | Confirmed compatible — `_plugin/` goes under `python/polars_statistics/_plugin/` |
| New API must follow `#[polars_expr]` / PyModel patterns | Not applicable — data files only |
| Distribution: Python 3.9–3.12 wheels + sdist | sdist already includes root plugin tree; wheel needs `_plugin/` copy — no compatibility concern |
| GSD Workflow Enforcement | All file writes in this phase go through `/gsd-execute-phase` |
