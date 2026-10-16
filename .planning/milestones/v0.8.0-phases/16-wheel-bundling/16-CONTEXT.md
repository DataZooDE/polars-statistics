# Phase 16: Wheel Bundling - Context

**Gathered:** 2026-09-21
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure phase — discuss skipped)

<domain>
## Phase Boundary

Make the Claude Code plugin + all five skill cheat-sheets travel inside the built
Python wheel so that after `pip install polars-statistics`, the marketplace,
`plugin.json`, and every `SKILL.md` are present on disk in the installed package.
Scope is maturin packaging configuration + a build/install verification. Out of
scope: authoring skills (done, Phases 13-15), the release itself (Phase 17).

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
Pure infrastructure/packaging phase — mechanism is at Claude's discretion, to be
resolved during research/planning. Key open technical questions the plan must answer:
- The plugin currently lives at repo ROOT: `.claude-plugin/marketplace.json` and
  `plugins/polars-statistics/` (with `.claude-plugin/plugin.json` and `skills/`). The
  Python package is under `python/polars_statistics/` (maturin `python-source = "python"`).
- Determine the correct maturin mechanism to bundle the root-level plugin tree into the
  wheel (`[tool.maturin] include` / `force-include`, target path inside the wheel), and
  WHERE the files should land in the installed package so they are discoverable on disk.
  Mirror the pyfda/fdars approach where applicable (they use
  `[tool.maturin] include = ["python/fdars/data/*.csv"]` for data under the python tree).
- Decide whether the canonical location should be the repo-root plugin (current) mirrored
  into the wheel, OR the plugin/skills relocated/copied under `python/polars_statistics/`
  so maturin picks them up automatically. Prefer the least-disruptive option that keeps
  the repo-root plugin working for local `/plugin marketplace add` AND lands the files in
  the wheel.
- Must not break the existing wheel build or CI gates (clippy, fmt, ruff, pytest).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Current `[tool.maturin]` in pyproject.toml: `python-source = "python"`,
  `module-name`, `bindings = "pyo3"`, `features`, `strip = true` — no `include` yet.
- pyfda reference: `[tool.maturin] include = ["python/fdars/data/*.csv"]` (globs relative
  to manifest dir; data under the python-source tree).
- The plugin tree: `.claude-plugin/marketplace.json` (root),
  `plugins/polars-statistics/.claude-plugin/plugin.json`,
  `plugins/polars-statistics/skills/*/SKILL.md` (5 skills), `skills/.gitkeep`,
  `plugins/polars-statistics/skills/ROUTING.md`.
- `scripts/verify_skill_examples.py` (Phase 15 harness — not necessarily bundled).

### Established Patterns
- maturin build via `uv run maturin` (maturin 1.13.3 available). CI builds wheels for 5
  platform combos + sdist.

### Integration Points
- Phase 17 releases this wheel to PyPI. The bundling config must be present and verified
  before release.

</code_context>

<specifics>
## Specific Ideas

- Verification MUST be behavioral (criterion #2): build the wheel, install it into a
  clean/temp environment, and assert marketplace.json + plugin.json + all five SKILL.md
  are present on disk in the installed location. A config-only check is insufficient.
- Keep the repo-root plugin working for local marketplace add (don't break local dev).

</specifics>

<deferred>
## Deferred Ideas

- None — infrastructure phase.

</deferred>
