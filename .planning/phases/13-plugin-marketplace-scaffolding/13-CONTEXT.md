# Phase 13: Plugin & Marketplace Scaffolding - Context

**Gathered:** 2026-09-20
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure phase — discuss skipped)

<domain>
## Phase Boundary

Deliver a valid, discoverable Claude Code plugin shell for polars-statistics: a
root `.claude-plugin/marketplace.json` listing the plugin, and
`plugins/polars-statistics/.claude-plugin/plugin.json` declaring the plugin
identity — mirroring the anofox-forecast layout. Scope is the empty, installable
shell only; the five skill cheat-sheets are authored in Phase 14 and bundled into
the wheel in Phase 16.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure phase.
The layout is fixed by the milestone decision (mirror anofox-forecast): root
`.claude-plugin/marketplace.json` + `plugins/polars-statistics/` with `plugin.json`
and a `skills/` tree. Use the anofox-forecast repo as the authoritative format
reference. Version in `plugin.json` should track the package version line (0.8.0).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- anofox-forecast repository (sibling `../anofox-forecast`) provides the
  authoritative marketplace.json / plugin.json format to mirror.
- Existing package metadata (name, description, homepage, repository, license)
  lives in `pyproject.toml` and `Cargo.toml` — reuse for plugin.json fields.

### Established Patterns
- Milestone decision locks the layout: `.claude-plugin/marketplace.json` at root;
  `plugins/polars-statistics/.claude-plugin/plugin.json` + `skills/`.
- `user-invocable: false` for auto-surfaced skills (relevant to Phase 14).

### Integration Points
- The `skills/` directory created here is populated in Phase 14 and bundled into
  the wheel via maturin package data in Phase 16.

</code_context>

<specifics>
## Specific Ideas

Mirror anofox-forecast exactly for the marketplace + plugin manifest format.
plugin.json must declare: name, description, version, author, homepage,
repository, license. marketplace.json must list the plugin with name, source
path, description, keywords.

</specifics>

<deferred>
## Deferred Ideas

None — infrastructure phase.

</deferred>
