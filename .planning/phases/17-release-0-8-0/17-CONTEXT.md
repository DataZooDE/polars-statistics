# Phase 17: Release 0.8.0 - Context

**Gathered:** 2026-10-15
**Status:** Ready for planning
**Mode:** Orchestrator-recorded from the user's decisions and repository precedent (Phase 12 / v0.7.0)

<domain>
## Phase Boundary

Ship polars-statistics 0.8.0 to production PyPI via the existing OIDC pipeline (`.github/workflows/publish.yml`, triggered by a published GitHub release), with all version sources aligned and the bundled skills present in the installed wheel (REL-07, REL-08).

</domain>

<decisions>
## Implementation Decisions

### Scope (user decisions)
- v0.8.0 ships everything from 13–16 plus 16.1–16.4: the plugin and skills, wheel bundling, upstream crate refresh (frozen at `anofox-statistics =0.4.7` / `anofox-regression =0.5.22`), penalized GLM, AFT, post-hoc, augment, ModelInfo, GLMM families, EB shrinkage and the ruff gate.
- Newer upstream (0.4.8+ / 0.5.23+) is deferred to v0.9.0 (UPST-11). Do not bump the anofox crates.

### Version alignment (REL-07; the v0.6.0 misaligned-version failure must not recur)
Bump 0.7.0 → 0.8.0 in `Cargo.toml` `[package] version`, `pyproject.toml`, `python/polars_statistics/__init__.py` `__version__`, and every SKILL.md version field. Check `plugins/polars-statistics/.claude-plugin/plugin.json` and `.claude-plugin/marketplace.json`; they may already say 0.8.0. Re-sync `python/polars_statistics/_plugin/` from `plugins/` (the CI drift guard `scripts/verify_wheel_bundling.py --drift-only` must pass). Add a single automated check that every version source reads 0.8.0. CHANGELOG: rename `[Unreleased]` to `[0.8.0] - <release date>` and add a fresh empty `[Unreleased]` above it. Keep the release-note item on Python 3.9 `get_type_hints` from Phase 16.4 if present.

### Release flow (follows the v0.7.0 precedent: PRs to main, then a tag and GitHub release)
Local `main` is ~200 commits ahead of `origin/main` and has never been pushed during this milestone. GitHub `main` receives changes through PRs.
1. **Pre-push local gate**: fmt, clippy -D warnings, `ruff check .`, full pytest, rust_api tests, stub drift, skill harness, wheel-bundling drift, `mkdocs build --strict`, and a release wheel build + clean-venv install smoke (import, `__version__ == "0.8.0"`, `_plugin` files present).
2. **CHECKPOINT (human-action / decision): push.** Pushing publishes the work to GitHub. Create a release branch (e.g. `release/v0.8.0`) from local main, push it, and open a PR to main with `gh pr create` (body ending with the Claude Code attribution line). Requires the user's explicit approval; the executor must stop and ask.
3. Wait for CI on the PR. All jobs must be green on the full matrix (Linux/macOS/Windows, Py 3.9–3.12). If CI fails, diagnose and fix on the branch (normal commits), then re-run.
4. **Open PR #47 (`ci/cost-guardrails`)** touches `.github/workflows/ci.yml`, which this milestone also changed (ruff step, rust_api step, wheel-bundling drift step). Surface the expected conflict to the user; do not merge or modify #47 without the user's approval.
5. **CHECKPOINT (human-action): merge.** Merging the PR to main is the user's action, or the executor's only with explicit approval.
6. **Optional TestPyPI dry run (recommended)**: `gh workflow run publish.yml -f publish_to=testpypi` on the merged main, then install from TestPyPI in a clean venv and smoke-check it. Requires approval because it publishes to TestPyPI.
7. **CHECKPOINT (human-action): production publish.** Tag `v0.8.0` on the merged main commit and create the GitHub release (`gh release create v0.8.0` with notes from the CHANGELOG `[0.8.0]` section). This fires `publish.yml` to production PyPI and CANNOT be undone. Requires the user's explicit approval right before the action, even if earlier steps were approved.
8. **Post-release smoke (REL-08)**: after the workflow succeeds, `pip install polars-statistics==0.8.0` in a clean venv; import, version check, a quick expression + PyModel call, and the bundled skill files present under `polars_statistics/_plugin/`.

### Other
- Stale local worktree `.claude/worktrees/agent-a88ea0e6ef2dbcd4c` (an old phase 16 attempt, uncommitted): do not push or merge it. Mention it to the user for cleanup.
- `workflow.use_worktrees` is false (sequential executors on main). Leave the config as-is unless the user asks.
- Never force-push. Never push directly to `main`.

</decisions>

<deferred>
## Deferred Ideas

- v0.9.0: upstream wave 3 (UPST-11).

</deferred>
