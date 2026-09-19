---
phase: 12
slug: release
status: in-progress
requirements: REL-05 (complete), REL-06 (pending publish confirmation)
---

# Phase 12 Summary — Release 0.7.0

## REL-05 — version bump (complete)

Bumped `0.6.0 → 0.7.0` across all four version sources, verified aligned:

- `Cargo.toml` → 0.7.0
- `pyproject.toml` → 0.7.0
- `python/polars_statistics/__init__.py` `__version__` → 0.7.0
- `Cargo.lock` → 0.7.0 (synced via `cargo metadata`)

Finalized CHANGELOG `[0.7.0] - 2026-09-20` summarizing the milestone (type stubs,
result ergonomics, contextual errors, API consistency, docs) and added the
`v0.6.0...v0.7.0` compare link. No test hardcodes the version.

## REL-06 — publish (pending explicit confirmation)

Once the bump PR merges to `main`, the release is triggered by tagging `v0.7.0`
at the merged commit and creating the GitHub release, which fires `publish.yml`
(Trusted Publishing via OIDC). **This step is gated on explicit user
confirmation** because production PyPI upload is irreversible (0.7.0 filenames
can never be reused).

Post-publish smoke check: confirm the workflow run succeeds, PyPI latest = 0.7.0,
and `pip install polars-statistics==0.7.0` imports cleanly with
`__version__ == "0.7.0"`.
