---
phase: 12
slug: release
status: complete
requirements: REL-05 (complete), REL-06 (complete)
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

## REL-06 — publish (complete)

Tagged `v0.7.0` at `9a93c1a` and created the GitHub release, which fired
`publish.yml` (Trusted Publishing via OIDC). Publish run `35508433341` succeeded:
sdist + all 5 platform wheels built and **published to production PyPI**.
PyPI latest is now `0.7.0`.

Post-publish smoke check (clean venv, installed from PyPI) — all green:

- `polars_statistics.__version__ == "0.7.0"`
- `py.typed` + `_polars_statistics.pyi` shipped in the wheel (Phase 8)
- `repr(OLS(...))` informative; `.to_dict()` works; `unnest`/`struct_to_dict`
  present (Phase 9)
- `.score()` returns R² (Phase 10)
- `with_intercept` emits `FutureWarning` (Phase 10)
