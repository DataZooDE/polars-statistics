---
quick_id: 260819-jkj
slug: fix-v0.6.0-pypi-release
date: 2026-08-19
---

# Fix stuck v0.6.0 PyPI release

## Problem

The `v0.6.0` publish workflow (run 32041071413, 2026-08-17) failed with
`400 File already exists` because the built wheels were named
`polars_statistics-0.5.0-*.whl`. Root cause: the version bump to 0.6.0 was only
applied to `Cargo.lock` — the three source-of-truth version files still declare
`0.5.0`, so maturin built 0.5.0-named artifacts that PyPI rejected as duplicates.

PyPI latest published = 0.5.0. Nothing 0.6.0 exists on PyPI, so the version
number is still free to (re)use.

## Fix

1. Bump version `0.5.0 → 0.6.0` in:
   - `pyproject.toml` `[project].version` (primary source maturin uses for wheel name)
   - `Cargo.toml` `[package].version`
   - `python/polars_statistics/__init__.py` `__version__`
2. Verify `Cargo.lock` already lists the package at 0.6.0 (consistent after bump).
3. Confirm no remaining `0.5.0` self-version references linger.
4. Commit atomically.
5. Re-trigger the PyPI publish (re-point `v0.6.0` tag + recreate GitHub release,
   or `workflow_dispatch → pypi`). Irreversible prod publish — confirm with user
   before firing.

## Verification

- `grep -rn '0\.6\.0'` across the three files returns the bumped version.
- `cargo metadata`/lockfile stays consistent (no downgrade back to 0.5.0).
- Wheel built by CI is named `polars_statistics-0.6.0-*`.
