---
quick_id: 260819-jkj
slug: fix-v0.6.0-pypi-release
date: 2026-08-19
status: complete
commit: (pending)
---

# Summary — Fix stuck v0.6.0 PyPI release

## What was wrong

The `v0.6.0` publish run (32041071413, 2026-08-17) failed with
`400 Bad Request — File already exists (polars_statistics-0.5.0-*.whl)`.
maturin built `0.5.0`-named wheels because the version bump had only landed in
`Cargo.lock`; the source-of-truth version files still declared `0.5.0`.

## What was already fixed on origin/main (PR #37, before this task)

Investigation showed the remote had partially self-healed since the failed run:

- `c0f1bed` bumped `Cargo.toml` + `pyproject.toml` to `0.6.0` (fixes the wheel name).
- `7e3c3bb` added `skip-existing:` to the production PyPI publish step, so a
  re-run tolerates the already-published 0.5.0 files instead of hard-failing.
- CHANGELOG `[0.6.0]` already finalized (dated 2026-08-17, compare link fixed).

## The remaining gap this task closed

PR #37 missed `python/polars_statistics/__init__.py`: `__version__` was still
`"0.5.0"`, so the installed package would report the wrong version at runtime
(`polars_statistics.__version__`) even though the wheel filename is correct.

**Change:** bump `__version__` `0.5.0 → 0.6.0`. Local commit rebuilt cleanly on
top of `origin/main` (the earlier duplicate local commit was discarded).

## Verification

- All four version sources now agree on `0.6.0`
  (`Cargo.toml`, `pyproject.toml`, `Cargo.lock`, `__init__.py`).

## Remaining (outward-facing — release trigger)

The publish still needs to be RE-TRIGGERED — it has not run since the fix, and
the `v0.6.0` git tag still points at the pre-fix commit (`ea9c60b`, 0.5.0 code):

1. `git push origin main` (this `__version__` fix).
2. Trigger the publish, either:
   - Re-point tag + recreate the GitHub v0.6.0 release
     (`git tag -f v0.6.0 <sha> && git push -f origin v0.6.0`, then republish the
     release) → fires `release: published`; or
   - `gh workflow run publish.yml -f publish_to=pypi` on `main`.
   With `skip-existing` in place, the run uploads the 0.6.0 artifacts and skips
   the pre-existing 0.5.0 files.

Production PyPI upload is irreversible, so the trigger is gated on explicit
user confirmation.
