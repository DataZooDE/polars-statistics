# Phase 7: Release 0.6.0 — Handoff

**Status:** REL-01 + REL-02 done autonomously; REL-03 + REL-04 require the maintainer (irreversible production publish).

## Done (reversible, committed)
- **REL-01:** version bumped 0.5.0 → 0.6.0 in `Cargo.toml`, `pyproject.toml`, `Cargo.lock`. Build compiles clean.
- **REL-02:** `CHANGELOG.md` `## [0.6.0] - 2026-08-17` finalized (full new API, crate bumps, breaking ICC contract, known limitations) + compare link `v0.5.0...v0.6.0`. Release notes = this CHANGELOG section.

## Remaining — maintainer action (outward-facing / irreversible)
The autonomous agent intentionally did NOT push or publish. To ship 0.6.0:

1. Push the version-bump commit and create the release tag (triggers `.github/workflows/publish.yml` OIDC → production PyPI):
   ```
   git push origin main
   git tag -a v0.6.0 -m "polars-statistics 0.6.0 — full anofox-* API parity"
   git push origin v0.6.0
   ```
2. Watch the GitHub Actions `publish.yml` run build the sdist + platform wheels and publish to PyPI via trusted publishing.
3. **REL-04 post-release smoke check** (in a clean venv):
   ```
   pip install polars-statistics==0.6.0
   python -c "import polars_statistics as ps; print(ps.__version__); [getattr(ps,n) for n in ['one_way_anova','icc','Gamma','GLMM','TheilSen','MomentAccumulator']]; print('0.6.0 import OK')"
   ```

## Pre-publish state (verified this session)
- clippy `-D warnings` CLEAN · cargo fmt CLEAN · `cargo test rust_api` 17 passed · `pytest` 602 passed / 5 skipped / 0 failed.
- Full OS×Python wheel matrix is exercised by the same GitHub Actions CI on push.

## Known limitations shipped in 0.6.0 (documented)
- `WLS.hc_inference` raises `NotImplementedError` (weighted HC needs crate support).
- `GLMM.factors()` returns `[]` after `.fit()` (follow-up).
