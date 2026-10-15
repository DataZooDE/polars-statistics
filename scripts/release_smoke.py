"""Reusable post-install smoke check for an installed polars-statistics wheel.

Run this with the interpreter of the venv under test, AFTER installing a
polars-statistics wheel into it:

    python scripts/release_smoke.py --expect 0.8.0

Checks (each prints one PASS/FAIL line):
  (a) polars_statistics.__version__ equals --expect.
  (b) The imported package directory is not inside this repo's python/
      directory (proves the installed artifact is under test, not the
      source tree).
  (c) The package's _plugin/ directory carries marketplace.json,
      .claude-plugin/plugin.json, skills/ROUTING.md, and exactly five
      skills/*/SKILL.md files.
  (d) The bundled plugin.json / marketplace.json / every SKILL.md
      frontmatter version all equal --expect.
  (e) The expression API (ps.ols, ps.ttest_ind) runs and returns sane
      results.
  (f) The PyModel API (OLS) runs and returns sane results.

Exit 0 and a final `SMOKE_OK {version} {package dir}` line on success.
Exit 1 and a final `SMOKE_FAIL {n} check(s)` line otherwise.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys


def _fail(failures: list[str], msg: str) -> None:
    failures.append(msg)
    print(f"  FAIL  {msg}")


def _pass(msg: str) -> None:
    print(f"  PASS  {msg}")


def _frontmatter_version(path: str) -> str | None:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    parts = text.split("---")
    if len(parts) < 3:
        return None
    frontmatter = parts[1]
    for line in frontmatter.splitlines():
        line = line.strip()
        if line.startswith("version:"):
            value = line.split(":", 1)[1].strip()
            return value.strip("'\"")
    return None


def check_version(expect: str, failures: list[str]) -> str | None:
    import polars_statistics

    version = polars_statistics.__version__
    if version == expect:
        _pass(f"polars_statistics.__version__ == {expect!r}")
    else:
        _fail(failures, f"polars_statistics.__version__ == {version!r}, expected {expect!r}")
    return version


def check_not_source_tree(failures: list[str]) -> str:
    import polars_statistics

    pkg_dir = os.path.dirname(os.path.abspath(polars_statistics.__file__))
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_python_dir = os.path.join(repo_root, "python")
    if os.path.commonpath([pkg_dir, repo_python_dir]) == repo_python_dir:
        _fail(
            failures,
            f"imported package dir {pkg_dir!r} is inside this repo's python/ directory "
            "— the installed artifact is not being tested",
        )
    else:
        _pass(f"imported package dir {pkg_dir!r} is outside the repo's python/ directory")
    return pkg_dir


def check_plugin_files(pkg_dir: str, failures: list[str]) -> None:
    plugin_dir = os.path.join(pkg_dir, "_plugin")
    required = [
        os.path.join(plugin_dir, "marketplace.json"),
        os.path.join(plugin_dir, ".claude-plugin", "plugin.json"),
        os.path.join(plugin_dir, "skills", "ROUTING.md"),
    ]
    missing = [p for p in required if not os.path.isfile(p)]
    if missing:
        _fail(failures, f"missing bundled plugin files: {missing}")
    else:
        _pass("marketplace.json, .claude-plugin/plugin.json, skills/ROUTING.md present")

    skill_md = sorted(glob.glob(os.path.join(plugin_dir, "skills", "*", "SKILL.md")))
    if len(skill_md) == 5:
        _pass("exactly five bundled skills/*/SKILL.md found")
    else:
        _fail(failures, f"expected exactly 5 bundled SKILL.md, found {len(skill_md)}")


def check_versions_in_bundle(pkg_dir: str, expect: str, failures: list[str]) -> None:
    import json

    plugin_dir = os.path.join(pkg_dir, "_plugin")

    plugin_json_path = os.path.join(plugin_dir, ".claude-plugin", "plugin.json")
    if os.path.isfile(plugin_json_path):
        with open(plugin_json_path, encoding="utf-8") as fh:
            plugin_data = json.load(fh)
        version = plugin_data.get("version")
        if version == expect:
            _pass(f"bundled plugin.json version == {expect!r}")
        else:
            _fail(failures, f"bundled plugin.json version == {version!r}, expected {expect!r}")

    marketplace_path = os.path.join(plugin_dir, "marketplace.json")
    if os.path.isfile(marketplace_path):
        with open(marketplace_path, encoding="utf-8") as fh:
            marketplace_data = json.load(fh)
        version = marketplace_data["plugins"][0].get("version")
        source = marketplace_data["plugins"][0].get("source")
        if version == expect:
            _pass(f"bundled marketplace.json plugins[0].version == {expect!r}")
        else:
            _fail(
                failures,
                f"bundled marketplace.json plugins[0].version == {version!r}, expected {expect!r}",
            )
        if source == ".":
            _pass("bundled marketplace.json plugins[0].source == '.'")
        else:
            _fail(failures, f"bundled marketplace.json plugins[0].source == {source!r}, expected '.'")

    for skill_path in sorted(glob.glob(os.path.join(plugin_dir, "skills", "*", "SKILL.md"))):
        version = _frontmatter_version(skill_path)
        if version == expect:
            _pass(f"{os.path.relpath(skill_path, pkg_dir)} frontmatter version == {expect!r}")
        else:
            _fail(
                failures,
                f"{os.path.relpath(skill_path, pkg_dir)} frontmatter version == {version!r}, "
                f"expected {expect!r}",
            )


def check_expression_api(failures: list[str]) -> None:
    import numpy as np
    import polars as pl
    import polars_statistics as ps

    rng = np.random.default_rng(0)
    n_half = 30
    x = rng.normal(size=n_half * 2)
    y = 2 + 3 * x + rng.normal(scale=0.1, size=n_half * 2)
    a = rng.normal(loc=0.0, size=n_half * 2)
    b = rng.normal(loc=1.0, size=n_half * 2)
    g = ["a"] * n_half + ["b"] * n_half

    df = pl.DataFrame({"g": g, "x": x, "y": y, "a": a, "b": b})

    fit_df = df.group_by("g").agg(ps.ols("y", "x").alias("fit"))
    if fit_df.height == 2:
        _pass("ps.ols group_by returns 2 rows")
    else:
        _fail(failures, f"ps.ols group_by returned {fit_df.height} rows, expected 2")

    ok = True
    for row in fit_df.iter_rows(named=True):
        fit = row["fit"]
        slope = fit["coefficients"][0]
        r_squared = fit["r_squared"]
        if abs(slope - 3.0) > 0.1:
            ok = False
            _fail(failures, f"ps.ols coefficients[0] == {slope!r}, expected within 0.1 of 3.0")
        if not (r_squared > 0.9):
            ok = False
            _fail(failures, f"ps.ols r_squared == {r_squared!r}, expected > 0.9")
    if ok:
        _pass("ps.ols coefficients[0] ~= 3.0 and r_squared > 0.9 for both groups")

    t_df = df.select(ps.ttest_ind("a", "b").alias("t"))
    p_value = t_df["t"][0]["p_value"]
    if 0.0 <= p_value <= 1.0:
        _pass(f"ps.ttest_ind p_value == {p_value!r} in [0, 1]")
    else:
        _fail(failures, f"ps.ttest_ind p_value == {p_value!r}, expected in [0, 1]")


def check_pymodel_api(failures: list[str]) -> None:
    import numpy as np
    from polars_statistics import OLS

    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    y = 2 + 3 * x + rng.normal(scale=0.1, size=60)

    model = OLS().fit(x.reshape(-1, 1), y)
    slope = model.coefficients[0]
    intercept = model.intercept

    if abs(slope - 3.0) > 0.1:
        _fail(failures, f"OLS().coefficients[0] == {slope!r}, expected within 0.1 of 3.0")
    else:
        _pass(f"OLS().coefficients[0] == {slope!r}, within 0.1 of 3.0")

    if intercept is None or abs(intercept - 2.0) > 0.1:
        _fail(failures, f"OLS().intercept == {intercept!r}, expected within 0.1 of 2.0")
    else:
        _pass(f"OLS().intercept == {intercept!r}, within 0.1 of 2.0")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect", required=True, help="Expected polars-statistics version")
    args = parser.parse_args(argv)

    failures: list[str] = []

    print("=== release_smoke: version ===")
    check_version(args.expect, failures)

    print("=== release_smoke: install location ===")
    pkg_dir = check_not_source_tree(failures)

    print("=== release_smoke: bundled plugin files ===")
    check_plugin_files(pkg_dir, failures)

    print("=== release_smoke: bundled plugin versions ===")
    check_versions_in_bundle(pkg_dir, args.expect, failures)

    print("=== release_smoke: expression API ===")
    check_expression_api(failures)

    print("=== release_smoke: PyModel API ===")
    check_pymodel_api(failures)

    print()
    if failures:
        print(f"SMOKE_FAIL {len(failures)} check(s)")
        return 1

    print(f"SMOKE_OK {args.expect} {pkg_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
