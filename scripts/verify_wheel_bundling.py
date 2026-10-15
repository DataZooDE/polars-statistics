#!/usr/bin/env python
"""
Verification harness for PKG-01 — the Claude Code plugin bundled into the wheel.

Two independent halves:

  Half A (drift guard, no wheel build needed):
    Every bundled skill under python/polars_statistics/_plugin/skills/ is
    byte-identical to its root plugins/polars-statistics/skills/ counterpart,
    exactly five bundled SKILL.md files exist, and _plugin/marketplace.json
    parses as JSON with its first plugin's "source" equal to ".".

  Half B (behavioral, PKG-01 proper):
    Locate the newest built wheel, install it into a fresh venv, and assert
    that marketplace.json, .claude-plugin/plugin.json, and all five SKILL.md
    files exist on disk under the installed polars_statistics/_plugin/.

Usage:
    uv run python scripts/verify_wheel_bundling.py                # both halves
    uv run python scripts/verify_wheel_bundling.py --drift-only    # Half A only

Exit codes:
    0  All requested checks passed
    1  One or more checks failed
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SKILL_NAMES = [
    "polars-statistics-anova",
    "polars-statistics-getting-started",
    "polars-statistics-hypothesis-tests",
    "polars-statistics-regression",
    "polars-statistics-results",
]
ROOT_SKILLS_DIR = "plugins/polars-statistics/skills"
BUNDLED_SKILLS_DIR = "python/polars_statistics/_plugin/skills"
BUNDLED_MARKETPLACE = "python/polars_statistics/_plugin/marketplace.json"
EXPECTED_MARKETPLACE_SOURCE = "."
REQUIRED_INSTALLED_FILES = [
    "_plugin/marketplace.json",
    "_plugin/.claude-plugin/plugin.json",
    "_plugin/skills/polars-statistics-anova/SKILL.md",
    "_plugin/skills/polars-statistics-getting-started/SKILL.md",
    "_plugin/skills/polars-statistics-hypothesis-tests/SKILL.md",
    "_plugin/skills/polars-statistics-regression/SKILL.md",
    "_plugin/skills/polars-statistics-results/SKILL.md",
]


def repo_root() -> str:
    """Resolve the repo root from this script's location."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Half A — drift guard
# ---------------------------------------------------------------------------

def _bytes_equal(path_a: str, path_b: str) -> bool:
    with open(path_a, "rb") as fh_a, open(path_b, "rb") as fh_b:
        return fh_a.read() == fh_b.read()


def run_drift_check(root: str) -> list[str]:
    """Return a list of failure messages; empty list means all checks passed."""
    failures: list[str] = []

    for name in SKILL_NAMES:
        root_path = os.path.join(root, ROOT_SKILLS_DIR, name, "SKILL.md")
        bundled_path = os.path.join(root, BUNDLED_SKILLS_DIR, name, "SKILL.md")
        if not os.path.isfile(root_path):
            failures.append(f"missing root skill file: {root_path}")
            continue
        if not os.path.isfile(bundled_path):
            failures.append(f"missing bundled skill file: {bundled_path}")
            continue
        if not _bytes_equal(root_path, bundled_path):
            failures.append(f"drift: {bundled_path} != {root_path}")

    root_routing = os.path.join(root, ROOT_SKILLS_DIR, "ROUTING.md")
    bundled_routing = os.path.join(root, BUNDLED_SKILLS_DIR, "ROUTING.md")
    if not os.path.isfile(root_routing):
        failures.append(f"missing root ROUTING.md: {root_routing}")
    elif not os.path.isfile(bundled_routing):
        failures.append(f"missing bundled ROUTING.md: {bundled_routing}")
    elif not _bytes_equal(root_routing, bundled_routing):
        failures.append(f"drift: {bundled_routing} != {root_routing}")

    bundled_glob = os.path.join(root, BUNDLED_SKILLS_DIR, "*", "SKILL.md")
    bundled_count = len(glob.glob(bundled_glob))
    if bundled_count != 5:
        failures.append(f"expected exactly 5 bundled SKILL.md, found {bundled_count}")

    marketplace_path = os.path.join(root, BUNDLED_MARKETPLACE)
    if not os.path.isfile(marketplace_path):
        failures.append(f"missing bundled marketplace.json: {marketplace_path}")
    else:
        try:
            with open(marketplace_path, encoding="utf-8") as fh:
                data = json.load(fh)
            source = data["plugins"][0]["source"]
            if source != EXPECTED_MARKETPLACE_SOURCE:
                failures.append(
                    f"marketplace.json plugins[0].source == {source!r}, "
                    f"expected {EXPECTED_MARKETPLACE_SOURCE!r}"
                )
        except (OSError, ValueError, KeyError, IndexError) as exc:
            failures.append(f"marketplace.json failed to parse/validate: {exc}")

    return failures


# ---------------------------------------------------------------------------
# Half B — behavioral clean-venv install check
# ---------------------------------------------------------------------------

def find_newest_wheel(root: str) -> str | None:
    pattern = os.path.join(root, "target", "wheels", "polars_statistics-*.whl")
    wheels = glob.glob(pattern)
    if not wheels:
        return None
    return max(wheels, key=os.path.getmtime)


def run_behavioral_check(root: str) -> list[str]:
    failures: list[str] = []

    wheel = find_newest_wheel(root)
    if wheel is None:
        failures.append(
            "no built wheel found under target/wheels/polars_statistics-*.whl "
            "— run `uv run maturin build --release --strip` first"
        )
        return failures

    tmp_dir = tempfile.mkdtemp(prefix="verify-wheel-bundling-")
    venv_dir = os.path.join(tmp_dir, "venv")
    try:
        venv.create(venv_dir, with_pip=True)
        venv_python = os.path.join(venv_dir, "bin", "python")
        if not os.path.isfile(venv_python):
            # Windows layout fallback
            venv_python = os.path.join(venv_dir, "Scripts", "python.exe")

        install = subprocess.run(
            [venv_python, "-m", "pip", "install", "--quiet", wheel],
            capture_output=True,
            text=True,
        )
        if install.returncode != 0:
            failures.append(
                f"pip install of {wheel} failed:\n{install.stdout}\n{install.stderr}"
            )
            return failures

        probe = (
            "import pathlib, polars_statistics, json\n"
            "pkg = pathlib.Path(polars_statistics.__file__).parent\n"
            f"required = {REQUIRED_INSTALLED_FILES!r}\n"
            "missing = [p for p in required if not (pkg / p).exists()]\n"
            "print(json.dumps({'pkg': str(pkg), 'missing': missing}))\n"
        )
        result = subprocess.run(
            [venv_python, "-c", probe],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            failures.append(f"install probe failed:\n{result.stdout}\n{result.stderr}")
            return failures

        payload = json.loads(result.stdout.strip().splitlines()[-1])
        if payload["missing"]:
            failures.append(
                f"missing after install under {payload['pkg']}: {payload['missing']}"
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the Claude Code plugin is bundled into the built wheel (PKG-01)."
    )
    parser.add_argument(
        "--drift-only",
        action="store_true",
        help="Run only Half A (drift guard) — no wheel build or venv install required.",
    )
    args = parser.parse_args()

    root = repo_root()
    all_failures: list[str] = []

    print("=== Half A: drift guard ===")
    drift_failures = run_drift_check(root)
    if drift_failures:
        for msg in drift_failures:
            print(f"  FAIL  {msg}")
    else:
        print("  PASS  all bundled files byte-identical to root; marketplace.json valid")
    all_failures.extend(drift_failures)

    if not args.drift_only:
        print("=== Half B: behavioral clean-venv install check ===")
        behavioral_failures = run_behavioral_check(root)
        if behavioral_failures:
            for msg in behavioral_failures:
                print(f"  FAIL  {msg}")
        else:
            print("  PASS  clean-venv install places all plugin files on disk")
        all_failures.extend(behavioral_failures)

    print()
    if all_failures:
        print(f"RESULT: FAIL — {len(all_failures)} check(s) failed")
        return 1

    print("RESULT: PASS — all requested checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
