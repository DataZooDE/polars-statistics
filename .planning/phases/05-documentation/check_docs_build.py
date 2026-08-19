"""
check_docs_build.py — DOCS-02 gate for Phase 5 plans.

Usage:
    python check_docs_build.py [--project-root <path>]

Strategy:
  1. If mkdocs is importable, run ``mkdocs build --strict`` into a temporary
     site directory and report any broken-reference errors.
  2. Otherwise (mkdocs not available in this environment), fall back to a
     structural check: parse mkdocs.yml and assert that every markdown file
     referenced in the nav exists under the docs/ directory.

Exits 0 on success (clean build or clean structural check).
Exits 1 on any error, printing a description of the failure.

The script never installs mkdocs and never requires a maturin rebuild.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def find_project_root(start: Path) -> Path:
    """Walk up from *start* until we find mkdocs.yml."""
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "mkdocs.yml").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find mkdocs.yml above {start}. "
        "Run this script from inside the project directory."
    )


def try_mkdocs_build(project_root: Path) -> tuple[bool, str]:
    """
    Attempt a real mkdocs build with --strict.

    Returns (ok, message). The site is built into a temporary directory so it
    does not pollute the working tree.
    """
    with tempfile.TemporaryDirectory(prefix="gsd_mkdocs_") as tmp_site:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mkdocs",
                "build",
                "--strict",
                "--site-dir",
                tmp_site,
                "--config-file",
                str(project_root / "mkdocs.yml"),
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        if result.returncode == 0:
            return True, "mkdocs build --strict: PASSED (real build)"
        else:
            combined = (result.stdout + result.stderr).strip()
            return False, f"mkdocs build --strict: FAILED\n{combined}"


def mkdocs_importable() -> bool:
    """Return True if mkdocs can be imported in the current Python environment."""
    result = subprocess.run(
        [sys.executable, "-c", "import mkdocs"],
        capture_output=True,
    )
    return result.returncode == 0


def collect_nav_paths(nav_item, paths: list[str]) -> None:
    """
    Recursively walk the mkdocs nav structure and collect all markdown paths.

    The nav is a list of dicts like::

        [{"Title": "path.md"}, {"Section": [{"Sub": "sub/path.md"}, ...]}, ...]
    """
    if isinstance(nav_item, list):
        for item in nav_item:
            collect_nav_paths(item, paths)
    elif isinstance(nav_item, dict):
        for value in nav_item.values():
            if isinstance(value, str):
                # Leaf nav entry: the string is a docs-relative path
                paths.append(value)
            else:
                collect_nav_paths(value, paths)


def structural_check(project_root: Path) -> tuple[bool, str]:
    """
    Fallback: parse mkdocs.yml and verify every nav-referenced file exists.

    Returns (ok, message).
    """
    try:
        import yaml
    except ImportError:
        # PyYAML not available — try ruamel.yaml
        try:
            from ruamel.yaml import YAML

            yaml_loader = YAML()

            def load_yaml(stream):
                return yaml_loader.load(stream)

        except ImportError:
            return (
                False,
                "Neither 'yaml' (PyYAML) nor 'ruamel.yaml' is available for the "
                "structural fallback check. Install PyYAML: pip install pyyaml",
            )
    else:
        def load_yaml(stream):
            return yaml.safe_load(stream)

    config_path = project_root / "mkdocs.yml"
    with config_path.open(encoding="utf-8") as fh:
        config = load_yaml(fh)

    if not config or "nav" not in config:
        return False, "mkdocs.yml has no 'nav' section — cannot run structural check"

    nav_paths: list[str] = []
    collect_nav_paths(config["nav"], nav_paths)

    docs_dir = project_root / config.get("docs_dir", "docs")
    missing: list[str] = []
    for rel_path in nav_paths:
        full_path = docs_dir / rel_path
        if not full_path.exists():
            missing.append(rel_path)

    if missing:
        joined = "\n  ".join(missing)
        return (
            False,
            f"Structural check FAILED — nav references {len(missing)} missing file(s):\n  {joined}",
        )

    return (
        True,
        f"Structural check PASSED (fallback): all {len(nav_paths)} nav-referenced "
        f"files exist under {docs_dir.relative_to(project_root)}/",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Path to the project root containing mkdocs.yml (default: auto-detect from cwd)",
    )
    args = parser.parse_args()

    try:
        project_root = (
            args.project_root.resolve()
            if args.project_root
            else find_project_root(Path.cwd())
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Strategy 1: real mkdocs build
    if mkdocs_importable():
        ok, message = try_mkdocs_build(project_root)
    else:
        # Strategy 2: structural fallback
        print(
            "INFO: mkdocs not importable in this environment — using structural fallback check",
            file=sys.stderr,
        )
        ok, message = structural_check(project_root)

    if ok:
        print(message)
        return 0
    else:
        print(f"FAIL: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
