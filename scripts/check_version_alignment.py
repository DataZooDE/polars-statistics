"""Single automated check that every version source in the repo agrees (REL-07).

The v0.6.0 release got stuck because the version bump reached only Cargo.lock;
maturin then built a wheel named with the stale version. This script collects
the version string from every independent source of truth in the repo and
fails if any of them disagree with each other (or with an optional --expect).

Sources (18 at the 0.8.0 release; skills are discovered by glob, so a sixth
skill directory is covered automatically):

  - Cargo.toml                                          [package] version
  - pyproject.toml                                      [project] version
  - python/polars_statistics/__init__.py                __version__
  - uv.lock                                              [[package]] "polars-statistics" version
  - plugins/polars-statistics/.claude-plugin/plugin.json             version
  - .claude-plugin/marketplace.json                      plugins[name=polars-statistics].version
  - python/polars_statistics/_plugin/.claude-plugin/plugin.json      version
  - python/polars_statistics/_plugin/marketplace.json     plugins[name=polars-statistics].version
  - plugins/polars-statistics/skills/*/SKILL.md           frontmatter version
  - python/polars_statistics/_plugin/skills/*/SKILL.md    frontmatter version

Public API:
  source_paths(root) -> list[Path]
  collect_versions(root) -> (dict[label, version], list[error message])
  check(root, expect=None) -> list[failure message]  (empty == aligned)
  main(argv=None) -> int

CLI:
  python scripts/check_version_alignment.py [--expect X.Y.Z]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PLUGIN_NAME = "polars-statistics"

CARGO_TOML = "Cargo.toml"
PYPROJECT_TOML = "pyproject.toml"
INIT_PY = "python/polars_statistics/__init__.py"
UV_LOCK = "uv.lock"
ROOT_PLUGIN_JSON = "plugins/polars-statistics/.claude-plugin/plugin.json"
ROOT_MARKETPLACE_JSON = ".claude-plugin/marketplace.json"
BUNDLED_PLUGIN_JSON = "python/polars_statistics/_plugin/.claude-plugin/plugin.json"
BUNDLED_MARKETPLACE_JSON = "python/polars_statistics/_plugin/marketplace.json"
ROOT_SKILLS_GLOB = "plugins/polars-statistics/skills/*/SKILL.md"
BUNDLED_SKILLS_GLOB = "python/polars_statistics/_plugin/skills/*/SKILL.md"


def _default_root() -> Path:
    return Path(__file__).resolve().parent.parent


def source_paths(root: Path) -> list[Path]:
    """Return every file this check reads from, relative to root."""
    root = Path(root)
    paths = [
        root / CARGO_TOML,
        root / PYPROJECT_TOML,
        root / INIT_PY,
        root / UV_LOCK,
        root / ROOT_PLUGIN_JSON,
        root / ROOT_MARKETPLACE_JSON,
        root / BUNDLED_PLUGIN_JSON,
        root / BUNDLED_MARKETPLACE_JSON,
    ]
    paths.extend(sorted(root.glob(ROOT_SKILLS_GLOB)))
    paths.extend(sorted(root.glob(BUNDLED_SKILLS_GLOB)))
    return paths


def _read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _toml_table_version(text: str, table_header: str) -> str | None:
    """Find `version = "X.Y.Z"` inside the given [table] header, before the next [ header."""
    lines = text.splitlines()
    in_table = False
    version_re = re.compile(r'^\s*version\s*=\s*"([^"]+)"\s*$')
    for line in lines:
        stripped = line.strip()
        if stripped == table_header:
            in_table = True
            continue
        if in_table and stripped.startswith("["):
            break
        if in_table:
            match = version_re.match(line)
            if match:
                return match.group(1)
    return None


def _cargo_version(root: Path, errors: list[str]) -> str | None:
    path = root / CARGO_TOML
    text = _read_text(path)
    if text is None:
        errors.append(f"missing source: {CARGO_TOML}")
        return None
    version = _toml_table_version(text, "[package]")
    if version is None:
        errors.append(f"{CARGO_TOML}: could not find [package] version")
    return version


def _pyproject_version(root: Path, errors: list[str]) -> str | None:
    path = root / PYPROJECT_TOML
    text = _read_text(path)
    if text is None:
        errors.append(f"missing source: {PYPROJECT_TOML}")
        return None
    version = _toml_table_version(text, "[project]")
    if version is None:
        errors.append(f"{PYPROJECT_TOML}: could not find [project] version")
    return version


def _init_py_version(root: Path, errors: list[str]) -> str | None:
    path = root / INIT_PY
    text = _read_text(path)
    if text is None:
        errors.append(f"missing source: {INIT_PY}")
        return None
    match = re.search(r'^__version__\s*=\s*"([^"]+)"\s*$', text, re.MULTILINE)
    if match is None:
        errors.append(f"{INIT_PY}: could not find __version__")
        return None
    return match.group(1)


def _uv_lock_version(root: Path, errors: list[str]) -> str | None:
    path = root / UV_LOCK
    text = _read_text(path)
    if text is None:
        errors.append(f"missing source: {UV_LOCK}")
        return None
    lines = text.splitlines()
    name_re = re.compile(r'^\s*name\s*=\s*"polars-statistics"\s*$')
    version_re = re.compile(r'^\s*version\s*=\s*"([^"]+)"\s*$')
    for idx, line in enumerate(lines):
        if name_re.match(line):
            # version is typically the next non-name line within this [[package]] block
            for later in lines[idx + 1 : idx + 10]:
                if later.strip().startswith("[[package]]"):
                    break
                match = version_re.match(later)
                if match:
                    return match.group(1)
    errors.append(f"{UV_LOCK}: could not find [[package]] polars-statistics version")
    return None


def _json_simple_version(path: Path, errors: list[str], label: str) -> str | None:
    text = _read_text(path)
    if text is None:
        errors.append(f"missing source: {label}")
        return None
    match = re.search(r'"version"\s*:\s*"([^"]+)"', text)
    if match is None:
        errors.append(f"{label}: could not find version field")
        return None
    return match.group(1)


def _marketplace_version(path: Path, errors: list[str], label: str) -> str | None:
    text = _read_text(path)
    if text is None:
        errors.append(f"missing source: {label}")
        return None
    # Find the plugin entry block whose "name" matches PLUGIN_NAME, then the
    # first "version" field after it (line-based, CRLF-tolerant — no anchoring
    # on end-of-line).
    lines = text.splitlines()
    name_re = re.compile(r'"name"\s*:\s*"' + re.escape(PLUGIN_NAME) + r'"')
    version_re = re.compile(r'"version"\s*:\s*"([^"]+)"')
    found_name = False
    for line in lines:
        if not found_name and name_re.search(line):
            found_name = True
            continue
        if found_name:
            match = version_re.search(line)
            if match:
                return match.group(1)
    errors.append(f"{label}: could not find version field for plugin {PLUGIN_NAME!r}")
    return None


def _skill_frontmatter_version(path: Path, errors: list[str], label: str) -> str | None:
    text = _read_text(path)
    if text is None:
        errors.append(f"missing source: {label}")
        return None
    parts = text.split("---")
    if len(parts) < 3:
        errors.append(f"{label}: no frontmatter block found")
        return None
    frontmatter = parts[1]
    match = re.search(r'^\s*version\s*:\s*"?([^"\n]+?)"?\s*$', frontmatter, re.MULTILINE)
    if match is None:
        errors.append(f"{label}: could not find frontmatter version field")
        return None
    return match.group(1)


def collect_versions(root: Path) -> tuple[dict[str, str], list[str]]:
    """Return ({label: version}, [error messages for missing/unparseable sources])."""
    root = Path(root)
    versions: dict[str, str] = {}
    errors: list[str] = []

    cargo = _cargo_version(root, errors)
    if cargo is not None:
        versions[CARGO_TOML] = cargo

    pyproject = _pyproject_version(root, errors)
    if pyproject is not None:
        versions[PYPROJECT_TOML] = pyproject

    init_py = _init_py_version(root, errors)
    if init_py is not None:
        versions[INIT_PY] = init_py

    uv_lock = _uv_lock_version(root, errors)
    if uv_lock is not None:
        versions[UV_LOCK] = uv_lock

    root_plugin_json = _json_simple_version(root / ROOT_PLUGIN_JSON, errors, ROOT_PLUGIN_JSON)
    if root_plugin_json is not None:
        versions[ROOT_PLUGIN_JSON] = root_plugin_json

    root_marketplace = _marketplace_version(
        root / ROOT_MARKETPLACE_JSON, errors, ROOT_MARKETPLACE_JSON
    )
    if root_marketplace is not None:
        versions[ROOT_MARKETPLACE_JSON] = root_marketplace

    bundled_plugin_json = _json_simple_version(
        root / BUNDLED_PLUGIN_JSON, errors, BUNDLED_PLUGIN_JSON
    )
    if bundled_plugin_json is not None:
        versions[BUNDLED_PLUGIN_JSON] = bundled_plugin_json

    bundled_marketplace = _marketplace_version(
        root / BUNDLED_MARKETPLACE_JSON, errors, BUNDLED_MARKETPLACE_JSON
    )
    if bundled_marketplace is not None:
        versions[BUNDLED_MARKETPLACE_JSON] = bundled_marketplace

    root_skills = sorted(root.glob(ROOT_SKILLS_GLOB))
    if not root_skills:
        errors.append(f"no SKILL.md found matching {ROOT_SKILLS_GLOB}")
    for path in root_skills:
        label = str(path.relative_to(root))
        version = _skill_frontmatter_version(path, errors, label)
        if version is not None:
            versions[label] = version

    bundled_skills = sorted(root.glob(BUNDLED_SKILLS_GLOB))
    if not bundled_skills:
        errors.append(f"no SKILL.md found matching {BUNDLED_SKILLS_GLOB}")
    for path in bundled_skills:
        label = str(path.relative_to(root))
        version = _skill_frontmatter_version(path, errors, label)
        if version is not None:
            versions[label] = version

    return versions, errors


def check(root: Path, expect: str | None = None) -> list[str]:
    """Return a list of failure messages; an empty list means every source agrees."""
    versions, errors = collect_versions(root)
    failures = list(errors)

    unique = set(versions.values())
    if len(unique) > 1:
        for label, version in sorted(versions.items()):
            failures.append(f"{version}  {label} disagrees with other sources")
    elif len(unique) == 1 and expect is not None:
        actual = unique.pop()
        if actual != expect:
            failures.append(f"all sources read {actual!r}, expected --expect {expect!r}")

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect", default=None, help="Expected version, e.g. 0.8.0")
    args = parser.parse_args(argv)

    root = _default_root()
    versions, _errors = collect_versions(root)

    for label, version in sorted(versions.items()):
        print(f"{version}  {label}")

    failures = check(root, expect=args.expect)
    for failure in failures:
        print(f"FAIL {failure}")

    if failures:
        print(f"RESULT: FAIL — {len(failures)} problem(s)")
        return 1

    unique = set(versions.values())
    version_label = sorted(unique)[0] if unique else "?"
    print(f"RESULT: PASS — {len(versions)} version sources aligned at {version_label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
