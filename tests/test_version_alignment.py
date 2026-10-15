"""Tests for scripts/check_version_alignment.py (REL-07).

Loads the script via importlib from its on-disk path (it is not an importable
package module) and runs it against both the live repo tree and tmp_path
copies with deliberately mutated/missing sources. Deliberately does not
hard-code any version literal, so it keeps passing across future version
bumps — it checks *agreement*, not a specific value.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_version_alignment.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("check_version_alignment", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _copy_sources(module, src_root: Path, dst_root: Path) -> None:
    """Copy every source_paths(src_root) file into dst_root, preserving relpath."""
    for path in module.source_paths(src_root):
        rel = path.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)


def test_check_empty_on_current_tree():
    module = _load_script()
    assert module.check(REPO_ROOT) == []


def test_check_fails_on_wrong_expect():
    module = _load_script()
    failures = module.check(REPO_ROOT, expect="9.9.9")
    assert failures


def test_check_passes_on_tmp_copy(tmp_path: Path):
    module = _load_script()
    _copy_sources(module, REPO_ROOT, tmp_path)
    assert module.check(tmp_path) == []


def test_check_fails_on_mutated_skill_version(tmp_path: Path):
    module = _load_script()
    _copy_sources(module, REPO_ROOT, tmp_path)

    mutated = None
    for path in module.source_paths(tmp_path):
        if path.name == "SKILL.md":
            mutated = path
            break
    assert mutated is not None, "expected at least one SKILL.md among source_paths"

    text = mutated.read_text(encoding="utf-8")
    new_text = text.replace('version: "0.8.0"', 'version: "0.0.0"', 1)
    assert new_text != text, "expected to find a version frontmatter field to mutate"
    mutated.write_text(new_text, encoding="utf-8")

    failures = module.check(tmp_path)
    assert failures
    rel = str(mutated.relative_to(tmp_path))
    assert any(rel in failure for failure in failures), failures


def test_check_fails_on_missing_source(tmp_path: Path):
    module = _load_script()
    _copy_sources(module, REPO_ROOT, tmp_path)

    target = tmp_path / "python" / "polars_statistics" / "_plugin" / ".claude-plugin" / "plugin.json"
    assert target.is_file()
    target.unlink()

    failures = module.check(tmp_path)
    assert failures


def test_collect_versions_finds_at_least_18_sources_and_matches_runtime():
    module = _load_script()
    versions, errors = module.collect_versions(REPO_ROOT)
    assert not errors
    assert len(versions) >= 18

    import polars_statistics

    common = set(versions.values())
    assert len(common) == 1
    assert common.pop() == polars_statistics.__version__
