#!/usr/bin/env python
"""
Verification harness for polars-statistics SKILL.md code examples.

Extracts every fenced ```python block from all five plugins/polars-statistics/skills/*/SKILL.md
files and executes each against the live installed package.

Usage:
    uv run --extra all python scripts/verify_skill_examples.py
    uv run --extra all python scripts/verify_skill_examples.py --only polars-statistics-results

Exit codes:
    0  All runnable blocks passed (or were explicitly SKIP-marked)
    1  One or more blocks FAILED or had an empty-reason SKIP marker
"""

import argparse
import glob
import os
import re
import sys
import textwrap
import traceback


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SKILLS_GLOB = "plugins/polars-statistics/skills/*/SKILL.md"
SKIP_MARKER = "# verify: skip -- "
EMPTY_REASON_MARKER = "# verify: skip"


# ---------------------------------------------------------------------------
# Startup check
# ---------------------------------------------------------------------------

def check_live_package():
    """Fail fast if the live package cannot be imported."""
    try:
        import polars_statistics  # noqa: F401
    except ImportError as exc:
        print(
            f"\nFATAL: cannot import polars_statistics — {exc}\n"
            "Run this script via: uv run --extra all python scripts/verify_skill_examples.py\n"
            "Ensure the package is installed (maturin develop, or wheel installed).",
            file=sys.stderr,
        )
        sys.exit(2)


# ---------------------------------------------------------------------------
# Block extraction
# ---------------------------------------------------------------------------

def extract_python_blocks(skill_path: str) -> list[dict]:
    """
    Extract all fenced ```python blocks from a SKILL.md file.

    Returns a list of dicts:
        {
            "path": str,           # source file path
            "block_index": int,    # 0-based index among all python blocks in file
            "source": str,         # raw block content (dedented)
            "first_line": str,     # first line of block for classification
            "line_no": int,        # approximate line number in file (1-based)
        }
    """
    with open(skill_path, encoding="utf-8") as fh:
        lines = fh.readlines()

    blocks = []
    block_index = 0
    in_block = False
    fence_char = None
    fence_length = 0
    block_lines = []
    block_start_line = 0

    for line_no, line in enumerate(lines, start=1):
        stripped = line.rstrip("\n")

        if not in_block:
            # Detect opening fence: ``` or ~~~, optionally with language token
            m = re.match(r'^(```+|~~~+)(\w*)', stripped)
            if m:
                fence_char = m.group(1)[0]
                fence_len = len(m.group(1))
                lang_token = m.group(2)
                if lang_token.startswith("python") or lang_token == "py":
                    in_block = True
                    fence_length = fence_len
                    block_lines = []
                    block_start_line = line_no
        else:
            # Detect closing fence: same char, same or greater length, nothing else
            m = re.match(r'^(' + re.escape(fence_char) + r'+)\s*$', stripped)
            if m and len(m.group(1)) >= fence_length:
                # End of block
                source = textwrap.dedent("".join(block_lines))
                first_line = block_lines[0].strip() if block_lines else ""
                blocks.append({
                    "path": skill_path,
                    "block_index": block_index,
                    "source": source,
                    "first_line": first_line,
                    "line_no": block_start_line,
                })
                block_index += 1
                in_block = False
                block_lines = []
                fence_char = None
                fence_length = 0
            else:
                block_lines.append(line.rstrip("\n") + "\n")

    return blocks


# ---------------------------------------------------------------------------
# Block classification
# ---------------------------------------------------------------------------

def classify_block(block: dict) -> tuple[str, str]:
    """
    Return (status, reason) where status is 'runnable' or 'skip'.

    A block is SKIP only when its first content line is exactly:
        # verify: skip -- <non-empty reason>
    Any block without this marker is treated as 'runnable'.
    An empty-reason marker (# verify: skip --  or # verify: skip) is reported as
    a FAIL-equivalent error so lazy blanket-skips cannot hide broken examples.
    """
    first = block["first_line"]

    if first.startswith(SKIP_MARKER):
        reason = first[len(SKIP_MARKER):].strip()
        if reason:
            return "skip", reason
        else:
            return "empty_reason", "(empty reason — required non-empty)"

    # Detect bare marker without the trailing ' -- '
    if re.match(r'^# verify:\s*skip\s*$', first):
        return "empty_reason", "(no reason text — use '# verify: skip -- <reason>')"

    return "runnable", ""


# ---------------------------------------------------------------------------
# Block execution
# ---------------------------------------------------------------------------

def execute_blocks(blocks: list[dict], skill_path: str) -> list[dict]:
    """
    Execute all runnable blocks from a single skill file in document order,
    sharing a single namespace across blocks in the same file.

    Returns a list of result dicts:
        {
            "path": str,
            "block_index": int,
            "line_no": int,
            "status": "PASS" | "FAIL" | "SKIP" | "EMPTY_REASON_FAIL",
            "reason": str,      # non-empty for SKIP
            "error": str,       # non-empty for FAIL
        }
    """
    results = []
    shared_namespace: dict = {}  # reset per file (caller's responsibility)

    for block in blocks:
        status_class, reason = classify_block(block)

        if status_class == "skip":
            results.append({
                "path": block["path"],
                "block_index": block["block_index"],
                "line_no": block["line_no"],
                "status": "SKIP",
                "reason": reason,
                "error": "",
            })
            continue

        if status_class == "empty_reason":
            results.append({
                "path": block["path"],
                "block_index": block["block_index"],
                "line_no": block["line_no"],
                "status": "EMPTY_REASON_FAIL",
                "reason": "",
                "error": f"SKIP marker present but reason is empty: {block['first_line']!r}",
            })
            continue

        # Runnable block — execute in shared namespace
        try:
            exec(compile(block["source"], f"{skill_path}:block{block['block_index']}", "exec"),
                 shared_namespace)
            results.append({
                "path": block["path"],
                "block_index": block["block_index"],
                "line_no": block["line_no"],
                "status": "PASS",
                "reason": "",
                "error": "",
            })
        except Exception:
            results.append({
                "path": block["path"],
                "block_index": block["block_index"],
                "line_no": block["line_no"],
                "status": "FAIL",
                "reason": "",
                "error": traceback.format_exc(),
            })

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def skill_name_from_path(path: str) -> str:
    """Extract the skill directory name from a SKILL.md path."""
    parts = path.replace("\\", "/").split("/")
    # plugins/polars-statistics/skills/<skill-name>/SKILL.md
    try:
        idx = parts.index("skills")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return os.path.dirname(path)


def print_report(all_results: list[dict], skill_paths_processed: list[str]) -> int:
    """
    Print per-block PASS/FAIL/SKIP report and summary.

    Returns non-zero if any block FAILed or had an EMPTY_REASON_FAIL.
    """
    total_pass = 0
    total_fail = 0
    total_skip = 0
    total_empty_reason = 0

    for path in skill_paths_processed:
        skill = skill_name_from_path(path)
        results_for_skill = [r for r in all_results if r["path"] == path]
        if not results_for_skill:
            print(f"\n[{skill}]  (no python blocks found)")
            continue

        print(f"\n[{skill}]  ({len(results_for_skill)} block(s))")
        for r in results_for_skill:
            block_id = f"block {r['block_index']:02d} @ line {r['line_no']}"
            if r["status"] == "PASS":
                total_pass += 1
                print(f"  PASS   {block_id}")
            elif r["status"] == "FAIL":
                total_fail += 1
                print(f"  FAIL   {block_id}")
                # Indent traceback
                for tb_line in r["error"].splitlines():
                    print(f"         {tb_line}")
            elif r["status"] == "SKIP":
                total_skip += 1
                print(f"  SKIP   {block_id}  -- {r['reason']}")
            elif r["status"] == "EMPTY_REASON_FAIL":
                total_empty_reason += 1
                print(f"  FAIL   {block_id}  [empty-reason skip marker]")
                print(f"         {r['error']}")

    grand_total = total_pass + total_fail + total_skip + total_empty_reason
    print(
        f"\n{'=' * 60}\n"
        f"SUMMARY  {len(skill_paths_processed)} skill(s) | "
        f"{grand_total} block(s) total: "
        f"{total_pass} PASS, {total_fail + total_empty_reason} FAIL, {total_skip} SKIP"
    )

    if total_fail + total_empty_reason == 0:
        print("RESULT: GREEN — all runnable blocks passed")
        return 0
    else:
        print(f"RESULT: RED — {total_fail + total_empty_reason} block(s) failed")
        return 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify polars-statistics SKILL.md code examples against the live package."
    )
    parser.add_argument(
        "--only",
        metavar="SKILL_DIR_NAME",
        help="Run only the named skill directory (e.g. polars-statistics-results)",
        default=None,
    )
    args = parser.parse_args()

    # Startup check
    check_live_package()

    # Discover skill files — glob, never a hardcoded list
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pattern = os.path.join(repo_root, SKILLS_GLOB)
    discovered = sorted(glob.glob(pattern))

    if not discovered:
        print(f"ERROR: No SKILL.md files found matching: {pattern}", file=sys.stderr)
        return 1

    if args.only:
        skill_paths = [p for p in discovered if skill_name_from_path(p) == args.only]
        if not skill_paths:
            available = [skill_name_from_path(p) for p in discovered]
            print(
                f"ERROR: --only '{args.only}' not found. Available: {available}",
                file=sys.stderr,
            )
            return 1
    else:
        skill_paths = discovered

    print(f"polars-statistics SKILL.md verification harness")
    print(f"Skills: {[skill_name_from_path(p) for p in skill_paths]}")

    all_results: list[dict] = []
    for path in skill_paths:
        blocks = extract_python_blocks(path)
        results = execute_blocks(blocks, path)
        all_results.extend(results)

    return print_report(all_results, skill_paths)


if __name__ == "__main__":
    sys.exit(main())
