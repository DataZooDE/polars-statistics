"""
check_changelog.py — DOCS-04 gate for Phase 5 plan 05-04.

Reads CHANGELOG.md and exits non-zero unless it finds:
  - The 0.6.0 heading
  - Every new statistics symbol (one_way_anova, two_way_anova, repeated_measures_anova,
    energy_distance_nd, icc)
  - Every new regression symbol (Gamma, GLMM, PSpline, TheilSen, RANSAC, BayesianRidge,
    ARD, LARS, PassiveAggressive, MomentAccumulator)
  - Both crate version strings (0.4.2 and 0.5.13)
  - Both limitation keywords (NotImplementedError and ICC)

Usage:
    python check_changelog.py [--project-root <path>]

Exits 0 on success; exits 1 with a list of missing items.
"""

import argparse
import re
import sys
from pathlib import Path


# ─── Required strings ─────────────────────────────────────────────────────────

REQUIRED_HEADING = "## [0.6.0]"

STATISTICS_SYMBOLS = [
    "one_way_anova",
    "two_way_anova",
    "repeated_measures_anova",
    "energy_distance_nd",
    "icc",
]

REGRESSION_SYMBOLS = [
    "Gamma",
    "GLMM",
    "PSpline",
    "TheilSen",
    "RANSAC",
    "BayesianRidge",
    "ARD",
    "LARS",
    "PassiveAggressive",
    "MomentAccumulator",
]

CRATE_VERSIONS = [
    "0.4.2",   # anofox-statistics new version
    "0.5.13",  # anofox-regression new version
]

LIMITATION_KEYWORDS = [
    "NotImplementedError",
    "ICC",
]

ALL_REQUIRED = (
    [REQUIRED_HEADING]
    + STATISTICS_SYMBOLS
    + REGRESSION_SYMBOLS
    + CRATE_VERSIONS
    + LIMITATION_KEYWORDS
)


def find_project_root(start: Path) -> Path:
    """Walk up from *start* until we find CHANGELOG.md."""
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "CHANGELOG.md").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find CHANGELOG.md above {start}. "
        "Run this script from inside the project directory."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Path to the project root containing CHANGELOG.md (default: auto-detect)",
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

    changelog_path = project_root / "CHANGELOG.md"
    if not changelog_path.exists():
        print(f"ERROR: CHANGELOG.md not found at {changelog_path}", file=sys.stderr)
        return 1

    content = changelog_path.read_text(encoding="utf-8")

    missing: list[str] = []
    for item in ALL_REQUIRED:
        if item not in content:
            missing.append(item)

    if missing:
        print(
            f"FAIL: {len(missing)} required item(s) not found in CHANGELOG.md:",
            file=sys.stderr,
        )
        for m in missing:
            print(f"  - {m!r}", file=sys.stderr)
        return 1

    print(
        f"OK: CHANGELOG.md 0.6.0 section contains all "
        f"{len(ALL_REQUIRED)} required items "
        f"({len(STATISTICS_SYMBOLS)} statistics symbols, "
        f"{len(REGRESSION_SYMBOLS)} regression symbols, "
        f"{len(CRATE_VERSIONS)} crate versions, "
        f"{len(LIMITATION_KEYWORDS)} limitation keywords)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
