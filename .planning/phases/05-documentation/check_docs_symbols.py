"""
check_docs_symbols.py — DOCS-02 symbol-presence gate for Phase 5 plan 05-03.

Greps docs/ recursively and exits non-zero unless every new symbol introduced in
Phase 3 (statistics) and Phase 4 (regression) appears at least once in the docs tree.

Usage:
    python check_docs_symbols.py [--project-root <path>]

Exits 0 if all symbols found.
Exits 1 if one or more symbols are missing, with a list of missing names.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ─── Symbols required in docs/ ───────────────────────────────────────────────

# New statistics expressions (Phase 3; one_way_anova added by 05-01 tracer)
STATISTICS_SYMBOLS = [
    "two_way_anova",
    "repeated_measures_anova",
    "energy_distance_nd",
    "icc",
]

# New regression PyModel classes (Phase 4)
PYMODEL_SYMBOLS = [
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

# New gamma_* diagnostic expressions (Phase 4)
GAMMA_DIAG_SYMBOLS = [
    "gamma_dispersion_deviance",
    "gamma_dispersion_pearson",
    "gamma_pearson_chi_squared",
    "gamma_standardized_pearson_residuals",
    "gamma_standardized_deviance_residuals",
]

ALL_SYMBOLS = STATISTICS_SYMBOLS + PYMODEL_SYMBOLS + GAMMA_DIAG_SYMBOLS


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


def symbol_in_docs(symbol: str, docs_dir: Path) -> bool:
    """Return True if *symbol* appears at least once in any .md file under docs_dir."""
    result = subprocess.run(
        ["grep", "-rql", symbol, str(docs_dir)],
        capture_output=True,
    )
    return result.returncode == 0


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

    docs_dir = project_root / "docs"
    if not docs_dir.is_dir():
        print(f"ERROR: docs/ directory not found at {docs_dir}", file=sys.stderr)
        return 1

    missing: list[str] = []
    for symbol in ALL_SYMBOLS:
        if not symbol_in_docs(symbol, docs_dir):
            missing.append(symbol)

    if missing:
        print(
            f"FAIL: {len(missing)} symbol(s) not found in {docs_dir.relative_to(project_root)}/:",
            file=sys.stderr,
        )
        for s in missing:
            print(f"  - {s}", file=sys.stderr)
        return 1

    print(
        f"OK: all {len(ALL_SYMBOLS)} required symbols found under "
        f"{docs_dir.relative_to(project_root)}/"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
