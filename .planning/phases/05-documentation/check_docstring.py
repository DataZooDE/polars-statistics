"""
check_docstring.py — DOCS-01 gate for Phase 5 plans.

Usage:
    python check_docstring.py <symbol_name> <python_file_path>

Exits 0 if the named function has a docstring that contains:
  - a "Parameters" section
  - at least one runnable ">>>" example line

Exits 1 otherwise, printing a descriptive error.

Example:
    python .planning/phases/05-documentation/check_docstring.py \\
        one_way_anova python/polars_statistics/exprs/parametric.py
"""

import ast
import sys


def find_function(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    """Return the first top-level FunctionDef with the given name, or None."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def check_docstring(symbol: str, filepath: str) -> tuple[bool, str]:
    """
    Parse *filepath* and check that *symbol*'s docstring meets DOCS-01 requirements.

    Returns (ok: bool, message: str).
    """
    try:
        with open(filepath, encoding="utf-8") as fh:
            source = fh.read()
    except FileNotFoundError:
        return False, f"File not found: {filepath}"

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as exc:
        return False, f"Syntax error in {filepath}: {exc}"

    node = find_function(tree, symbol)
    if node is None:
        return False, f"Symbol '{symbol}' not found in {filepath}"

    docstring = ast.get_docstring(node)
    if not docstring:
        return False, f"'{symbol}' has no docstring"

    if "Parameters" not in docstring:
        return False, f"'{symbol}' docstring is missing a 'Parameters' section"

    if ">>>" not in docstring:
        return False, f"'{symbol}' docstring has no runnable '>>>' example lines"

    return True, f"OK: '{symbol}' has Parameters section and runnable examples"


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <symbol_name> <python_file_path>", file=sys.stderr)
        return 2

    symbol = sys.argv[1]
    filepath = sys.argv[2]

    ok, message = check_docstring(symbol, filepath)
    if ok:
        print(message)
        return 0
    else:
        print(f"FAIL: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
