"""
check_rust_docs.py — DOCS-03 gate for Phase 5 plans.

Verifies that:
1. Every new PyModel struct carries a `///` class doc block with a runnable `>>>` example.
2. Every new `pub fn *_fit` expression wrapper has a `///` line immediately above it.
3. `py_wls.rs` `hc_inference` doc mentions `NotImplementedError`.

Usage:
    python .planning/phases/05-documentation/check_rust_docs.py

Exits 0 if all checks pass, 1 otherwise (prints failures to stderr).
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def read_file(rel_path: str) -> str:
    return (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")


def check_pyclass_has_example(content: str, struct_name: str) -> tuple[bool, str]:
    """Assert that the /// block immediately above #[pyclass ... struct_name has a >>> line."""
    # Find the struct definition
    struct_pat = re.compile(
        r'((?:///[^\n]*\n)+)'   # capture preceding /// lines
        r'\s*#\[pyclass[^\]]*\]\s*\n'
        r'pub struct ' + re.escape(struct_name),
        re.MULTILINE,
    )
    m = struct_pat.search(content)
    if not m:
        return False, f"No #[pyclass] struct '{struct_name}' with preceding /// block found"
    doc_block = m.group(1)
    if ">>>" not in doc_block:
        return False, f"Class doc for '{struct_name}' is missing a '>>>' runnable example"
    return True, f"OK: '{struct_name}' has /// class doc with runnable example"


def check_fn_has_doc(content: str, fn_name: str) -> tuple[bool, str]:
    """Assert that a `///` line immediately precedes `pub fn fn_name`."""
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if re.match(r'\s*pub fn ' + re.escape(fn_name) + r'\b', line):
            # Look backwards for a non-blank line
            for j in range(i - 1, max(i - 5, -1), -1):
                prev = lines[j].strip()
                if prev:
                    if prev.startswith("///"):
                        return True, f"OK: 'pub fn {fn_name}' has a /// doc comment"
                    else:
                        return False, (
                            f"'pub fn {fn_name}' is not immediately preceded by a /// line "
                            f"(found: {prev!r})"
                        )
            return False, f"'pub fn {fn_name}' has no preceding doc line"
    return False, f"'pub fn {fn_name}' not found in file"


def check_wls_hc_mentions_not_implemented(content: str) -> tuple[bool, str]:
    """Assert that py_wls.rs hc_inference doc mentions NotImplementedError."""
    lines = content.splitlines()
    # Find the line with fn hc_inference
    fn_line = None
    for i, line in enumerate(lines):
        if re.match(r'\s*fn hc_inference\b', line):
            fn_line = i
            break
    if fn_line is None:
        return False, "Could not find 'fn hc_inference' in py_wls.rs"
    # Walk backwards collecting all /// lines (skip #[...] attribute lines and blank lines)
    doc_lines: list[str] = []
    for j in range(fn_line - 1, max(fn_line - 60, -1), -1):
        stripped = lines[j].strip()
        if stripped.startswith("///"):
            doc_lines.insert(0, stripped)
        elif stripped.startswith("#[") or stripped == "":
            # skip attribute/blank lines between doc and fn
            continue
        else:
            # hit non-doc, non-attribute code
            break
    doc_block = "\n".join(doc_lines)
    if "NotImplementedError" not in doc_block:
        return False, (
            "py_wls.rs hc_inference /// doc does not mention 'NotImplementedError' "
            f"(collected doc block: {doc_block[:200]!r})"
        )
    return True, "OK: py_wls.rs hc_inference doc mentions NotImplementedError"


def main() -> int:
    failures: list[str] = []

    # ---- 1. New PyModel structs: class doc + >>> example ----
    pymodel_structs = {
        "src/pymodels/py_gamma.rs": "PyGamma",
        "src/pymodels/py_glmm.rs": "PyGLMM",
        "src/pymodels/py_pspline.rs": "PyPSpline",
        "src/pymodels/py_theil_sen.rs": "PyTheilSen",
        "src/pymodels/py_ransac.rs": "PyRANSAC",
        "src/pymodels/py_bayesian_ridge.rs": "PyBayesianRidge",
        "src/pymodels/py_ard.rs": "PyARD",
        "src/pymodels/py_lars.rs": "PyLARS",
        "src/pymodels/py_passive_aggressive.rs": "PyPassiveAggressive",
        "src/pymodels/py_moment_accumulator.rs": "PyMomentAccumulator",
    }
    for rel_path, struct_name in pymodel_structs.items():
        try:
            content = read_file(rel_path)
        except FileNotFoundError:
            failures.append(f"MISSING FILE: {rel_path}")
            continue
        ok, msg = check_pyclass_has_example(content, struct_name)
        if not ok:
            failures.append(f"[{rel_path}] {msg}")
        else:
            print(msg)

    # ---- 2. New *_fit expression wrappers: /// immediately above fn ----
    expression_fns = {
        "src/expressions/parametric.rs": [
            "two_way_anova_fit",
            "repeated_measures_anova_fit",
        ],
        "src/expressions/modern.rs": [
            "energy_distance_nd_fit",
        ],
        "src/expressions/correlation.rs": [
            "icc_fit",
        ],
        "src/expressions/regression.rs": [
            "gamma_dispersion_deviance_fit",
            "gamma_dispersion_pearson_fit",
            "gamma_pearson_chi_squared_fit",
            "gamma_standardized_pearson_residuals_fit",
            "gamma_standardized_deviance_residuals_fit",
        ],
    }
    for rel_path, fn_names in expression_fns.items():
        try:
            content = read_file(rel_path)
        except FileNotFoundError:
            failures.append(f"MISSING FILE: {rel_path}")
            continue
        for fn_name in fn_names:
            ok, msg = check_fn_has_doc(content, fn_name)
            if not ok:
                failures.append(f"[{rel_path}] {msg}")
            else:
                print(msg)

    # ---- 3. WLS hc_inference NotImplementedError caveat ----
    try:
        wls_content = read_file("src/pymodels/py_wls.rs")
        ok, msg = check_wls_hc_mentions_not_implemented(wls_content)
        if not ok:
            failures.append(f"[src/pymodels/py_wls.rs] {msg}")
        else:
            print(msg)
    except FileNotFoundError:
        failures.append("MISSING FILE: src/pymodels/py_wls.rs")

    # ---- Summary ----
    if failures:
        print("\nFAILURES:", file=sys.stderr)
        for f in failures:
            print(f"  FAIL: {f}", file=sys.stderr)
        return 1

    print("\nAll Rust doc checks PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
