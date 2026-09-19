"""Drift guard for type stubs (TYPE-03).

Ensures the checked-in type stubs stay in sync with the runtime API so they
cannot silently rot:

1. Every public class in the compiled ``polars_statistics._polars_statistics``
   module has a ``class`` entry in ``_polars_statistics.pyi``.
2. Every runtime member (getters/methods) of each class is present in the stub.
3. Every re-exported expression builder / formula helper on the top-level
   ``polars_statistics`` package is typed (has inline annotations or a stub).

If any public symbol is missing from the stubs, the test fails with an
actionable message pointing at ``scripts/gen_stubs.py``.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import polars as pl
import polars_statistics as ps
import polars_statistics._polars_statistics as _rust

PKG_DIR = Path(ps.__file__).parent
PYI_PATH = PKG_DIR / "_polars_statistics.pyi"


def _parse_stub() -> dict[str, set[str]]:
    """Return {class_name: {member_names}} parsed from the .pyi stub."""
    tree = ast.parse(PYI_PATH.read_text())
    classes: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            members: set[str] = set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members.add(item.name)
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    members.add(item.target.id)
            classes[node.name] = members
    return classes


def _runtime_classes() -> dict[str, type]:
    return {
        name: obj
        for name in dir(_rust)
        if not name.startswith("_")
        and inspect.isclass(obj := getattr(_rust, name))
    }


def test_stub_file_exists_and_marker_present():
    assert PYI_PATH.exists(), f"Missing type stub: {PYI_PATH}"
    assert (PKG_DIR / "py.typed").exists(), "Missing py.typed marker"


def test_every_runtime_class_is_in_stub():
    stub = _parse_stub()
    runtime = _runtime_classes()
    missing = sorted(set(runtime) - set(stub))
    assert not missing, (
        f"Type stub is missing classes: {missing}. "
        "Regenerate with `.venv/bin/python scripts/gen_stubs.py`."
    )


def test_every_runtime_member_is_in_stub():
    stub = _parse_stub()
    runtime = _runtime_classes()
    errors: list[str] = []
    for cname, cls in runtime.items():
        stub_members = stub.get(cname, set())
        runtime_members = {
            m for m in dir(cls) if not m.startswith("__")
        }
        missing = sorted(runtime_members - stub_members)
        if missing:
            errors.append(f"{cname}: missing {missing}")
    assert not errors, (
        "Type stub out of sync with runtime members:\n  "
        + "\n  ".join(errors)
        + "\nRegenerate with `.venv/bin/python scripts/gen_stubs.py`."
    )


def test_stub_has_no_stale_classes():
    """Every class declared in the stub must exist at runtime."""
    stub = _parse_stub()
    runtime = _runtime_classes()
    # AidResult etc. are all runtime classes; the stub should not invent ones.
    stale = sorted(set(stub) - set(runtime))
    assert not stale, (
        f"Type stub declares classes not present at runtime: {stale}. "
        "Regenerate with `.venv/bin/python scripts/gen_stubs.py`."
    )


def _public_expr_builders() -> list[str]:
    """Top-level callables re-exported from the exprs package (TYPE-02)."""
    names = []
    for name in ps.__all__:
        obj = getattr(ps, name, None)
        if inspect.isfunction(obj):
            names.append(name)
    return names


def test_expression_builders_are_typed():
    """Every public expression builder has a full return + param annotation."""
    untyped: list[str] = []
    for name in _public_expr_builders():
        fn = getattr(ps, name)
        try:
            sig = inspect.signature(fn)
        except (ValueError, TypeError):
            continue
        if sig.return_annotation is inspect.Signature.empty:
            untyped.append(f"{name} (no return annotation)")
            continue
        for pname, param in sig.parameters.items():
            if param.annotation is inspect.Parameter.empty:
                untyped.append(f"{name}:{pname} (no annotation)")
    assert not untyped, "Expression builders missing type annotations:\n  " + "\n  ".join(
        untyped
    )


def test_expression_builders_return_pl_expr():
    """Spot-check that the annotated return type resolves to polars.Expr."""
    for name in ["ols", "ttest_ind", "pearson", "logistic"]:
        fn = getattr(ps, name)
        sig = inspect.signature(fn)
        ret = sig.return_annotation
        # Annotation may be a string ("pl.Expr") under `from __future__`.
        assert "Expr" in str(ret), f"{name} does not return pl.Expr (got {ret!r})"
    # And a concrete build actually returns a polars Expr:
    assert isinstance(ps.ttest_ind("a", "b"), pl.Expr)
    assert isinstance(ps.ols("y", "x"), pl.Expr)
