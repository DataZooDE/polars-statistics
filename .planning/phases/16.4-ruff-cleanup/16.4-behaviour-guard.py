"""Behaviour-neutrality guard for Phase 16.4 (ruff cleanup).

Two subcommands:

``api OUT_JSON``
    Import ``polars_statistics`` and write a public-API fingerprint (defining
    module + qualname + parameter shape + ``get_type_hints`` resolution per
    public name) to ``OUT_JSON``. Used to prove that annotation modernisation
    and import reordering do not change which objects the public API exposes.

``ast REF PATH...``
    Compare each ``PATH`` against the version at git ref ``REF``, after
    normalising away import statements, all annotations, and constant-only
    f-strings. Proves that a ruff ``--unsafe-fixes`` run changed nothing
    except those cosmetic categories.
"""

from __future__ import annotations

import ast
import inspect
import json
import pkgutil
import subprocess
import sys
import typing


def _iter_submodule_names(package) -> list[str]:
    names = []
    for info in pkgutil.iter_modules(package.__path__):
        names.append(f"{package.__name__}.{info.name}")
    return names


def _public_api_modules():
    import polars_statistics
    import polars_statistics.exprs
    import polars_statistics.formula
    import polars_statistics.models
    import polars_statistics.results

    modules = {
        "polars_statistics": polars_statistics,
        "polars_statistics.exprs": polars_statistics.exprs,
        "polars_statistics.models": polars_statistics.models,
        "polars_statistics.formula": polars_statistics.formula,
        "polars_statistics.results": polars_statistics.results,
    }
    for name in _iter_submodule_names(polars_statistics.exprs):
        modules[name] = __import__(name, fromlist=["_"])
    for name in _iter_submodule_names(polars_statistics.formula):
        modules[name] = __import__(name, fromlist=["_"])
    return dict(sorted(modules.items()))


def _param_entry(obj):
    try:
        sig = inspect.signature(obj)
    except (ValueError, TypeError):
        return None
    params = []
    for pname, param in sig.parameters.items():
        params.append([pname, param.kind.name, repr(param.default)])
    return params


def _hints_resolve(obj):
    if not inspect.isfunction(obj):
        return None
    try:
        typing.get_type_hints(obj)
        return True
    except Exception:
        return False


def cmd_api(out_path: str) -> int:
    import polars_statistics

    print(f"fingerprinted: {polars_statistics.__file__}")

    modules = _public_api_modules()
    fingerprint = {}
    entries = 0

    for mod_name, module in modules.items():
        all_names = getattr(module, "__all__", None)
        fingerprint[f"{mod_name}.__all__"] = (
            sorted(all_names) if all_names is not None else None
        )
        for name in sorted(dir(module)):
            if name.startswith("__") and name.endswith("__"):
                continue
            obj = getattr(module, name)
            obj_module = getattr(obj, "__module__", None)
            is_in_all = all_names is not None and name in all_names
            is_submodule = inspect.ismodule(obj) and str(
                getattr(obj, "__name__", "")
            ).startswith("polars_statistics")
            is_own_object = isinstance(obj_module, str) and obj_module.startswith(
                "polars_statistics"
            )
            if not (is_in_all or is_submodule or is_own_object):
                continue

            key = f"{mod_name}.{name}"
            if inspect.ismodule(obj):
                fingerprint[key] = ["module", obj.__name__]
            else:
                fingerprint[key] = {
                    "module": obj_module,
                    "qualname": getattr(obj, "__qualname__", type(obj).__name__),
                    "params": _param_entry(obj),
                    "hints_resolve": _hints_resolve(obj),
                }
            entries += 1

    with open(out_path, "w") as f:
        json.dump(fingerprint, f, sort_keys=True, indent=1)

    print(f"entries: {entries}")
    return 0


class _NormalizeTransformer(ast.NodeTransformer):
    """Strip imports, annotations, and constant f-strings for AST comparison."""

    _ANNOTATION_PLACEHOLDER = ast.Constant(value="__ANNOTATION__")

    def visit_Import(self, node):  # noqa: N802
        return None

    def visit_ImportFrom(self, node):  # noqa: N802
        return None

    def _strip_args(self, args: ast.arguments) -> None:
        for arg_list_name in ("posonlyargs", "args", "kwonlyargs"):
            for a in getattr(args, arg_list_name, []):
                a.annotation = None
        if args.vararg is not None:
            args.vararg.annotation = None
        if args.kwarg is not None:
            args.kwarg.annotation = None

    def visit_FunctionDef(self, node):  # noqa: N802
        node.returns = None
        self._strip_args(node.args)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node):  # noqa: N802
        node.returns = None
        self._strip_args(node.args)
        self.generic_visit(node)
        return node

    def visit_AnnAssign(self, node):  # noqa: N802
        node.annotation = self._ANNOTATION_PLACEHOLDER
        self.generic_visit(node)
        return node

    def visit_JoinedStr(self, node):  # noqa: N802
        self.generic_visit(node)
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                return node
        return ast.copy_location(ast.Constant(value="".join(parts)), node)


def _normalized_dump(source: str) -> str:
    tree = ast.parse(source)
    tree = _NormalizeTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, include_attributes=False)


def cmd_ast(ref: str, paths: list[str]) -> int:
    if not paths:
        return 2

    total = len(paths)
    equal_count = 0
    any_missing = False

    for path in paths:
        try:
            old_source = subprocess.run(
                ["git", "show", f"{ref}:{path}"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except subprocess.CalledProcessError:
            print(f"DIFFERS: {path}")
            any_missing = True
            continue

        with open(path) as f:
            new_source = f.read()

        old_dump = _normalized_dump(old_source)
        new_dump = _normalized_dump(new_source)

        if old_dump == new_dump:
            equal_count += 1
        else:
            print(f"DIFFERS: {path}")

    print(f"AST-equal: {equal_count} of {total}")

    if any_missing or equal_count != total:
        return 1
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: 16.4-behaviour-guard.py api OUT_JSON | ast REF PATH...", file=sys.stderr)
        return 2

    subcommand, *rest = argv
    if subcommand == "api":
        if len(rest) != 1:
            print("usage: 16.4-behaviour-guard.py api OUT_JSON", file=sys.stderr)
            return 2
        return cmd_api(rest[0])
    if subcommand == "ast":
        if len(rest) < 1:
            print("usage: 16.4-behaviour-guard.py ast REF PATH...", file=sys.stderr)
            return 2
        ref, *paths = rest
        if not paths:
            return 2
        return cmd_ast(ref, paths)

    print(f"unknown subcommand: {subcommand}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
