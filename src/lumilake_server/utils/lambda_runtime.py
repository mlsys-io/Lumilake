"""The LambdaOp function contract: source validation and the function namespace.

A LambdaOp function is ``fn(row: tuple[str, ...]) -> str``, written as a lambda
or as module source that defines a one-parameter ``def`` named by ``fn_name``.
The source may put imports, constants and helper functions before that ``def``.
It sees a fixed namespace: a small set of builtins plus ``json``, ``re``,
``math``, ``np`` and ``pd`` (``np``/``pd`` imported only when the code names
them), and it may ``import`` any standard-library module, numpy or pandas.

The namespace fixes what names resolve; it is not a security boundary, because
the object graph reaches every class without a single builtin. Caller-supplied
code is therefore never materialized in the Lumilake server process: it runs
only where the deployment isolates it, as a FlowMesh ``python`` task
(``runtime/python_step.py``) in a per-task container with no network and an
unprivileged uid. That container, not this namespace, is what makes allowing
standard-library imports safe.

A LambdaOp that feeds an LLM is different: it is inlined into the LLM's FlowMesh
task as a ``graph_template`` function and evaluated by the FlowMesh worker's own
``safe_eval`` namespace, which has no ``__import__`` and binds the first name the
code defines. ``validate_inline_source`` holds such code to that stricter
grammar so it fails at submit with a clear message rather than at run time.

This file uses the standard library only. The generated FlowMesh task embeds
its source verbatim, so the task materializes the function with the same rules
the server validates against.
"""

import ast
import builtins
import json
import math
import re
import sys
import types
from collections.abc import Callable
from typing import Any

SAFE_BUILTIN_NAMES = (
    "int",
    "float",
    "str",
    "bool",
    "list",
    "dict",
    "tuple",
    "set",
    "len",
    "sum",
    "max",
    "min",
    "abs",
    "round",
    "sorted",
    "reversed",
    "enumerate",
    "zip",
    "map",
    "filter",
    "any",
    "all",
    "range",
    "isinstance",
)

_SOURCE_NAME = "<lambda_op>"
_LIBRARY_ROOTS = ("numpy", "pandas")


def _importable(name: str) -> bool:
    root = name.partition(".")[0]
    return root in sys.stdlib_module_names or root in _LIBRARY_ROOTS


def _not_importable_message(name: str) -> str:
    return (
        f"import of {name!r} is not available in a LambdaOp function; only the"
        " Python standard library, numpy and pandas can be imported"
    )


def _lambda_import(
    name: str,
    globals: Any = None,
    locals: Any = None,
    fromlist: Any = (),
    level: int = 0,
) -> Any:
    """``__import__`` for LambdaOp code: the standard library, numpy and pandas.

    CPython resolves ``__import__`` in the calling frame's builtins, which are
    restricted here, so both the caller's own ``import`` statements and the
    lazy imports numpy/pandas make from C land in this hook.
    """
    if level != 0:
        raise ImportError("relative imports are not available in a LambdaOp function")
    if not _importable(name):
        raise ImportError(_not_importable_message(name))
    return builtins.__import__(name, globals, locals, fromlist, level)


def _imported_names(module: ast.AST) -> list[tuple[str, int]]:
    """Every ``(module name, level)`` an import statement in ``module`` names."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            found.extend((alias.name, 0) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append((node.module or "", node.level))
    return found


def _param_count(args: ast.arguments) -> int:
    count = len(args.posonlyargs) + len(args.args) + len(args.kwonlyargs)
    return count + (1 if args.vararg else 0) + (1 if args.kwarg else 0)


def _parse(code: str) -> ast.Expression | ast.Module:
    src = code.strip()
    try:
        if src.startswith("lambda"):
            return ast.parse(src, mode="eval")
        return ast.parse(src, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"LambdaOp code does not parse: {exc}") from exc


def _check_lambda(tree: ast.Expression) -> str:
    if not isinstance(tree.body, ast.Lambda):
        raise ValueError("LambdaOp code is not a lambda expression")
    if _param_count(tree.body.args) != 1:
        raise ValueError("Function must accept exactly 1 parameter (a tuple)")
    return "<lambda>"


def _check_def(stmt: ast.FunctionDef) -> str:
    if _param_count(stmt.args) != 1:
        raise ValueError(
            "Function must accept exactly 1 parameter (a tuple), but has"
            f" {_param_count(stmt.args)} parameters. Expected signature:"
            " fn(args: tuple[str, ...]) -> str"
        )
    return stmt.name


def validate_source(code: str, fn_name: str | None = None) -> str:
    """Check LambdaOp source without running any of it; return the function name.

    Parsing only: a ``def`` runs its decorators, defaults and annotations when
    executed, and a module body can hold arbitrary statements, so validation
    never executes the code.

    ``fn_name`` is the name the LambdaOp declares: the code must define a
    top-level one-parameter function of that name. Imports, constants and
    helper functions may come before it. Without ``fn_name`` the first
    top-level function is used. A lambda has no name, so ``fn_name`` labels it
    only. Every import must name the standard library, numpy or pandas.
    """
    tree = _parse(code)
    if isinstance(tree, ast.Expression):
        return _check_lambda(tree)
    for name, level in _imported_names(tree):
        if level != 0:
            raise ValueError("LambdaOp code must not use relative imports")
        if not _importable(name):
            raise ValueError(_not_importable_message(name))
    defs = [
        stmt
        for stmt in tree.body
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if fn_name is None:
        target = defs[0] if defs else None
    else:
        target = next((d for d in defs if d.name == fn_name), None)
    if target is None:
        if fn_name is None:
            raise ValueError("LambdaOp code must define a top-level function")
        defined = ", ".join(repr(d.name) for d in defs) or "none"
        raise ValueError(
            f"LambdaOp code defines no top-level function named {fn_name!r}"
            f" (fn_name); top-level functions defined: {defined}"
        )
    if isinstance(target, ast.AsyncFunctionDef):
        raise ValueError("LambdaOp function must not be async")
    return _check_def(target)


def validate_inline_source(code: str, fn_name: str | None = None) -> str:
    """Check source for a LambdaOp the FlowMesh worker evaluates inline.

    A LambdaOp that feeds an LLM becomes a ``graph_template`` function step,
    which the FlowMesh worker materializes in its own restricted namespace: no
    ``__import__``, and the function is the first name the code binds. Code
    that a standalone LambdaOp accepts but that namespace cannot run is
    rejected here, at submit, with the reason.
    """
    name = validate_source(code, fn_name)
    tree = _parse(code)
    if isinstance(tree, ast.Expression):
        return name
    imports = _imported_names(tree)
    if imports:
        raise ValueError(
            f"it imports {imports[0][0]!r}, but a LambdaOp that feeds an LLM is"
            " evaluated inline by the FlowMesh worker, where import statements"
            " are not available. Use the preloaded json, re, math, np and pd"
            " modules, or move this logic to a LambdaOp that is a workflow"
            " output or feeds only other LambdaOps"
        )
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            if stmt.name != name:
                break
            return name
        if isinstance(
            stmt,
            (
                ast.Assign,
                ast.AnnAssign,
                ast.AugAssign,
                ast.ClassDef,
                ast.AsyncFunctionDef,
            ),
        ):
            break
    raise ValueError(
        f"a LambdaOp that feeds an LLM must define {name!r} as the first name"
        " its code binds (the FlowMesh worker calls the first definition), so"
        " move constants and helper functions inside it"
    )


def materialize(
    code: str, fn_name: str | None = None
) -> Callable[[tuple[Any, ...]], Any]:
    """Build the function from validated source in the LambdaOp namespace.

    ``fn_name`` is checked against the source as in ``validate_source``.
    """
    defined_name = validate_source(code, fn_name)
    src = code.strip()
    used = {node.id for node in ast.walk(ast.parse(src)) if isinstance(node, ast.Name)}
    namespace: dict[str, Any] = {
        "__name__": "lambda_op",
        "__builtins__": {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES},
        "json": json,
        "re": re,
        "math": math,
        # SDK-serialized code may name these in annotations, which run at def time.
        "Message": dict,
        "ops": types.SimpleNamespace(
            SingleDtype=str | list[dict[str, str]], Message=dict
        ),
    }
    if "np" in used:
        import numpy

        namespace["np"] = numpy
    if "pd" in used:
        import pandas

        namespace["pd"] = pandas
    namespace["__builtins__"]["__import__"] = _lambda_import
    if src.startswith("lambda"):
        return eval(compile(src, _SOURCE_NAME, "eval"), namespace)
    exec(compile(src, _SOURCE_NAME, "exec"), namespace)
    return namespace[defined_name]
