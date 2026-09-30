"""The LambdaOp function contract: source validation and the function namespace.

A LambdaOp function is ``fn(row: tuple[str, ...]) -> str``, written as a lambda
or as source whose first binding is a one-parameter ``def``. It sees a fixed
namespace: a small set of builtins plus ``json``, ``re``, ``math``, ``np`` and
``pd``. ``np`` and ``pd`` are imported only when the code names them.

The namespace fixes what names resolve; it is not a security boundary, because
the object graph reaches every class without a single builtin. Caller-supplied
code is therefore never materialized in the Lumilake server process: it runs
only where the deployment isolates it, as a FlowMesh ``python`` task
(``runtime/python_step.py``).

This file uses the standard library only. The generated FlowMesh task embeds
its source verbatim, so the task materializes the function with the same rules
the server validates against.
"""

import ast
import builtins
import json
import math
import re
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


def _param_count(args: ast.arguments) -> int:
    count = len(args.posonlyargs) + len(args.args) + len(args.kwonlyargs)
    return count + (1 if args.vararg else 0) + (1 if args.kwarg else 0)


def validate_source(code: str) -> str:
    """Check LambdaOp source without running any of it; return the function name.

    Parsing only: a ``def`` runs its decorators, defaults and annotations when
    executed, and a module body can hold arbitrary statements, so validation
    never executes the code.
    """
    src = code.strip()
    try:
        if src.startswith("lambda"):
            node = ast.parse(src, mode="eval").body
            if not isinstance(node, ast.Lambda):
                raise ValueError("LambdaOp code is not a lambda expression")
            if _param_count(node.args) != 1:
                raise ValueError("Function must accept exactly 1 parameter (a tuple)")
            return "<lambda>"
        module = ast.parse(src, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"LambdaOp code does not parse: {exc}") from exc
    for stmt in module.body:
        if isinstance(stmt, ast.AsyncFunctionDef):
            raise ValueError("LambdaOp function must not be async")
        if isinstance(stmt, ast.FunctionDef):
            if _param_count(stmt.args) != 1:
                raise ValueError(
                    "Function must accept exactly 1 parameter (a tuple), but has"
                    f" {_param_count(stmt.args)} parameters. Expected signature:"
                    " fn(args: tuple[str, ...]) -> str"
                )
            return stmt.name
        if isinstance(
            stmt,
            (ast.Assign, ast.AnnAssign, ast.ClassDef, ast.Import, ast.ImportFrom),
        ):
            break
    raise ValueError("LambdaOp code must define a function as its first binding")


def materialize(code: str) -> Callable[[tuple[Any, ...]], Any]:
    """Build the function from validated source in the LambdaOp namespace."""
    fn_name = validate_source(code)
    src = code.strip()
    used = {node.id for node in ast.walk(ast.parse(src)) if isinstance(node, ast.Name)}
    namespace: dict[str, Any] = {
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
    if src.startswith("lambda"):
        return eval(compile(src, _SOURCE_NAME, "eval"), namespace)
    exec(compile(src, _SOURCE_NAME, "exec"), namespace)
    return namespace[fn_name]
