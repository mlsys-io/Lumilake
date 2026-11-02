"""Runtime used by generated standalone LambdaOp and reshape FlowMesh tasks."""

import ast
import builtins
import json
import math
import os
import re
import symtable
import types
from collections.abc import Callable
from typing import Any


def _text(value):
    return value if isinstance(value, str) else json.dumps(value)


def _walk_path(value, parts, where):
    for idx, part in enumerate(parts):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                raise ValueError(f"bad JSON at {where}")
        if (
            isinstance(value, dict)
            and list(value) == ["df"]
            and isinstance(value["df"], str)
        ):
            try:
                table = json.loads(value["df"])
            except ValueError:
                raise ValueError(f"bad JSON at {where}")
            if part not in table:
                raise ValueError(f"missing {part!r} at {where}")
            column = table[part]
            if not isinstance(column, dict):
                raise ValueError(f"missing {part!r} at {where}")
            value = [column[k] for k in sorted(column, key=int)]
            rest = parts[idx + 1 :]
            if rest:
                return [_walk_path(v, rest, where) for v in value]
            return value
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise ValueError(f"missing {part!r} at {where}")
    return value


def _row_content(row):
    try:
        return row["json"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return row.get("text", row)


def _column(entry, inputs, text):
    if entry["kind"] == "literal":
        if "path" in entry:
            with open(entry["path"], encoding="utf-8") as fh:
                values = json.load(fh)
        else:
            values = entry["values"]
        return [text(v) for v in values]
    with open(os.path.join(inputs[entry["node"]], "results.json")) as fh:
        envelope = json.load(fh)
    result = envelope.get("result", envelope)
    items = result.get("items")
    if items is None and isinstance(result.get("value"), dict):
        items = result["value"].get("items")
    if items is None and isinstance(result.get("text"), str):
        items = [{"output": result["text"]}]
    if not isinstance(items, list):
        raise ValueError(f"upstream {entry['node']!r} produced no items")
    path = entry.get("path")
    if path:
        parts = path.split(".")
        return [
            _walk_path(i, parts, f"{entry['node']} item {idx} path {path}")
            for idx, i in enumerate(items)
        ]
    if entry.get("flatten"):
        flat = []
        for i in items:
            if isinstance(i, dict) and "rows" in i:
                for row in i["rows"]:
                    flat.append(text(_row_content(row)))
            else:
                out = i["output"] if isinstance(i, dict) and "output" in i else i
                if isinstance(out, list):
                    flat.extend(text(v) for v in out)
                else:
                    flat.append(text(out))
        return flat
    return [
        text(i["output"]) if isinstance(i, dict) and "output" in i else text(i)
        for i in items
    ]


_SAFE_BUILTIN_BASE_NAMES = (
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

EXCEPTION_BUILTIN_NAMES = (
    "Exception",
    "ArithmeticError",
    "AssertionError",
    "AttributeError",
    "IndexError",
    "KeyError",
    "LookupError",
    "NotImplementedError",
    "OverflowError",
    "RuntimeError",
    "StopIteration",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
)

SAFE_BUILTIN_NAMES = _SAFE_BUILTIN_BASE_NAMES + EXCEPTION_BUILTIN_NAMES
NAMESPACE_GLOBAL_NAMES = frozenset(("json", "re", "math", "Message", "ops"))
_SOURCE_NAME = "<lambda_op>"


def _lambda_import(
    name: str,
    globals: Any = None,
    locals: Any = None,
    fromlist: Any = (),
    level: int = 0,
) -> Any:
    if level != 0:
        raise ImportError("relative imports are not available in a LambdaOp function")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _imported_names(module: ast.AST) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            found.extend((alias.name, 0) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append((node.module or "", node.level))
    return found


def _unbound_global_names(code: str) -> list[str]:
    table = symtable.symtable(code.strip(), _SOURCE_NAME, "exec")
    module_symbols = {s.get_name() for s in table.get_symbols()}
    free: set[str] = set()
    for symbol in table.get_symbols():
        if (
            symbol.is_referenced()
            and not symbol.is_assigned()
            and not symbol.is_imported()
        ):
            free.add(symbol.get_name())

    def _walk(scope: symtable.SymbolTable) -> None:
        for child in scope.get_children():
            for symbol in child.get_symbols():
                if symbol.is_global() and symbol.get_name() not in module_symbols:
                    free.add(symbol.get_name())
            _walk(child)

    _walk(table)
    return sorted(free)


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


def _check_free_names(code: str) -> None:
    allowed = set(SAFE_BUILTIN_NAMES) | set(NAMESPACE_GLOBAL_NAMES)
    free = _unbound_global_names(code)
    allowed |= {"np", "pd", "__import__"}
    unavailable = sorted(name for name in free if name not in allowed)
    if unavailable:
        raise ValueError(
            f"it references {unavailable[0]!r}, which the LambdaOp namespace"
            " does not provide"
        )


def _validate_source(code: str, fn_name: str | None = None) -> str:
    tree = _parse(code)
    if isinstance(tree, ast.Expression):
        _check_free_names(code)
        body = tree.body
        if not isinstance(body, ast.Lambda):
            raise ValueError("LambdaOp code is not a lambda expression")
        if _param_count(body.args) != 1:
            raise ValueError("Function must accept exactly 1 parameter (a tuple)")
        return "<lambda>"
    for name, level in _imported_names(tree):
        if level != 0:
            raise ValueError("LambdaOp code must not use relative imports")
    _check_free_names(code)
    defs = [
        stmt
        for stmt in tree.body
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    target = (
        defs[0]
        if fn_name is None
        else next((d for d in defs if d.name == fn_name), None)
    )
    if target is None:
        defined = ", ".join(repr(d.name) for d in defs) or "none"
        raise ValueError(
            f"LambdaOp code defines no top-level function named {fn_name!r}"
            f" (fn_name); top-level functions defined: {defined}"
        )
    if isinstance(target, ast.AsyncFunctionDef):
        raise ValueError("LambdaOp function must not be async")
    if _param_count(target.args) != 1:
        raise ValueError("Function must accept exactly 1 parameter (a tuple)")
    return target.name


def _materialize(code: str, fn_name: str | None = None) -> Callable:
    defined_name = _validate_source(code, fn_name)
    src = code.strip()
    used = {node.id for node in ast.walk(ast.parse(src)) if isinstance(node, ast.Name)}
    namespace: dict[str, Any] = {
        "__name__": "lambda_op",
        "__builtins__": {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES},
        "json": json,
        "re": re,
        "math": math,
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


def materialize(code, fn_name):
    return _materialize(code, fn_name)
