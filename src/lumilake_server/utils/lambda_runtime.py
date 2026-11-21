"""The LambdaOp function contract: source validation and the function namespace.

A LambdaOp function takes one argument, its row as a tuple of decoded values,
and returns a string; a non-string return is archived as its JSON text. JSON
object or array text arrives as the object or list it holds, and a serialized
table as its rows. The function is written as a lambda, or as module source
that defines a one-parameter ``def`` named by ``fn_name``, optionally after
imports, constants and helpers. It sees a fixed namespace: a small set of
builtins plus ``json``, ``re``, ``math``, ``np`` and ``pd``
(``np``/``pd`` imported only when the code names them), and it may ``import``
any module the container image provides.

The namespace fixes what names resolve; it is not a security boundary, because
the object graph reaches every class without a single builtin. Caller-supplied
code is therefore never materialized in the Lumilake server process: it runs
only where the deployment isolates it, as a FlowMesh ``python`` task
(``runtime/python_step.py``) in a per-task container with no network and an
unprivileged uid. That container, not this namespace, is what makes allowing
imports safe.

A LambdaOp that feeds an LLM is different: it is inlined into the LLM's FlowMesh
task as a ``graph_template`` function and evaluated by the FlowMesh worker's own
``safe_eval`` namespace, which has no ``__import__`` and calls the first name the
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
import symtable
import types
from collections.abc import Callable
from typing import Any

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

# Built-in exception classes user code reasonably raises or catches.
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

# Names the standalone namespace exposes but FlowMesh's inline ``safe_eval``
# namespace does not (its SAFE_BUILTINS has no exception classes), so inline
# code that references them is rejected at submit.
INLINE_UNAVAILABLE_BUILTIN_NAMES = EXCEPTION_BUILTIN_NAMES

# The standalone namespace binds ``Message`` and ``ops`` so an SDK callable's
# annotations resolve there; the inline namespace lacks both.
INLINE_UNAVAILABLE_ANNOTATION_NAMES = ("Message", "ops")

# Names the standalone LambdaOp namespace provides beyond the safe builtins.
# ``np``/``pd`` are provided only when the code names them, so they are checked
# separately; the rest are always present.
NAMESPACE_GLOBAL_NAMES = frozenset(("json", "re", "math", "Message", "ops"))

_SOURCE_NAME = "<lambda_op>"

# The same decode as ``lambda_task_helpers``, in the form FlowMesh's inline namespace
# can evaluate; a test pins the two decoders to agree.
DECODER_SOURCE = """\
def _serialized_table(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except:
            return None
    if (
        isinstance(value, dict)
        and list(value) == ["df"]
        and isinstance(value["df"], str)
    ):
        try:
            table = json.loads(value["df"])
        except:
            return None
        if isinstance(table, dict):
            return table
    return None


def _table_rows(table):
    rows = {}
    for name, column in table.items():
        if isinstance(column, dict):
            for index, value in column.items():
                rows.setdefault(index, {})[name] = value
    try:
        indexes = sorted(rows, key=int)
    except:
        indexes = list(rows)
    return [rows[index] for index in indexes]


def _decode(value):
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except:
            return value
        if isinstance(decoded, (dict, list)):
            return _decode(decoded)
        return value
    if isinstance(value, dict):
        table = _serialized_table(value)
        if table is None and "table" in value:
            table = _serialized_table(value["table"])
        if table is not None:
            return _table_rows(table)
    return value
"""


def _top_level_bindings(code: str) -> set[str]:
    """Every name the code's own top-level scope binds."""
    table = symtable.symtable(code.strip(), _SOURCE_NAME, "exec")
    return {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_assigned() or symbol.is_imported()
    }


def inline_lambda_source(code: str, fn_name: str) -> str:
    """The function step source a row-mode LambdaOp inlines into an LLM."""
    src = code.strip()
    tree = _parse(src)
    if isinstance(tree, ast.Expression):
        src = f"_op = (\n{src}\n)"
        globals_ = {"_op"}
        call = "_op(row)"
    else:
        globals_ = _top_level_bindings(src)
        call = f"{fn_name}(row)"
    globals_ |= {"_decode", "_serialized_table", "_table_rows"}
    return (
        "global " + ", ".join(sorted(globals_)) + "\n"
        "def adapt(args):\n"
        "    row = tuple(_decode(v) for v in args)\n"
        f"    value = {call}\n"
        "    return value if isinstance(value, str) else json.dumps(value)\n"
        "\n\n" + DECODER_SOURCE + "\n\n" + src + "\n"
    )


def callable_source(fn: Callable[..., Any], code: str) -> str:
    """The function source behind an SDK callable, or ``""`` when unreadable."""
    stripped = code.strip()
    if fn.__name__ != "<lambda>":
        return stripped
    try:
        tree = ast.parse(stripped, mode="exec")
    except SyntaxError:
        return ""
    lambdas = [node for node in ast.walk(tree) if isinstance(node, ast.Lambda)]
    if len(lambdas) != 1:
        return ""
    return ast.get_source_segment(stripped, lambdas[0]) or ""


def _lambda_import(
    name: str,
    globals: Any = None,
    locals: Any = None,
    fromlist: Any = (),
    level: int = 0,
) -> Any:
    """``__import__`` for standalone LambdaOp code: any module the image provides.

    CPython resolves ``__import__`` in the calling frame's builtins, which are
    restricted here, so both the caller's own ``import`` statements and the
    lazy imports numpy/pandas make from C land in this hook. A standalone task
    runs in its own container, so the image is the only import boundary.
    """
    if level != 0:
        raise ImportError("relative imports are not available in a LambdaOp function")
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


def _unbound_global_names(code: str) -> list[str]:
    """Names the code references but never binds, imports, or provides itself.

    The standalone LambdaOp namespace supplies only ``SAFE_BUILTIN_NAMES`` plus
    ``json``/``re``/``math``/``Message``/``ops`` (and ``np``/``pd`` when named).
    Any other free name would raise ``NameError`` at run time on the worker, so
    it is rejected at submit. A name referenced at module scope without being
    bound or imported is free, and a name referenced in any nested scope that
    is not bound anywhere in the module is free too.
    """
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


def _check_lambda(tree: ast.Expression) -> str:
    if not isinstance(tree.body, ast.Lambda):
        raise ValueError("LambdaOp code is not a lambda expression")
    if _param_count(tree.body.args) != 1:
        raise ValueError("Function must accept exactly 1 parameter (a tuple)")
    return "<lambda>"


def _check_def(stmt: ast.FunctionDef, max_params: int = 1) -> str:
    count = _param_count(stmt.args)
    if count <= max_params:
        return stmt.name
    if max_params == 1:
        raise ValueError(
            "Function must accept exactly 1 parameter (a tuple), but has"
            f" {count} parameters. Expected signature:"
            " fn(args: tuple[str, ...]) -> str"
        )
    raise ValueError(
        f"Function must accept at most {max_params} parameters, but has"
        f" {count}. Expected signature: fn(args: tuple[str, ...], seed: int)"
    )


def _check_free_names(code: str) -> None:
    """Reject free names the standalone namespace does not provide.

    The namespace supplies ``SAFE_BUILTIN_NAMES``, ``json``/``re``/``math``/
    ``Message``/``ops``, and ``np``/``pd`` when the code names them. Any other
    name the code references without binding, importing, or defining would
    raise ``NameError`` on the worker, so reject it at submit with the name.
    """
    allowed = set(SAFE_BUILTIN_NAMES) | set(NAMESPACE_GLOBAL_NAMES)
    free = _unbound_global_names(code)
    # ``np``/``pd`` are provided lazily only when the code names them, so they
    # are always allowed; ``__import__`` is bound to the restricted import hook.
    allowed |= {"np", "pd", "__import__"}
    unavailable = sorted(name for name in free if name not in allowed)
    if unavailable:
        raise ValueError(
            f"it references {unavailable[0]!r}, which the LambdaOp namespace"
            " does not provide. Only the safe builtins, json, re, math, np, pd"
            " and Message are available; bind or import the name, or define it"
            " as a helper"
        )


def validate_source(code: str, fn_name: str | None = None, max_params: int = 1) -> str:
    """Check LambdaOp source without running any of it; return the function name.

    Parsing only: a ``def`` runs its decorators, defaults and annotations when
    executed, and a module body can hold arbitrary statements, so validation
    never executes the code.

    ``fn_name`` is the name the LambdaOp declares: the code must define a
    top-level one-parameter function of that name. Imports, constants and
    helper functions may come before it. Without ``fn_name`` the first
    top-level function is used. A lambda has no name, so ``fn_name`` labels it
    only. Imports are not restricted here: a standalone task runs in its own
    container, so the image is the only import boundary.
    """
    tree = _parse(code)
    if isinstance(tree, ast.Expression):
        _check_free_names(code)
        return _check_lambda(tree)
    for name, level in _imported_names(tree):
        if level != 0:
            raise ValueError("LambdaOp code must not use relative imports")
    _check_free_names(code)
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
    return _check_def(target, max_params)


def validate_inline_source(code: str, fn_name: str | None = None) -> str:
    """Check source for a LambdaOp the FlowMesh worker evaluates inline."""
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
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    unavailable = sorted(
        used.intersection(INLINE_UNAVAILABLE_BUILTIN_NAMES),
        key=INLINE_UNAVAILABLE_BUILTIN_NAMES.index,
    )
    if unavailable:
        raise ValueError(
            f"it references {unavailable[0]!r}, but a LambdaOp that feeds an"
            " LLM is evaluated inline by the FlowMesh worker, where that name"
            " is not defined. Use the standalone LambdaOp form (a workflow"
            " output or feeding only other LambdaOps) to raise or catch"
            " exceptions"
        )
    unbound = set(_unbound_global_names(code))
    annotated = next(
        (n for n in INLINE_UNAVAILABLE_ANNOTATION_NAMES if n in unbound), None
    )
    if annotated is not None:
        raise ValueError(
            f"it references {annotated!r}, which a LambdaOp that feeds an LLM cannot"
            " use inline: the FlowMesh worker's namespace has no Message or ops, and a"
            " def evaluates its annotations there. Annotate with builtin types, or move"
            " this logic to a LambdaOp that runs as its own python task"
        )
    if any(isinstance(stmt, ast.AnnAssign) for stmt in tree.body):
        raise ValueError(
            "it annotates a top-level assignment, which a LambdaOp that feeds an LLM"
            " cannot use inline: the generated wrapper runs the code as a module, whose"
            " annotations land in the namespace before its function does. Assign"
            " without an annotation, or move this logic to a LambdaOp that runs as its"
            " own python task"
        )
    top_level = _top_level_bindings(code)
    collisions = sorted(top_level & _INLINE_WRAPPER_NAMES)
    if collisions:
        raise ValueError(
            f"it binds {collisions[0]!r}, which a LambdaOp that feeds an LLM cannot use"
            " inline: its function step embeds a generated wrapper that already binds"
            " that name. Rename the binding or move this logic to a LambdaOp that runs"
            " as its own python task"
        )
    return name


def _inline_names() -> frozenset[str]:
    """The names the inline wrapper binds around the op's code."""
    tree = ast.parse(DECODER_SOURCE)
    return frozenset(
        (
            "adapt",
            "_op",
            *(f.name for f in tree.body if isinstance(f, ast.FunctionDef)),
        )
    )


_INLINE_WRAPPER_NAMES = _inline_names()


def materialize(
    code: str, fn_name: str | None = None, max_params: int = 1
) -> Callable[..., Any]:
    """Build the function from validated source in the LambdaOp namespace.

    ``fn_name`` and ``max_params`` are checked against the source as in
    ``validate_source``; a seeded LambdaOp declares two.
    """
    defined_name = validate_source(code, fn_name, max_params)
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
