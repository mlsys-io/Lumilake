"""
Safe function evaluation module.

Provides secure execution of user-defined functions in a restricted environment.
Functions are materialized (compiled) and executed with limited builtins and module
access to prevent malicious code execution while supporting common data operations.

Security model:
- Two-phase execution: materialize (compile) then execute (run)
- Restricted builtins: only safe operations (no open, eval, exec, import, etc.)
- Limited module access: json, re, math, numpy, pandas, pyarrow
- Type validation: enforces Callable[[tuple[str, ...]], str] signature
- Isolated execution: exec() with explicit safe_globals/safe_locals

Typical usage:
    fn_obj = safe_materialize_function("lambda args: args[0].upper()")
    result = safe_execute_function(fn_obj, ("hello",))  # Returns "HELLO"
"""

import inspect
import json
import math
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

# Whitelist of safe built-in functions and types available during function execution.
# Excludes dangerous operations like open, eval, exec, compile, __import__, etc.
SAFE_BUILTINS = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "len": len,
    "sum": sum,
    "max": max,
    "min": min,
    "abs": abs,
    "round": round,
    "sorted": sorted,
    "reversed": reversed,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "any": any,
    "all": all,
    "range": range,
    "isinstance": isinstance,
}

# Whitelist of safe modules available during function execution.
# Provides common data manipulation and serialization tools without system access.
SAFE_MODULES = {
    "json": json,
    "re": re,
    "math": math,
    "np": np,
    "pd": pd,
}


def safe_materialize_function(
    fn_code: str, extra_globals: dict[str, object] | None = None
) -> Callable:
    """
    Compile function source code into a callable object with restricted builtins.

    Two-phase security model: materialize (this function) creates the function object,
    then safe_execute_function runs it in an isolated environment. This separation
    enables signature validation, function caching, and additional security checks
    before execution.

    Supported function formats:
    - Lambda expressions: "lambda args: args[0].upper()"
    - Named functions: "def transform(args):\\n    return args[0].upper()"

    Type signature enforcement:
    - Must accept exactly 1 parameter (tuple of strings or list of messages)
    - Should return a string (validated at execution time)
    - Signature: Callable[[tuple[str, ...]], str]

    Args:
        fn_code: Python source code defining a function or lambda expression
        extra_globals: Additional safe globals to expose during materialization

    Returns:
        Compiled function object with restricted builtin access

    Raises:
        RuntimeError: If compilation fails or signature doesn't match required format

    Examples:
        >>> fn = safe_materialize_function("lambda args: args[0].upper()")
        >>> fn = safe_materialize_function("def f(args):\\n    return json.dumps(args)")
    """
    safe_globals = {
        "__builtins__": SAFE_BUILTINS,
        **SAFE_MODULES,
    }
    if extra_globals:
        safe_globals.update(extra_globals)

    fn_code_stripped = fn_code.strip()

    # Case 1: Lambda expression (use eval - lambdas are expressions)
    if fn_code_stripped.startswith("lambda"):
        try:
            fn_obj = eval(fn_code_stripped, safe_globals, {})
            if not callable(fn_obj):
                raise RuntimeError("Lambda expression did not produce a callable")
        except Exception as e:
            raise RuntimeError(
                f"Lambda compilation failed: {e}\nCode: {fn_code}"
            ) from e

    # Case 2: Function definition (use exec - def is a statement)
    else:
        safe_locals: dict[str, Any] = {}

        # Execute the function definition (creates function object in locals)
        try:
            exec(fn_code_stripped, safe_globals, safe_locals)
        except Exception as e:
            raise RuntimeError(
                f"Function definition failed: {e}\nCode: {fn_code}"
            ) from e

        # Find the function object
        if not safe_locals:
            raise RuntimeError("Function definition did not create any objects")

        # Get the function (usually the first/only item in locals)
        fn_name = list(safe_locals.keys())[0]
        fn_obj = safe_locals[fn_name]

        if not callable(fn_obj):
            raise RuntimeError(f"Defined object '{fn_name}' is not callable")

    # Validate function signature: should accept a single tuple parameter
    try:
        sig = inspect.signature(fn_obj)
        params = list(sig.parameters.values())

        # Check that function takes exactly 1 parameter (the tuple)
        if len(params) != 1:
            raise RuntimeError(
                "Function must accept exactly 1 parameter (a tuple), but has"
                f" {len(params)} parameters. Expected signature: fn(args: tuple[str,"
                " ...]) -> str"
            )

        # Note: We can't easily validate that it returns a string at compile time,
        # but we'll validate at execution time

    except ValueError:
        # Some builtins don't have inspectable signatures, allow them
        pass

    return fn_obj
