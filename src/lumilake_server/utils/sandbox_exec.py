"""Run a LambdaOp's user code outside the server process.

A LambdaOp carries caller-supplied Python. The server evaluates it at build time
in two places — a LambdaOp whose inputs are all literal is folded into a
constant, and an API-mode LLMChatOp renders a function step into its request
body — and until this module it did so with ``exec`` in its own process. The
restricted-builtins namespace was the only guard, and it is escapable (the
object graph reaches every class without a single builtin), so any caller could
run arbitrary code with the server's credentials and memory.

Now every evaluation happens in a child interpreter:

- ``python -I -B``: isolated mode, so no ``PYTHON*`` variable or user site
  applies, and no bytecode is written.
- An empty environment apart from ``PATH``/``LANG`` and single-threaded BLAS,
  and a fresh temporary working directory removed afterwards. Nothing the
  server holds in its environment is passed down.
- Resource limits set by the child on itself before it reads the user code:
  address space (``memory_mb``), CPU seconds, open files, file size. The child
  also sets ``PR_SET_NO_NEW_PRIVS``.
- A wall-clock timeout that kills the child's whole process group.
- The server marks itself non-dumpable (``PR_SET_DUMPABLE=0``) before the first
  child starts, so a child that escapes the namespace cannot read the server's
  ``/proc/<pid>/environ`` or ``/proc/<pid>/mem`` even though it runs as the
  same uid.
- Inside the child the function sees the same restricted namespace as before
  (``SAFE_BUILTINS`` plus ``json``/``re``/``math``/``np``/``pd``), so valid code
  behaves exactly as it did.

One child runs a whole batch of rows. Input and output travel as JSON over
stdin/stdout; each row's result is ``str(fn(row))``, the LambdaOp contract.

What this does NOT give: a separate uid, network isolation or a filesystem
view. A child that escapes the namespace can still read files the server's uid
can read and open network connections. Full isolation is the FlowMesh
``python`` task (a per-task container with no network and uid 65534), which a
standalone LambdaOp already uses — see ``runtime/python_step.py``.
"""

from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from typing import Any

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MEMORY_MB = 1024
MAX_TIMEOUT_S = 600.0
MAX_MEMORY_MB = 8192
MIN_MEMORY_MB = 128

_OPEN_FILES_LIMIT = 64
_FILE_SIZE_LIMIT = 1 << 20

# The namespace the user function sees. Names only: the child resolves them in
# its own interpreter. Kept identical to utils/func_serialization.SAFE_BUILTINS.
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

# Runs in the child. Limits are applied before the payload is read, so the user
# code never runs without them. numpy/pandas are imported only when the code
# names them: they cost most of the child's start-up time and address space.
_RUNNER = r"""
import json, resource, sys, traceback

def _limit(which, value):
    try:
        resource.setrlimit(which, (value, value))
    except (ValueError, OSError):
        pass

_cfg = json.loads(sys.argv[1])
_limit(resource.RLIMIT_AS, _cfg["memory_bytes"])
_limit(resource.RLIMIT_CPU, _cfg["cpu_seconds"])
_limit(resource.RLIMIT_NOFILE, _cfg["open_files"])
_limit(resource.RLIMIT_FSIZE, _cfg["file_size"])
_limit(resource.RLIMIT_CORE, 0)
try:
    import ctypes
    ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0)  # PR_SET_NO_NEW_PRIVS
except Exception:
    pass

def _emit(obj):
    sys.stdout.write(json.dumps(obj))
    sys.stdout.flush()

try:
    _payload = json.load(sys.stdin)
    import builtins, math, re
    _code = _payload["code"]
    _globals = {
        "__builtins__": {n: getattr(builtins, n) for n in _payload["builtins"]},
        "json": json, "re": re, "math": math,
    }
    if "np" in _payload["modules"]:
        import numpy
        _globals["np"] = numpy
    if "pd" in _payload["modules"]:
        import pandas
        _globals["pd"] = pandas
    # SDK-serialized code may name these in annotations, which run at def time.
    import types
    _globals["Message"] = dict
    _globals["ops"] = types.SimpleNamespace(
        SingleDtype=str | list[dict[str, str]], Message=dict
    )
    _globals.update(_payload.get("extra_globals") or {})
    _src = _code.strip()
    if _src.startswith("lambda"):
        _fn = eval(compile(_src, "<lambda_op>", "eval"), _globals, {})
    else:
        _locals = {}
        exec(compile(_src, "<lambda_op>", "exec"), _globals, _locals)
        _fn = _locals[_payload["fn_name"]]
    _results = [str(_fn(tuple(row))) for row in _payload["rows"]]
except BaseException:
    _emit({"error": traceback.format_exc(limit=-8)})
    sys.exit(0)
_emit({"results": _results})
"""

_parent_hardened = False


class SandboxError(RuntimeError):
    """User code failed, timed out, or exceeded a limit in the sandbox."""


def _harden_parent() -> None:
    """Make the server non-dumpable once, so a same-uid child cannot read its
    environment or memory through /proc. Linux only; best effort elsewhere."""
    global _parent_hardened
    if _parent_hardened:
        return
    _parent_hardened = True
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        ctypes.CDLL(None, use_errno=True).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE
    except Exception:  # pragma: no cover - platform dependent
        pass


def _param_count(args: ast.arguments) -> int:
    count = len(args.posonlyargs) + len(args.args) + len(args.kwonlyargs)
    return count + (1 if args.vararg else 0) + (1 if args.kwarg else 0)


def validate_lambda_source(code: str) -> str:
    """Check LambdaOp source without running any of it; return the function name.

    Parsing only: ``exec`` of a ``def`` already runs its decorators, default
    values and annotations, and a module body can hold arbitrary statements, so
    validation must never execute. Enforces what the in-process materializer
    enforced — a lambda, or source whose first binding is a function, taking
    exactly one parameter.
    """
    src = code.strip()
    try:
        if src.startswith("lambda"):
            node = ast.parse(src, mode="eval").body
            if not isinstance(node, ast.Lambda):
                raise SandboxError("LambdaOp code is not a lambda expression")
            if _param_count(node.args) != 1:
                raise SandboxError("Function must accept exactly 1 parameter (a tuple)")
            return "<lambda>"
        module = ast.parse(src, mode="exec")
    except SyntaxError as exc:
        raise SandboxError(f"LambdaOp code does not parse: {exc}") from exc
    for stmt in module.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if isinstance(stmt, ast.AsyncFunctionDef):
                raise SandboxError("LambdaOp function must not be async")
            if _param_count(stmt.args) != 1:
                raise SandboxError(
                    "Function must accept exactly 1 parameter (a tuple), but has"
                    f" {_param_count(stmt.args)} parameters. Expected signature:"
                    " fn(args: tuple[str, ...]) -> str"
                )
            return stmt.name
        if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.ClassDef, ast.Import)):
            break
        if isinstance(stmt, ast.ImportFrom):
            break
    raise SandboxError("LambdaOp code must define a function as its first binding")


def resolve_limits(
    timeout_s: float | None, memory_mb: int | None
) -> tuple[float, int]:
    """Defaults and bounds for a LambdaOp's ``timeout_s``/``memory_mb``."""
    timeout = DEFAULT_TIMEOUT_S if timeout_s is None else float(timeout_s)
    memory = DEFAULT_MEMORY_MB if memory_mb is None else int(memory_mb)
    if not 0 < timeout <= MAX_TIMEOUT_S:
        raise ValueError(f"timeout_s must be in (0, {MAX_TIMEOUT_S:g}], got {timeout_s}")
    if not MIN_MEMORY_MB <= memory <= MAX_MEMORY_MB:
        raise ValueError(
            f"memory_mb must be in [{MIN_MEMORY_MB}, {MAX_MEMORY_MB}], got {memory_mb}"
        )
    return timeout, memory


def _modules_named(code: str) -> list[str]:
    names = {
        node.id
        for node in ast.walk(ast.parse(code.strip()))
        if isinstance(node, ast.Name)
    }
    return [m for m in ("np", "pd") if m in names]


def run_lambda(
    code: str,
    rows: Sequence[Sequence[Any]],
    *,
    timeout_s: float | None = None,
    memory_mb: int | None = None,
    extra_globals: dict[str, Any] | None = None,
) -> list[str]:
    """Evaluate ``str(fn(tuple(row)))`` for every row in one sandboxed child.

    ``extra_globals`` must be JSON-serializable: it is recreated in the child,
    never shared. Raises ``SandboxError`` with the user's traceback on an
    exception, and on a timeout, a limit kill or malformed output.
    """
    fn_name = validate_lambda_source(code)
    timeout, memory = resolve_limits(timeout_s, memory_mb)
    if not rows:
        return []
    _harden_parent()

    cfg = {
        "memory_bytes": memory * 1024 * 1024,
        "cpu_seconds": int(timeout) + 1,
        "open_files": _OPEN_FILES_LIMIT,
        "file_size": _FILE_SIZE_LIMIT,
    }
    payload = json.dumps(
        {
            "code": code,
            "fn_name": fn_name,
            "rows": [list(row) for row in rows],
            "builtins": list(SAFE_BUILTIN_NAMES),
            "modules": _modules_named(code),
            "extra_globals": extra_globals or {},
        }
    )
    env = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    with tempfile.TemporaryDirectory(prefix="lumilake-lambda-") as workdir:
        proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-I", "-B", "-c", _RUNNER, json.dumps(cfg)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=workdir,
            env=env,
            start_new_session=True,
            close_fds=True,
        )
        try:
            out, err = proc.communicate(payload.encode(), timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            proc.communicate()
            raise SandboxError(
                f"LambdaOp code exceeded its {timeout:g}s time limit"
            ) from None
        finally:
            if proc.poll() is None:  # pragma: no cover - defensive
                _kill_group(proc)
                proc.wait()

    return _parse_result(out, err, proc.returncode, memory, len(rows))


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def _parse_result(
    out: bytes, err: bytes, returncode: int, memory_mb: int, n_rows: int
) -> list[str]:
    try:
        result = json.loads(out.decode() or "null")
    except (UnicodeDecodeError, json.JSONDecodeError):
        result = None
    if isinstance(result, dict) and "error" in result:
        message = str(result["error"]).rstrip()
        if "MemoryError" in message:
            message += f"\n(LambdaOp memory limit: {memory_mb} MiB)"
        raise SandboxError(f"LambdaOp code failed:\n{message}")
    if isinstance(result, dict) and isinstance(result.get("results"), list):
        results = result["results"]
        if len(results) == n_rows and all(isinstance(r, str) for r in results):
            return results
    tail = err.decode(errors="replace").strip()[-2000:]
    if returncode < 0:
        sig = signal.Signals(-returncode).name
        reason = {
            "SIGXCPU": "exceeded its CPU time limit",
            "SIGKILL": f"was killed (memory limit {memory_mb} MiB or time limit)",
            "SIGXFSZ": "exceeded its file size limit",
        }.get(sig, f"was terminated by {sig}")
        raise SandboxError(f"LambdaOp code {reason}" + (f":\n{tail}" if tail else ""))
    if "MemoryError" in tail:
        raise SandboxError(f"LambdaOp code exceeded its {memory_mb} MiB memory limit")
    raise SandboxError(
        f"LambdaOp sandbox exited {returncode} without a result"
        + (f":\n{tail}" if tail else "")
    )


class SandboxedFunction:
    """A LambdaOp function that only ever runs in the sandbox.

    Stands in for the callable a deserialized LambdaOp used to hold: calling it
    evaluates one row in a child process. ``map`` evaluates many rows in one.
    """

    def __init__(
        self,
        code: str,
        fn_name: str | None = None,
        *,
        timeout_s: float | None = None,
        memory_mb: int | None = None,
    ) -> None:
        detected = validate_lambda_source(code)
        resolve_limits(timeout_s, memory_mb)
        self.code = code
        self.__name__ = fn_name or detected
        self.timeout_s = timeout_s
        self.memory_mb = memory_mb

    def map(self, rows: Sequence[Sequence[Any]]) -> list[str]:
        return run_lambda(
            self.code, rows, timeout_s=self.timeout_s, memory_mb=self.memory_mb
        )

    def __call__(self, args: Sequence[Any]) -> str:
        return self.map([args])[0]

    def __repr__(self) -> str:
        return f"SandboxedFunction({self.__name__!r})"
