import re
from collections.abc import Callable, Sequence
from typing import Any, Literal, overload

import dill

from lumilake_server.ops.data_ops import DataOp
from lumilake_server.ops.ops import FunctionalOp, Op, SingleDtype
from lumilake_server.utils.lambda_runtime import validate_source

MAX_TIMEOUT_S = 600.0
MIN_MEMORY_MB = 128
MAX_MEMORY_MB = 8192

# A row-mode LambdaOp receives one tuple of scalar values per row and returns
# a single scalar. A list-mode LambdaOp receives each input as its whole list
# of JSON values and returns a list of JSON values (one output item per
# element). The overloads below let ``mode`` select the callable contract so a
# list-mode callable needs no ``type: ignore``.
RowLambdaFn = Callable[[tuple[SingleDtype, ...]], str]
ListLambdaFn = Callable[[tuple[list[Any], ...]], list[Any]]


def _validate_limits(timeout_s: float | None, memory_mb: int | None) -> None:
    if timeout_s is not None:
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise ValueError(f"timeout_s must be a number, got {timeout_s!r}")
        if not 0 < timeout_s <= MAX_TIMEOUT_S:
            raise ValueError(
                f"timeout_s must be in (0, {MAX_TIMEOUT_S:g}], got {timeout_s}"
            )
    if memory_mb is not None:
        if isinstance(memory_mb, bool) or not isinstance(memory_mb, int):
            raise ValueError(f"memory_mb must be an integer, got {memory_mb!r}")
        if not MIN_MEMORY_MB <= memory_mb <= MAX_MEMORY_MB:
            raise ValueError(
                f"memory_mb must be in [{MIN_MEMORY_MB}, {MAX_MEMORY_MB}],"
                f" got {memory_mb}"
            )


class SubmittedFunction:
    """Stand-in for a LambdaOp function that arrived as source text.

    A submitted graph carries caller-supplied code, which the server never
    executes in its own process: this holds the parse-validated source and
    refuses to be called. The code runs only as a FlowMesh ``python`` task
    (``runtime/python_step.py``).
    """

    def __init__(self, code: str, fn_name: str) -> None:
        validate_source(code, fn_name)
        self.code = code
        self.__name__ = fn_name

    def __call__(self, args: tuple[SingleDtype, ...]) -> str:
        raise RuntimeError(
            f"LambdaOp function '{self.__name__}' was submitted as source; the"
            " server does not execute submitted code"
        )


@Op.registry.register("FormatOp")
class FormatOp(FunctionalOp):
    template: str

    def __init__(
        self, template: str, *args: list[str] | Op, **kwargs: list[str] | Op
    ) -> None:
        format_args: list[int] = []
        format_kwargs: dict[str, int] = {}
        inputs: list[Op] = []

        for arg in args:
            arg = arg if isinstance(arg, Op) else DataOp(arg)
            try:
                i = inputs.index(arg)
                format_args.append(i)
            except ValueError:
                format_args.append(len(inputs))
                inputs.append(arg)

        for k, v in kwargs.items():
            v = v if isinstance(v, Op) else DataOp(v)
            try:
                i = inputs.index(v)
                format_kwargs[k] = i
            except ValueError:
                format_kwargs[k] = len(inputs)
                inputs.append(v)

        super().__init__(inputs)
        self.template = template
        self._format_args = format_args
        self._format_kwargs = format_kwargs

    @property
    def format_args(self) -> list[Op]:
        return [self.inputs[i] for i in self._format_args]

    @property
    def format_kwargs(self) -> dict[str, Op]:
        return {k: self.inputs[i] for k, i in self._format_kwargs.items()}

    def _serialize(self) -> dict[str, Any]:
        format_args = [arg.id for arg in self.format_args]
        format_kwargs = {k: v.id for k, v in self.format_kwargs.items()}
        return dict(
            template=self.template, format_args=format_args, format_kwargs=format_kwargs
        )

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "FormatOp":
        format_args = [other_ops[arg] for arg in data["format_args"]]
        format_kwargs = {k: other_ops[v] for k, v in data["format_kwargs"].items()}
        return cls(data["template"], *format_args, **format_kwargs)


def format_op(template: str, *args: list[str] | Op, **kwargs: list[str] | Op) -> Op:
    return FormatOp(template, *args, **kwargs)


@Op.registry.register("LambdaOp")
class LambdaOp(Op):
    fn: RowLambdaFn | ListLambdaFn

    LAMBDA_MODES = ("row", "list")

    @overload
    def __init__(
        self,
        inputs: Sequence[list[str] | Op],
        fn: RowLambdaFn,
        code: str | None = None,
        mode: Literal["row"] = "row",
        timeout_s: float | None = None,
        memory_mb: int | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        inputs: Sequence[list[str] | Op],
        fn: ListLambdaFn,
        code: str | None = None,
        mode: Literal["list"] = "list",
        timeout_s: float | None = None,
        memory_mb: int | None = None,
    ) -> None: ...

    def __init__(
        self,
        inputs: Sequence[list[str] | Op],
        fn: RowLambdaFn | ListLambdaFn,
        code: str | None = None,
        mode: str = "row",
        timeout_s: float | None = None,
        memory_mb: int | None = None,
    ) -> None:
        if mode not in self.LAMBDA_MODES:
            raise ValueError(
                f"LambdaOp mode must be one of {self.LAMBDA_MODES} (got {mode!r})"
            )
        _validate_limits(timeout_s, memory_mb)
        input_ops: list[Op] = []
        for inp in inputs:
            if isinstance(inp, Op):
                input_ops.append(inp)
            else:
                input_ops.append(DataOp(inp))
        super().__init__(input_ops)
        self.fn = fn
        self.mode = mode
        # Limits for the FlowMesh python task a standalone LambdaOp compiles to.
        self.timeout_s = timeout_s
        self.memory_mb = memory_mb
        if code:
            self.code: str = code
        else:
            fn_serialized = (
                dill.source.getsource(fn)
                .replace("ops.SingleDtype", "str | list[dict[str, str]]")
                .replace("Message", "dict[str, str]")
                .replace(".role", "['role']")
                .replace(".content", "['content']")
            )
            if self.fn.__closure__:
                closure_vars = {}
                if self.fn.__code__.co_freevars:
                    for var_name, cell in zip(
                        self.fn.__code__.co_freevars, self.fn.__closure__
                    ):
                        try:
                            closure_vars[var_name] = cell.cell_contents
                        except ValueError:
                            pass
                for var_name, var_value in closure_vars.items():
                    pattern = r"\b" + re.escape(var_name) + r"\b"
                    # Callable replacement: re.sub would otherwise interpret backslashes
                    # in repr() as backrefs.
                    replacement: str = repr(var_value)

                    def _replace(_m: "re.Match[str]", r: str = replacement) -> str:
                        return r

                    fn_serialized = re.sub(pattern, _replace, fn_serialized)
            self.code = fn_serialized

    def _serialize(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "fn_name": self.fn.__name__,
            "_code": self.code,
            "_inputs": [inp.id for inp in self.inputs],
            "mode": self.mode,
        }
        if self.timeout_s is not None:
            data["timeout_s"] = self.timeout_s
        if self.memory_mb is not None:
            data["memory_mb"] = self.memory_mb
        return data

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "LambdaOp":
        fn_name = data.get("fn_name")
        code = data.get("_code")
        if not fn_name or not code:
            raise ValueError("LambdaOp serialization missing function code or name")

        timeout_s = data.get("timeout_s")
        memory_mb = data.get("memory_mb")
        try:
            fn = SubmittedFunction(code, fn_name)
        except ValueError as exc:
            raise ValueError(f"Invalid LambdaOp function '{fn_name}': {exc}") from exc

        input_ops = [other_ops[inp] for inp in data["_inputs"]]
        return cls(
            inputs=input_ops,
            fn=fn,
            code=code,
            mode=data.get("mode", "row"),
            timeout_s=timeout_s,
            memory_mb=memory_mb,
        )


def lambda_op(inputs: list[list[str] | Op], fn: RowLambdaFn) -> LambdaOp:
    return LambdaOp(inputs, fn)
