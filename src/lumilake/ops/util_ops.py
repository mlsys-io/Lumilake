import re
from collections.abc import Callable, Sequence
from typing import Any

import dill

from lumilake.common import Message
from lumilake.ops import ops
from lumilake.ops.data_ops import DataOp
from lumilake.ops.ops import FunctionalOp, Op, SingleDtype
from lumilake.utils.func_serialization import safe_materialize_function


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
    fn: Callable[[tuple[SingleDtype, ...]], str]

    def __init__(
        self,
        inputs: Sequence[list[str] | Op],
        fn: Callable[[tuple[SingleDtype, ...]], str],
        code: str | None = None,
    ) -> None:
        input_ops: list[Op] = []
        for inp in inputs:
            if isinstance(inp, Op):
                input_ops.append(inp)
            else:
                input_ops.append(DataOp(inp))
        super().__init__(input_ops)
        self.fn = fn
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
        return {
            "fn_name": self.fn.__name__,
            "_code": self.code,
            "_inputs": [inp.id for inp in self.inputs],
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "LambdaOp":
        fn_name = data.get("fn_name")
        code = data.get("_code")
        if not fn_name or not code:
            raise ValueError("LambdaOp serialization missing function code or name")

        extra_globals: dict[str, Any] = {
            "ops": ops,
            "Message": Message,
        }

        fn = safe_materialize_function(code, extra_globals)
        if not callable(fn):
            raise ValueError(f"Failed to deserialize LambdaOp function '{fn_name}'")

        input_ops = [other_ops[inp] for inp in data["_inputs"]]
        return cls(inputs=input_ops, fn=fn, code=code)


def lambda_op(
    inputs: list[list[str] | Op], fn: Callable[[tuple[SingleDtype, ...]], str]
) -> LambdaOp:
    return LambdaOp(inputs, fn)
