import re
from collections.abc import Callable, Sequence
from typing import Any, Literal, Self

import dill
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from lumilake_server.ops.data_ops import DataOp
from lumilake_server.ops.ops import FunctionalOp, Op, SingleDtype
from lumilake_server.ops.spec_validation import validate_fields
from lumilake_server.utils.lambda_runtime import validate_source

MAX_TIMEOUT_S = 600.0
# FlowMesh caps a python task's timeoutSeconds at 3600; a LambdaOp that runs in
# a custom container image may need longer than the default cap.
MAX_IMAGE_TIMEOUT_S = 3600.0
MIN_MEMORY_MB = 128
MAX_MEMORY_MB = 8192

LambdaMode = Literal["row", "list", "aligned", "expand", "reduce"]

RowLambdaFn = Callable[[tuple[SingleDtype, ...]], str]
ListLambdaFn = Callable[[tuple[list[Any], ...]], list[Any]]
ExpandLambdaFn = Callable[[tuple[Any, ...]], list[Any]]
ReduceLambdaFn = Callable[[list[tuple[Any, ...]]], Any]


class LambdaSpec(BaseModel):
    """The YAML `LambdaOp` entry fields, built by every source of the op."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    fn_name: StrictStr
    code: StrictStr = Field(alias="_code")
    mode: LambdaMode = "row"
    by: StrictStr | None = None
    timeout_s: StrictFloat | None = Field(default=None, gt=0)
    memory_mb: StrictInt | None = Field(
        default=None, ge=MIN_MEMORY_MB, le=MAX_MEMORY_MB
    )
    image: StrictStr | None = None
    expected_runtime_s: StrictFloat | None = Field(default=None, gt=0)
    cache: StrictBool | None = None

    @field_validator("image")
    @classmethod
    def _image_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must be a non-empty string")
        return value

    @model_validator(mode="after")
    def _by_needs_reduce(self) -> Self:
        if self.mode != "reduce":
            if self.by is not None:
                raise ValueError(
                    f"'by' is only valid in reduce mode (got mode {self.mode!r})"
                )
            return self
        parts = self.by.split(".") if self.by is not None else []
        if not parts or any(not part or part != part.strip() for part in parts):
            raise ValueError(
                "reduce mode requires 'by' to be a dotted path of non-empty segments"
                f" (got {self.by!r})"
            )
        return self

    @model_validator(mode="after")
    def _runtime_within_timeout(self) -> Self:
        cap = MAX_IMAGE_TIMEOUT_S if self.image else MAX_TIMEOUT_S
        if self.timeout_s is not None and self.timeout_s > cap:
            raise ValueError(f"timeout_s must be in (0, {cap:g}], got {self.timeout_s}")
        effective_timeout = (
            self.timeout_s if self.timeout_s is not None else MAX_TIMEOUT_S
        )
        if (
            self.expected_runtime_s is not None
            and self.expected_runtime_s > effective_timeout
        ):
            raise ValueError(
                f"expected_runtime_s must be in (0, {effective_timeout:g}], got"
                f" {self.expected_runtime_s}"
            )
        return self


def build_lambda_spec(
    fields: dict[str, Any], where: str, names: dict[str, str] | None = None
) -> LambdaSpec:
    """Validate one `LambdaOp` entry, naming the op and field in the error."""
    return validate_fields(LambdaSpec, fields, where, names)


def _serialize_fn(fn: Callable[..., Any]) -> str:
    """The source of a Python-authored Lambda, with its closure values inlined."""
    fn_serialized = (
        dill.source.getsource(fn)
        .replace("ops.SingleDtype", "str | list[dict[str, str]]")
        .replace("Message", "dict[str, str]")
        .replace(".role", "['role']")
        .replace(".content", "['content']")
    )
    if fn.__closure__:
        closure_vars = {}
        if fn.__code__.co_freevars:
            for var_name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
                try:
                    closure_vars[var_name] = cell.cell_contents
                except ValueError:
                    pass
        for var_name, var_value in closure_vars.items():
            pattern = r"\b" + re.escape(var_name) + r"\b"
            # Callable replacement: re.sub would otherwise interpret backslashes in
            # repr() as backrefs.
            replacement: str = repr(var_value)

            def _replace(_m: re.Match[str], r: str = replacement) -> str:
                return r

            fn_serialized = re.sub(pattern, _replace, fn_serialized)
    return fn_serialized


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
    format_paths: dict[str, str]

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
        self.format_paths = {}
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
        serialized = dict(
            template=self.template,
            format_args=format_args,
            format_kwargs=format_kwargs,
        )
        if self.format_paths:
            serialized["format_paths"] = self.format_paths
        return serialized

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, Op]) -> Self:
        format_args = [other_ops[arg] for arg in data["format_args"]]
        format_kwargs = {k: other_ops[v] for k, v in data["format_kwargs"].items()}
        op = cls(data["template"], *format_args, **format_kwargs)
        op.format_paths = data.get("format_paths", {})
        return op


def format_op(template: str, *args: list[str] | Op, **kwargs: list[str] | Op) -> Op:
    return FormatOp(template, *args, **kwargs)


@Op.registry.register("LambdaOp")
class LambdaOp(Op):
    fn: RowLambdaFn | ListLambdaFn | ExpandLambdaFn | ReduceLambdaFn

    def __init__(
        self,
        inputs: Sequence[list[str] | Op],
        fn: Callable[..., Any],
        code: str | None = None,
        mode: LambdaMode = "row",
        timeout_s: float | None = None,
        memory_mb: int | None = None,
        image: str | None = None,
        expected_runtime_s: float | None = None,
        cache: bool | None = None,
        sample_value: Any | None = None,
        by: str | None = None,
    ) -> None:
        input_ops: list[Op] = [
            inp if isinstance(inp, Op) else DataOp(inp) for inp in inputs
        ]
        super().__init__(input_ops)
        self.fn = fn
        self.sample_value = sample_value
        self.code = code or _serialize_fn(fn)
        self.spec = build_lambda_spec(
            {
                "fn_name": fn.__name__,
                "_code": self.code,
                "mode": mode,
                "by": by,
                "timeout_s": timeout_s,
                "memory_mb": memory_mb,
                "image": image,
                "expected_runtime_s": expected_runtime_s,
                "cache": cache,
            },
            "LambdaOp",
        )

    @property
    def mode(self) -> LambdaMode:
        return self.spec.mode

    @property
    def by(self) -> str | None:
        return self.spec.by

    @property
    def timeout_s(self) -> float | None:
        return self.spec.timeout_s

    @property
    def memory_mb(self) -> int | None:
        return self.spec.memory_mb

    @property
    def image(self) -> str | None:
        return self.spec.image

    @property
    def expected_runtime_s(self) -> float | None:
        return self.spec.expected_runtime_s

    @property
    def cache(self) -> bool | None:
        """Opt-in task caching; passed to FlowMesh only when set."""
        return self.spec.cache

    @property
    def takes_whole_columns(self) -> bool:
        """List and aligned Lambdas receive whole columns and return a list."""
        return self.mode in ("list", "aligned")

    def _serialize(self) -> dict[str, Any]:
        data = self.spec.model_dump(by_alias=True, exclude_none=True)
        data["_inputs"] = [inp.id for inp in self.inputs]
        if self.sample_value is not None:
            data["sample_value"] = self.sample_value
        return data

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, Op]) -> Self:
        spec = build_lambda_spec(data, f"LambdaOp '{data.get('_id', '?')}'")
        try:
            fn: Callable[..., Any] = SubmittedFunction(spec.code, spec.fn_name)
        except ValueError as exc:
            raise ValueError(
                f"Invalid LambdaOp function '{spec.fn_name}': {exc}"
            ) from exc
        op = cls(
            inputs=[other_ops[inp] for inp in data["_inputs"]],
            fn=fn,
            code=spec.code,
            mode=spec.mode,
            by=spec.by,
            timeout_s=spec.timeout_s,
            memory_mb=spec.memory_mb,
            image=spec.image,
            expected_runtime_s=spec.expected_runtime_s,
            cache=spec.cache,
            sample_value=data.get("sample_value"),
        )
        return op


def lambda_op(inputs: list[list[str] | Op], fn: RowLambdaFn) -> LambdaOp:
    return LambdaOp(inputs, fn)
