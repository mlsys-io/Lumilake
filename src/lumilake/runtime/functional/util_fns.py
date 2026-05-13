import itertools
from collections.abc import AsyncGenerator, Callable, Iterable
from contextlib import asynccontextmanager

from lumilake.common import Slice
from lumilake.ops import SingleDtype
from lumilake.runtime.data import Data, DataType
from lumilake.runtime.functional.fns import FnInput
from lumilake.utils import utils

# Worker RPC constructs removed: this module operates on Data directly.


class FormatFnInput(FnInput):
    NAME: str = "format"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        template: str,
        format_args: list[Data],
        format_kwargs: dict[str, Data],
    ):
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=max_iter,
            args=format_args,
            kwargs=format_kwargs,
            task_id=op_id,
        )
        self.template = template

    @property
    def output_type(self) -> DataType:
        return DataType.TEXT

    async def _run(self) -> AsyncGenerator[Data | None, None]:
        template = self.template
        # args/kwargs are Data instances; call Data.as_text() to get text lists
        format_args = [arg.as_text() for arg in self.args]
        format_kwargs = {k: v.as_text() for k, v in self.kwargs.items()}
        cur_outputs = []
        first_data = next(iter(itertools.chain(self.args, self.kwargs.values())))
        for i in range(len(first_data)):
            formatted = template.format(
                *[arg[i] for arg in format_args],
                **{k: v[i] for k, v in format_kwargs.items()},
            )
            cur_outputs.append(formatted)
        yield first_data.into_text(cur_outputs)


class LambdaFnInput(FnInput):
    NAME: str = "lambda"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        inputs: list[Data],
        fn: Callable[[tuple[SingleDtype, ...]], str],
    ) -> None:
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=max_iter,
            args=inputs,
            kwargs={},
            task_id=op_id,
        )
        self.fn = fn

    @property
    def output_type(self) -> DataType:
        return DataType.TEXT

    async def _run(self) -> AsyncGenerator[Data | None, None]:
        # self.args is a list of Data
        data = self.args
        inputs = [d.data for d in data]
        out = [self.fn(fn_args) for fn_args in zip(*inputs)]
        yield data[0].into_text(out)


class SliceFnInput(FnInput):
    NAME: str = "slice"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        inp: Data,
        indices: Iterable[int] | Slice,
    ) -> None:
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=max_iter,
            args=[inp],
            kwargs={},
            task_id=op_id,
        )
        self.indices = indices

    @property
    def output_type(self) -> DataType:
        raise ValueError("SliceFnInput has dynamic output type")

    async def _run(self) -> AsyncGenerator[Data | None, None]:
        indices = utils.indices_to_list(self.indices)
        # args are Data instances; use Data.get_by_indices
        yield self.args[0].get_by_indices(indices, uncheck=True)


class ConcatFnInput(FnInput):
    NAME: str = "concat"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        inputs: list[Data],
    ) -> None:
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=max_iter,
            args=inputs,
            kwargs={},
            task_id=op_id,
        )

    @property
    def output_type(self) -> DataType:
        raise ValueError("ConcatFnInput has dynamic output type")

    @asynccontextmanager
    async def input_iterator(self, is_concat: bool = True):
        async with super().input_iterator(is_concat) as it:
            yield it

    async def _run(self) -> AsyncGenerator[Data | None, None]:
        if len(self.args) == 0:
            raise ValueError("ConcatFnInput must have at least one input")
        ret = self.args[0]
        for arg in self.args[1:]:
            ret += arg
        yield ret
