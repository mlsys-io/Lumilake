import functools
import re
from collections.abc import AsyncGenerator, Callable, Iterable
from contextlib import asynccontextmanager

from lumilake.ops.cond_ops import Predicate
from lumilake.runtime.data import Data, DataType
from lumilake.runtime.functional.fns import FnInput


class SwitchFnInput(FnInput):
    NAME: str = "switch"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        inp: Data,
        cond_args: list[Data],
        pred: Predicate | None,
        branch: bool,
        dead_on_empty: bool,
    ) -> None:
        if isinstance(pred, list) and len(pred) != len(cond_args):
            raise ValueError("Inconsistent number of predicates")
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=max_iter,
            args=cond_args,
            kwargs=dict(inp=inp),
            task_id=op_id,
        )
        self.pred = pred
        self.branch = branch
        self.dead_on_empty = dead_on_empty

    @property
    def output_type(self) -> DataType:
        raise ValueError("SwitchFnInput has dynamic output type")

    async def run(self) -> AsyncGenerator[Data | None, None]:
        async with self.input_iterator() as input_iterator:
            async for inp in input_iterator:
                # Worker-level 'is_dead' signals are not supported here.
                assert not inp.is_dead(), "is_dead unsupported"
                if (
                    inp.max_iter is not None
                    and inp.max_iter >= 0
                    and inp._iteration >= inp.max_iter
                ):
                    # Exceeds the maximum number of iterations. This control
                    # flow is worker-specific and should not occur here.
                    assert False, "loop-control branch not supported"
                else:
                    async for data in inp._run():
                        yield data

    async def _run(self) -> AsyncGenerator[Data | None, None]:
        def apply_regex(patterns: list[str], args: list[str]) -> bool:
            assert len(patterns) == len(args)
            return not all(
                re.match(pattern, arg) for pattern, arg in zip(patterns, args)
            ) and all(len(arg) > 0 for arg in args)

        def apply_not_regex(patterns: list[str], args: list[str]) -> bool:
            return not apply_regex(patterns, args)

        def apply_pred(
            pred: Callable[..., bool],
            iteration: Iterable[int],
            args: list[str],
        ) -> bool:
            return pred(*iteration, *args)

        def apply_not_pred(
            pred: Callable[..., bool],
            iteration: Iterable[int],
            args: list[str],
        ) -> bool:
            return not pred(*iteration, *args)

        inp_arg = self.kwargs["inp"]
        inp_data = inp_arg

        if self.pred is None:
            if self.branch:
                # Forward the input if no predicate is provided.
                yield inp_data
            else:
                yield inp_data.into_empty(inp_data.dtype)
            return

        cond_args: list[list[str]] = [
            (
                arg.as_text()
                if arg.is_text()
                else [msg_data[-1].content for msg_data in arg.as_message()]
            )
            for arg in self.args
        ]

        pred: Callable[[list[str]], bool]
        if isinstance(self.pred, str):
            if self.branch:
                pred = functools.partial(apply_regex, [self.pred] * len(cond_args))
            else:
                pred = functools.partial(apply_not_regex, [self.pred] * len(cond_args))
        elif isinstance(self.pred, list):
            if self.branch:
                pred = functools.partial(apply_regex, self.pred)
            else:
                pred = functools.partial(apply_not_regex, self.pred)
        else:
            # 'looping' is a worker-specific attribute; disallow here.
            assert not getattr(self, "looping", False), "looping unsupported"
            iteration = ()
            if self.branch:
                pred = functools.partial(apply_pred, self.pred, iteration)
            else:
                pred = functools.partial(apply_not_pred, self.pred, iteration)

        pred_list = [pred(cond) for cond in zip(*cond_args)]
        out = inp_data.filter(pred_list)
        if out.is_empty() and self.dead_on_empty:
            yield None
        else:
            yield out


class MergeFnInput(FnInput):
    NAME: str = "merge"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        inputs: list[Data],
    ):
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
        raise ValueError("MergeFnInput has dynamic output type")

    async def _run(self) -> AsyncGenerator[Data | None, None]:
        # Merge semantics: filter out empty Data and treat falsy Data as dead.
        outputs = [arg for arg in self.args if arg is not None]
        if len(outputs) == 0:
            yield None
        elif len(outputs) > 1:
            raise ValueError("Multiple alive inputs found")
        yield outputs[0]


class EnterFnInput(FnInput):
    NAME: str = "enter"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        init_inp: Data,
        future_inp: Data,
    ) -> None:
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=max_iter,
            args=[init_inp, future_inp],
            kwargs={},
            task_id=op_id,
        )

    @property
    def output_type(self) -> DataType:
        raise ValueError("EnterFnInput has dynamic output type")

    def _get_worker_request(self, *args, **kwargs):
        assert False, "_get_worker_request unsupported"

    def _get_unsub_request(self, *args, **kwargs):
        assert False, "_get_unsub_request unsupported"

    def _get_arg(self) -> Data:
        return self.args[0] if self._iteration == 1 else self.args[1]

    def get_all_args(self) -> set[Data]:
        return {self._get_arg()}

    async def resolve(self) -> "EnterFnInput":
        # Cross-worker resolution is not supported here.
        assert False, "resolve unsupported"

    @asynccontextmanager
    async def input_iterator(self, is_concat: bool = False):
        # Align with FnInput.input_iterator signature: yield an
        # AsyncGenerator that yields this input once. This avoids
        # type incompatibility complaints from static checkers.
        async def iterate_async() -> AsyncGenerator["FnInput", None]:
            # Simple iterator: yield self once.
            yield self

        self._iteration += 1
        yield iterate_async()

    async def _run(self):
        arg = self._get_arg()
        # mark_resolve / remote resolution not supported
        assert arg is not None
        # If arg is local Data, return it directly
        yield arg


class ExitFnInput(FnInput):
    NAME: str = "exit"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        inp: Data,
    ) -> None:
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=None,
            args=[inp],
            kwargs={},
            task_id=op_id,
        )

    @property
    def output_type(self) -> DataType:
        raise ValueError("ExitFnInput has dynamic output type")

    async def run(self) -> AsyncGenerator[Data | None, None]:
        # Override to ignore dead signals from the input
        async with self.input_iterator() as input_iterator:
            async for inp in input_iterator:
                # 'is_dead' semantics are worker-specific; not supported here.
                assert not inp.is_dead(), "is_dead unsupported"
                yield inp.args[0]
