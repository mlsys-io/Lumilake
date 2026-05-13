from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from lumilake.runtime.data import Data, DataType

"""Functional input primitives used by the optimizer.

This module provides compact, well-typed input classes (FnInput and
specializations) used during logical graph compilation and optimization.
They operate on runtime-level `Data` objects and avoid runtime worker
protocols.
"""


class FnInput(ABC):
    NAME: str = "fn"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        args: list[Data],
        kwargs: dict,
        task_id: str,
    ) -> None:
        self.request_id = request_id
        self.op_id = op_id
        self.is_eager = is_eager
        self.ref_count = ref_count
        self.max_iter = max_iter
        # Arguments are plain Data instances.
        self.args: list[Data] = args
        self.kwargs: dict = kwargs
        self._iteration = 0
        # Optional task identifier (used by some functional inputs).
        self.task_id = task_id

    def is_dead(self) -> bool:
        return False

    def get_cache_keys(self) -> dict[int, object] | None:
        return None

    def empty_output(self) -> object:
        return None

    @asynccontextmanager
    async def input_iterator(self, *args, **kwargs):
        """Async context manager that yields this input once.

        Many functional inputs use an ``async with self.input_iterator(puller)``
        which expects an async iterator. This context manager yields this
        instance exactly once to keep implementations simple.
        """

        async def _agen() -> AsyncGenerator["FnInput", None]:
            yield self

        yield _agen()

    async def run(self) -> AsyncGenerator[Data | None, None]:
        async with self.input_iterator() as input_iterator:
            async for inp in input_iterator:
                if (
                    self.max_iter is not None
                    and self.max_iter >= 0
                    and self._iteration > self.max_iter
                ):
                    # Exceeds the maximum number of iterations.
                    yield None
                elif inp.is_dead():
                    yield None
                else:
                    async for data in inp._run():
                        yield data

    @abstractmethod
    def _run(self) -> AsyncGenerator[Data | None, None]:
        raise NotImplementedError()


class FnInputBatch(list):
    """A thin list-like batch of `FnInput` instances used in some
    functional utilities. Kept minimal and explicit for compilation logic.
    """

    pass


class DataFnInput(FnInput):
    NAME: str = "data"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        data: list[str],
    ) -> None:
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=None,
            args=[],
            kwargs={},
            task_id=op_id,
        )
        self.data = data

    @property
    def output_type(self) -> DataType:
        return DataType.TEXT

    async def _run(self) -> AsyncGenerator[Data | None, None]:
        yield Data.text(self.data, indices=None)


class InputFnInput(FnInput):
    NAME: str = "input"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        inputs: list[str],
    ) -> None:
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=None,
            args=[],
            kwargs={},
            task_id=op_id,
        )
        self.inputs = inputs

    @property
    def output_type(self) -> DataType:
        return DataType.TEXT

    async def _run(self) -> AsyncGenerator[Data | None, None]:
        yield Data.text(self.inputs, indices=None)


class OutputFnInput(FnInput):
    NAME: str = "output"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        output: Data,
    ) -> None:
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=None,
            args=[output],
            kwargs={},
            task_id=op_id,
        )

    @property
    def output_type(self) -> DataType:
        raise ValueError("OutputFnInput has dynamic output type")

    async def _run(self) -> AsyncGenerator[Data | None, None]:
        # args stores Data instances directly
        yield self.args[0]
