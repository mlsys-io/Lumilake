import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import TypeVar

from lumilake.log import log_on_exception_async
from lumilake.runtime.utils.queue import AIOQueue, AsyncQueue

E = TypeVar("E")
R = TypeVar("R")
Ctx = TypeVar("Ctx")


class EventLoop[E, R, Ctx](ABC):
    """
    Abstract base class for event-driven processing loops.

    This class provides a framework for processing events asynchronously
    using a handler function. Events are queued in an input channel and
    processed sequentially or concurrently depending on the implementation.
    Results can optionally be stored in a result pool for retrieval.

    Type Parameters:
        E: Event type - the type of events to be processed
        R: Result type - the type of results produced by event processing
        Ctx: Context type - the type of context passed to the handler function

    Args:
        handler_func: Async function that processes events and returns results
        in_channel: Queue for incoming events
        result_collector: Optional pool or queue for storing results with string keys
        context_manager: Optional async context manager for handler execution
        key_prefix: Optional prefix for generated result keys
    """

    def __init__(
        self,
        handler_func: Callable[[E, Ctx | None], Awaitable[R]],
        in_channel: AsyncQueue,
        context_manager: AbstractAsyncContextManager[Ctx] | None = None,
    ):
        self._handler_func = handler_func
        self._in_channel = in_channel
        self._context_manager = context_manager

        self._is_started: bool = False
        self._is_stopped: bool = False

    def is_started(self) -> bool:
        return self._is_started

    def is_stopped(self) -> bool:
        return self._is_stopped

    def is_running(self) -> bool:
        return self.is_started() and not self.is_stopped()

    def _check_running(self) -> None:
        if not self.is_started():
            raise ValueError("Event loop has not been started.")
        if self.is_stopped():
            raise ValueError("Event loop has been stopped.")

    async def start(self) -> None:
        if self.is_started():
            return
        await self._start_handler()
        self._is_started = True

    async def stop(self) -> None:
        self._in_channel.close()
        await self.join()

    async def join(self) -> None:
        if self.is_stopped():
            return
        await self._join_handler()
        self._is_stopped = True

    async def add_event(self, event: E) -> None:
        self._check_running()
        await self._in_channel.put(event)

    @log_on_exception_async(ignore=[asyncio.CancelledError])
    async def loop(self, *args, **kwargs) -> None:
        if self._context_manager is None:
            await self._loop_func(None, *args, **kwargs)
        else:
            async with self._context_manager as ctx:
                await self._loop_func(ctx, *args, **kwargs)

    @abstractmethod
    async def _start_handler(self) -> None:
        pass

    @abstractmethod
    async def _join_handler(self) -> None:
        pass

    @abstractmethod
    async def _loop_func(self, ctx: Ctx | None, *args, **kwargs) -> None:
        pass


class AsyncEventLoop(EventLoop[E, R, Ctx]):
    """
    Asynchronous event loop implementation that processes events sequentially.

    This implementation runs within the current async event loop and processes events
    one at a time in the order they arrive. The processing is handled by an asyncio Task
    that continuously reads from the input channel and invokes the handler function.
    """

    def __init__(
        self,
        handler_func: Callable[[E, Ctx | None], Awaitable[R]],
        in_channel: AsyncQueue | None = None,
        context_manager: AbstractAsyncContextManager[Ctx] | None = None,
    ):
        super().__init__(
            handler_func=handler_func,
            in_channel=in_channel or AIOQueue(),
            context_manager=context_manager,
        )
        self._loop_task: asyncio.Task | None = None

    @property
    def loop_task(self) -> asyncio.Task:
        if self._loop_task is None:
            raise ValueError("Event loop has not been started.")
        return self._loop_task

    async def _start_handler(self) -> None:
        assert self._loop_task is None
        self._loop_task = asyncio.create_task(self.loop())

    async def _join_handler(self) -> None:
        assert self._loop_task is not None
        await self._loop_task
        self._loop_task = None

    async def _loop_func(self, ctx: Ctx | None, *args, **kwargs) -> None:
        assert self._in_channel is not None
        while True:
            item = await self._in_channel.get()
            await self._handler_func(item, ctx)
