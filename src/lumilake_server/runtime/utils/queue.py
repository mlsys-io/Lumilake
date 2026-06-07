import asyncio
import queue
import threading
from abc import ABC, abstractmethod
from queue import Empty, Full
from typing import TypeVar

from lumilake.log import get_default_logger

logger = get_default_logger()

T = TypeVar("T")


class CloseSignal:
    pass


_CLOSE_SIGNAL = CloseSignal()
_FULL_QUEUE_RETRY_SECONDS: float = 0.01


class AsyncQueue[T](ABC):
    @abstractmethod
    async def get(self) -> T:
        pass

    @abstractmethod
    def get_nowait(self) -> T:
        pass

    @abstractmethod
    async def get_all(self) -> list[T]:
        pass

    @abstractmethod
    async def put(self, obj: T) -> None:
        pass

    @abstractmethod
    def put_nowait(self, obj: T) -> None:
        pass

    @abstractmethod
    def empty(self) -> bool:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class AIOQueue[T](AsyncQueue):
    """
    Asynchronous wrapper for asyncio queue.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[T | CloseSignal] = asyncio.Queue()
        self._is_closed: bool = False

    def _unwrap(self, message: T | CloseSignal) -> T:
        if isinstance(message, CloseSignal):
            self._is_closed = True
            raise asyncio.CancelledError()
        return message

    async def get(self) -> T:
        return self._unwrap(await self._queue.get())

    def get_nowait(self) -> T:
        return self._unwrap(self._queue.get_nowait())

    async def get_all(self) -> list[T]:
        items: list[T] = [await self.get()]
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if isinstance(item, CloseSignal):
                self._is_closed = True
                break
            items.append(item)
        return items

    async def put(self, obj: T) -> None:
        if self._is_closed:
            raise asyncio.CancelledError()
        await self._queue.put(obj)

    def put_nowait(self, obj: T) -> None:
        if self._is_closed:
            raise asyncio.CancelledError()
        self._queue.put_nowait(obj)

    def empty(self) -> bool:
        return self._queue.empty()

    def close(self) -> None:
        if not self._is_closed:
            self._is_closed = True
            self._queue.put_nowait(_CLOSE_SIGNAL)


class TSQueue[T](AsyncQueue):
    """
    Asynchronous wrapper for thread-safe queue.
    """

    def __init__(self, delay: float | None = None) -> None:
        self._queue: queue.Queue[T | CloseSignal] = queue.Queue()
        self._delay: float | None = delay
        self._is_closed: bool = False
        self._waiters: set[asyncio.Future[None]] = set()
        self._waiters_lock = threading.Lock()

    def _unwrap(self, message: T | CloseSignal) -> T:
        if isinstance(message, CloseSignal):
            self._is_closed = True
            raise asyncio.CancelledError()
        return message

    @staticmethod
    def _set_waiter_result(waiter: asyncio.Future[None]) -> None:
        if not waiter.done():
            waiter.set_result(None)

    def _wake_waiters(self) -> None:
        with self._waiters_lock:
            waiters = list(self._waiters)
            self._waiters.clear()
        for waiter in waiters:
            try:
                waiter.get_loop().call_soon_threadsafe(self._set_waiter_result, waiter)
            except RuntimeError:
                pass

    async def get(self, delay: float | None = None) -> T:
        delay = self._delay if delay is None else delay
        while True:
            try:
                return self.get_nowait()
            except Empty:
                if self._is_closed:
                    raise asyncio.CancelledError()
                if delay is not None:
                    await asyncio.sleep(delay)
                    continue
                waiter = asyncio.get_running_loop().create_future()
                with self._waiters_lock:
                    self._waiters.add(waiter)
                try:
                    try:
                        return self.get_nowait()
                    except Empty:
                        if self._is_closed:
                            raise asyncio.CancelledError()
                        await waiter
                finally:
                    with self._waiters_lock:
                        self._waiters.discard(waiter)

    def get_nowait(self) -> T:
        return self._unwrap(self._queue.get_nowait())

    async def get_all(self) -> list[T]:
        items: list[T] = [await self.get()]
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if isinstance(item, CloseSignal):
                self._is_closed = True
                break
            items.append(item)
        return items

    async def put(self, obj: T, delay: float | None = None) -> None:
        delay = self._delay if delay is None else delay
        retry_delay = _FULL_QUEUE_RETRY_SECONDS if delay is None else delay
        while True:
            try:
                self.put_nowait(obj)
                return
            except Full:
                await asyncio.sleep(retry_delay)

    def put_nowait(self, obj: T) -> None:
        if self._is_closed:
            raise asyncio.CancelledError()
        self._queue.put_nowait(obj)
        self._wake_waiters()

    def empty(self) -> bool:
        return self._queue.empty()

    def close(self) -> None:
        if not self._is_closed:
            self._is_closed = True
            self._queue.put_nowait(_CLOSE_SIGNAL)
            self._wake_waiters()
