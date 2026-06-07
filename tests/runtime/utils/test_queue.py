import asyncio
from typing import Any

import pytest

from lumilake_server.runtime.utils import queue as queue_module
from lumilake_server.runtime.utils.queue import TSQueue


class CountingTSQueue(TSQueue[str]):
    def __init__(self) -> None:
        super().__init__()
        self.get_nowait_calls = 0

    def get_nowait(self) -> str:
        self.get_nowait_calls += 1
        return super().get_nowait()


@pytest.mark.asyncio
async def test_tsqueue_default_get_waits_for_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_sleep = asyncio.sleep
    sleep_calls: list[float | None] = []

    async def fake_sleep(delay: float | None = None, *args: Any, **kwargs: Any) -> None:
        sleep_calls.append(delay)
        await original_sleep(0)

    monkeypatch.setattr(queue_module.asyncio, "sleep", fake_sleep)
    queue = CountingTSQueue()

    get_task = asyncio.create_task(queue.get())
    await original_sleep(0)

    queue.put_nowait("ready")

    assert await asyncio.wait_for(get_task, timeout=1.0) == "ready"
    assert sleep_calls == []
    assert queue.get_nowait_calls <= 3


@pytest.mark.asyncio
async def test_tsqueue_close_wakes_all_default_get_waiters() -> None:
    queue: TSQueue[str] = TSQueue()
    get_tasks = [asyncio.create_task(queue.get()) for _ in range(3)]
    await asyncio.sleep(0)

    queue.close()

    for task in get_tasks:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
