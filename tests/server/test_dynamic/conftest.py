import asyncio
import time
from typing import Any

import pytest


@pytest.fixture
def wait_for_child():
    """Wait until a dynamic parent has registered at least one child job."""

    async def _wait(job_routes: Any, job_id: str, timeout: float = 5.0) -> Any:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            parent = job_routes.jobs[job_id]
            if parent.child_job_ids:
                return parent
            await asyncio.sleep(0.005)
        raise AssertionError(
            f"round 0 child was never created for {job_id} within {timeout}s"
        )

    return _wait
