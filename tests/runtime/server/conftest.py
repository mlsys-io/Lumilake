from collections.abc import Generator

import pytest
from support.runtime_server import cleanup_runtime_result_dirs, make_server

import lumilake.utils.job_storage as job_storage_module
from lumilake.utils.job_storage import InMemoryJobStorage


@pytest.fixture
def server_factory():
    return make_server


@pytest.fixture(autouse=True)
def _reset_job_storage() -> Generator[None, None, None]:
    job_storage_module._job_storage = InMemoryJobStorage()
    try:
        yield
    finally:
        cleanup_runtime_result_dirs()
