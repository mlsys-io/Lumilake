import os
import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _TESTS_DIR.parent

for path in (str(_TESTS_DIR), str(_ROOT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

for _key in ("LUMILAKE_REQUIRE_IDENTITY_PROVIDER",):
    os.environ[_key] = ""

from lumilake_server.runtime import flowmesh_client  # noqa: E402


@pytest.fixture(autouse=True)
def _drop_per_test_http_client_cache():
    """Drop the per-loop httpx client cache between tests; the loop dies
    with the test so we cannot ``aclose()`` safely."""
    yield
    with flowmesh_client._http_clients_lock:
        flowmesh_client._http_clients.clear()
