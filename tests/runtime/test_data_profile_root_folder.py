"""Root-level S3 retrieval templates (e.g. ``{id}.json``) resolve to an empty
folder prefix. ``_list_blobs_for_folders`` must still list at root so the
downstream ``_average_size_for_folders`` finds the folder in the snapshot."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from lumilake_server.runtime.data_profile_utils import (
    _average_size_for_folders,
    _list_blobs_for_folders,
)


@pytest.mark.anyio
async def test_root_folder_is_listed_with_empty_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def _fake_list(prefix: str) -> tuple[dict[str, int | None], list[str]]:
        captured["prefix"] = prefix
        return ({"abc.json": 12, "def.json": 34}, [])

    monkeypatch.setattr("lumilake.envs.LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr("lumilake.envs.LUMID_DATA_TOKEN", "tok")

    with patch(
        "lumilake_server.runtime.data_profile_utils.lumid_list_blobs",
        new=AsyncMock(side_effect=_fake_list),
    ):
        sizes, folder_paths = await _list_blobs_for_folders([""])

    assert captured["prefix"] == ""
    assert sizes == {"abc.json": 12, "def.json": 34}
    assert "" in folder_paths


@pytest.mark.anyio
async def test_average_size_for_root_template_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full chain: empty folder + root listing + averaging must not raise."""

    async def _fake_list(prefix: str) -> tuple[dict[str, int | None], list[str]]:
        return ({"abc.json": 10, "def.json": 20}, [])

    monkeypatch.setattr("lumilake.envs.LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr("lumilake.envs.LUMID_DATA_TOKEN", "tok")

    with patch(
        "lumilake_server.runtime.data_profile_utils.lumid_list_blobs",
        new=AsyncMock(side_effect=_fake_list),
    ):
        sizes, folder_paths = await _list_blobs_for_folders([""])

    avg = _average_size_for_folders(
        folders=[""],
        listing_sizes=sizes,
        listing_folders=folder_paths,
        graph_key="g",
        node_id="n",
        org_id="org",
    )
    assert avg == 15.0


@pytest.mark.anyio
async def test_non_root_folder_sends_trailing_slash_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-empty folders must be queried with a trailing slash so lumid-data-app's
    literal string-prefix match cannot return sibling-prefix keys like ``foo.json``
    when the folder is ``foo/``."""
    captured: dict[str, Any] = {}

    async def _fake_list(prefix: str) -> tuple[dict[str, int | None], list[str]]:
        captured["prefix"] = prefix
        return ({"a.json": 10}, [])

    monkeypatch.setattr("lumilake.envs.LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr("lumilake.envs.LUMID_DATA_TOKEN", "tok")

    with patch(
        "lumilake_server.runtime.data_profile_utils.lumid_list_blobs",
        new=AsyncMock(side_effect=_fake_list),
    ):
        sizes, folder_paths = await _list_blobs_for_folders(["foo/"])

    assert captured["prefix"] == "foo/"
    assert sizes == {"foo/a.json": 10}
    assert "foo/" in folder_paths


@pytest.mark.anyio
async def test_folder_marker_keys_are_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero-byte directory-marker objects (keys ending with ``/``) returned by
    S3-compatible stores must be ignored so ``_average_size_for_folders`` does
    not try to derive a folder prefix from them and crash."""

    async def _fake_list(prefix: str) -> tuple[dict[str, int | None], list[str]]:
        return ({"a.json": 10, "sub/": 0, "sub/b.json": 30}, [])

    monkeypatch.setattr("lumilake.envs.LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr("lumilake.envs.LUMID_DATA_TOKEN", "tok")

    with patch(
        "lumilake_server.runtime.data_profile_utils.lumid_list_blobs",
        new=AsyncMock(side_effect=_fake_list),
    ):
        sizes, folder_paths = await _list_blobs_for_folders(["foo/"])

    assert sizes == {"foo/a.json": 10, "foo/sub/b.json": 30}
    assert "foo/sub/" not in sizes

    avg = _average_size_for_folders(
        folders=["foo/"],
        listing_sizes=sizes,
        listing_folders=folder_paths,
        graph_key="g",
        node_id="n",
        org_id="org",
    )
    assert avg == 10.0
