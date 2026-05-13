"""Tests for the config loader/saver."""

from pathlib import Path

import pytest

from lumilake.sdk import LumilakeConfig


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "config.toml"
    cfg = LumilakeConfig(base_url="http://localhost:19000")
    cfg.save(target)
    assert target.exists()
    loaded = LumilakeConfig.load(target)
    assert loaded.base_url == cfg.base_url


def test_load_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="config not found"):
        LumilakeConfig.load(tmp_path / "absent.toml")


def test_save_creates_parent(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "config.toml"
    LumilakeConfig(base_url="x").save(target)
    assert target.exists()
