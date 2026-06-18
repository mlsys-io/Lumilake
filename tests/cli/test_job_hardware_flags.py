"""Tests for `_build_hardware_payload` in the job CLI commands."""

import json
from pathlib import Path

import pytest
import typer
from lumilake_cli.commands.job import _build_hardware_payload


def test_returns_none_when_all_inputs_unset() -> None:
    assert _build_hardware_payload(None, None, None, None, None) is None


def test_only_cpu_flag_produces_cpu_field() -> None:
    assert _build_hardware_payload(4, None, None, None, None) == {"cpu": 4}


def test_only_hardware_json_loads_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "hw.json"
    path.write_text(
        json.dumps({"cpu": 2, "memory": "8Gi", "gpu": 1, "gpu_memory": "16Gi"})
    )
    assert _build_hardware_payload(None, None, None, None, path) == {
        "cpu": 2,
        "memory": "8Gi",
        "gpu": 1,
        "gpu_memory": "16Gi",
    }


def test_flag_wins_over_hardware_json_on_conflict(tmp_path: Path) -> None:
    path = tmp_path / "hw.json"
    path.write_text(json.dumps({"cpu": 2, "memory": "8Gi"}))
    result = _build_hardware_payload(16, None, None, None, path)
    assert result == {"cpu": 16, "memory": "8Gi"}


def test_missing_hardware_json_file_exits_nonzero(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    with pytest.raises(typer.Exit) as excinfo:
        _build_hardware_payload(None, None, None, None, missing)
    assert excinfo.value.exit_code == 1


def test_invalid_json_exits_nonzero(tmp_path: Path) -> None:
    path = tmp_path / "hw.json"
    path.write_text("not json {")
    with pytest.raises(typer.Exit) as excinfo:
        _build_hardware_payload(None, None, None, None, path)
    assert excinfo.value.exit_code == 1


def test_non_dict_json_exits_nonzero(tmp_path: Path) -> None:
    path = tmp_path / "hw.json"
    path.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(typer.Exit) as excinfo:
        _build_hardware_payload(None, None, None, None, path)
    assert excinfo.value.exit_code == 1


def test_unknown_field_in_hardware_json_exits_nonzero(tmp_path: Path) -> None:
    path = tmp_path / "hw.json"
    path.write_text(json.dumps({"cpu": 2, "tpu": 4}))
    with pytest.raises(typer.Exit) as excinfo:
        _build_hardware_payload(None, None, None, None, path)
    assert excinfo.value.exit_code == 1
