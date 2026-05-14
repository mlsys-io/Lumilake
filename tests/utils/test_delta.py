"""Round-trip tests for the Delta Lake I/O helpers.

Uses local-filesystem tables via tmp_path. S3-specific code paths are
not covered by unit tests; they are exercised in integration against a
real MinIO/S3 when the harness runs end-to-end.
"""

from typing import Any

import pyarrow as pa
import pytest

from lumilake_server.utils.delta import list_delta_versions, read_delta, write_delta


@pytest.fixture
def sample_records() -> list[dict[str, Any]]:
    return [
        {"item_id": "p_1001", "entity": "Tesla", "sentiment": "bullish"},
        {"item_id": "p_1002", "entity": "Ford", "sentiment": "neutral"},
        {"item_id": "p_1003", "entity": "Tesla", "sentiment": "bearish"},
    ]


def test_write_then_read_round_trip(tmp_path, sample_records):
    path = str(tmp_path / "facts")
    write_delta(sample_records, path, mode="overwrite")
    table = read_delta(path)
    assert table.num_rows == 3
    assert set(table.column_names) == {"item_id", "entity", "sentiment"}
    assert sorted(table["item_id"].to_pylist()) == ["p_1001", "p_1002", "p_1003"]


def test_append_produces_new_version(tmp_path, sample_records):
    path = str(tmp_path / "facts")
    write_delta(sample_records[:1], path, mode="overwrite")
    write_delta(sample_records[1:], path, mode="append")

    # Latest read sees all rows.
    latest = read_delta(path)
    assert latest.num_rows == 3

    # Time-travel to version 0 recovers the initial write only.
    v0 = read_delta(path, version=0)
    assert v0.num_rows == 1
    assert v0["item_id"].to_pylist() == ["p_1001"]


def test_overwrite_replaces_contents(tmp_path, sample_records):
    path = str(tmp_path / "facts")
    write_delta(sample_records, path, mode="overwrite")
    write_delta(
        [{"item_id": "new", "entity": "Rivian", "sentiment": "neutral"}],
        path,
        mode="overwrite",
    )

    latest = read_delta(path)
    assert latest.num_rows == 1
    assert latest["item_id"].to_pylist() == ["new"]


def test_column_projection(tmp_path, sample_records):
    path = str(tmp_path / "facts")
    write_delta(sample_records, path, mode="overwrite")
    projected = read_delta(path, columns=["item_id", "sentiment"])
    assert set(projected.column_names) == {"item_id", "sentiment"}


def test_history_reports_versions(tmp_path, sample_records):
    path = str(tmp_path / "facts")
    write_delta(sample_records[:1], path, mode="overwrite")
    write_delta(sample_records[1:2], path, mode="append")
    write_delta(sample_records[2:], path, mode="append")

    history = list_delta_versions(path)
    assert len(history) == 3
    versions = sorted(entry["version"] for entry in history)
    assert versions == [0, 1, 2]


def test_accepts_pyarrow_table_directly(tmp_path, sample_records):
    path = str(tmp_path / "facts")
    arrow_table = pa.Table.from_pylist(sample_records)
    write_delta(arrow_table, path, mode="overwrite")
    assert read_delta(path).num_rows == 3


def test_accepts_pandas_dataframe(tmp_path, sample_records):
    pd = pytest.importorskip("pandas")
    path = str(tmp_path / "facts")
    df = pd.DataFrame(sample_records)
    write_delta(df, path, mode="overwrite")
    assert read_delta(path).num_rows == 3


def test_rejects_unsupported_data_type(tmp_path):
    path = str(tmp_path / "facts")
    with pytest.raises(TypeError):
        write_delta(42, path, mode="overwrite")  # type: ignore[arg-type]
