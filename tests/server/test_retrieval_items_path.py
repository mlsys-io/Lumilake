import pytest

from lumilake_server.common import retrieval_items_path


def test_retrieval_items_path_sql() -> None:
    assert retrieval_items_path("sql") == "items.table"


def test_retrieval_items_path_s3() -> None:
    assert retrieval_items_path("s3") == "items.content"


def test_retrieval_items_path_agent() -> None:
    assert retrieval_items_path("agent") == "items.table"


def test_retrieval_items_path_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported retrieval mode"):
        retrieval_items_path("unknown")
