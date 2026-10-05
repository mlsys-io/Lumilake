"""Regression: ``_aggregate_output_node`` must fail closed on malformed
``OutputOp`` paths (empty dotted segments) instead of silently dropping them
and resolving a different value."""

import json

import pytest

from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager


async def _aggregate(output_path: str | None) -> list[str]:
    return await FlowmeshRuntimeManager()._aggregate_output_node(
        output_op_id="output",
        output_task_id="task-1",
        request_id="req-1",
        items=[{"output": "silently accepted"}],
        output_path=output_path,
        list_lambda=False,
        list_lambda_cardinality=False,
    )


@pytest.mark.asyncio
async def test_aggregate_rejects_empty_segment() -> None:
    with pytest.raises(RuntimeError, match="malformed path"):
        await _aggregate("items..output")


@pytest.mark.asyncio
async def test_aggregate_rejects_trailing_dot() -> None:
    with pytest.raises(RuntimeError, match="malformed path"):
        await _aggregate("items.output.")


@pytest.mark.asyncio
async def test_aggregate_rejects_leading_dot() -> None:
    with pytest.raises(RuntimeError, match="malformed path"):
        await _aggregate("items..output")


@pytest.mark.asyncio
async def test_aggregate_rejects_bare_items_dot() -> None:
    with pytest.raises(RuntimeError, match="malformed path"):
        await _aggregate("items.")


@pytest.mark.asyncio
async def test_aggregate_rejects_bracket_only_segment() -> None:
    with pytest.raises(RuntimeError, match="malformed path"):
        await _aggregate("items.rows.[0].output")


@pytest.mark.asyncio
async def test_aggregate_valid_indexed_path_resolves() -> None:
    items = [{"rows": [{"json": {"choices": [{"message": {"content": "ok"}}]}}]}]
    values = await FlowmeshRuntimeManager()._aggregate_output_node(
        output_op_id="output",
        output_task_id="task-1",
        request_id="req-1",
        items=items,
        output_path="items.rows.json.choices[0].message.content",
        list_lambda=False,
        list_lambda_cardinality=False,
    )
    assert values == ["ok"]


@pytest.mark.asyncio
async def test_aggregate_valid_bracket_index_resolves() -> None:
    items = [{"rows": [{"output": "ok"}]}]
    values = await FlowmeshRuntimeManager()._aggregate_output_node(
        output_op_id="output",
        output_task_id="task-1",
        request_id="req-1",
        items=items,
        output_path="items.rows[0].output",
        list_lambda=False,
        list_lambda_cardinality=False,
    )
    assert values == ["ok"]


async def _aggregate_list_lambda(output_path: str | None) -> list[str]:
    return await FlowmeshRuntimeManager()._aggregate_output_node(
        output_op_id="output",
        output_task_id="task-1",
        request_id="req-1",
        items=[
            {"output": {"fid": "f1", "statement": "s1"}},
            {"output": {"fid": "f2", "statement": "s2"}},
        ],
        output_path=output_path,
        list_lambda=True,
        list_lambda_cardinality=False,
    )


@pytest.mark.asyncio
async def test_aggregate_list_lambda_projection_walks_each_item() -> None:
    """A list-mode Lambda output accepts the same ``items.<field>[.<sub>...]``
    grammar as other outputs; the walked field is collected across every item
    into one whole-list value."""
    values = await _aggregate_list_lambda("items.output.fid")
    assert len(values) == 1
    assert json.loads(values[0]) == ["f1", "f2"]


@pytest.mark.asyncio
async def test_aggregate_list_lambda_explicit_default_equals_default() -> None:
    """``path: items.output`` on a list-mode Lambda equals the builder default
    (one whole-list value holding every item)."""
    explicit = await _aggregate_list_lambda("items.output")
    default = await _aggregate_list_lambda(None)
    assert explicit == default
    assert len(explicit) == 1
    assert json.loads(explicit[0]) == [
        {"fid": "f1", "statement": "s1"},
        {"fid": "f2", "statement": "s2"},
    ]


@pytest.mark.asyncio
async def test_aggregate_list_lambda_rejects_malformed_paths() -> None:
    """A list-mode Lambda output rejects the same malformed paths as other
    outputs (empty segments, bracket-only segments)."""
    for bad in ("items.", "items..x", "items.[0]"):
        with pytest.raises(RuntimeError, match="malformed path"):
            await _aggregate_list_lambda(bad)
