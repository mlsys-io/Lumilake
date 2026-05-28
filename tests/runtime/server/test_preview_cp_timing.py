"""Verify /jobs/preview surfaces job-manager + optimizer timings."""

import pytest

from lumilake_server.common import GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
    LLMChatOp,
    OpMessage,
    as_output,
    input_placeholder,
)


def _tiny_preview_graphs() -> dict:
    stock = input_placeholder("Stock")
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    return {"preview_graph": compiled}


@pytest.mark.asyncio
async def test_preview_schedule_surfaces_control_plane_timings(
    server_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = server_factory()
    graphs = _tiny_preview_graphs()

    async def _stub_select_workers(_runtime_graph) -> tuple[list[str], dict]:
        return ["gpu-worker"], {"gpu-worker": {"has_gpu": True}}

    monkeypatch.setattr(
        server, "_select_preview_workers_and_profiles", _stub_select_workers
    )

    preview = await server.preview_schedule(graphs=graphs, data_profile_results={})

    assert preview.selection_seconds is not None
    assert preview.clustering_seconds is not None
    assert preview.optimization_seconds is not None
    assert isinstance(preview.selection_seconds, float)
    assert isinstance(preview.clustering_seconds, float)
    assert isinstance(preview.optimization_seconds, float)
    assert preview.selection_seconds >= 0.0
    assert preview.clustering_seconds >= 0.0
    assert preview.optimization_seconds > 0.0
    assert preview.selected_workers == ["gpu-worker"]
    assert preview.merged_runtime_node_count > 0
