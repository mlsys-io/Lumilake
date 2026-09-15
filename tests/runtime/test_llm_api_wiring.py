import pytest
from lumilake import envs

from lumilake_server.common import ApiConfig, GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
    DataRetrievalOp,
    LLMChatOp,
    OpMessage,
    as_output,
    input_placeholder,
)
from lumilake_server.runtime.runtime_graph import RuntimeGraph, RuntimeGraphBuilder

_LUMID_URL = "http://lumid-data"
_LUMID_TOKEN = "test-token"


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", _LUMID_URL)
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", _LUMID_TOKEN)


def _build_api_graph(**config_kwargs) -> tuple[RuntimeGraph, str]:
    stock = input_placeholder("Stock")
    api = config_kwargs.pop("api", None) or ApiConfig(
        url="https://api.example.com/v1/chat/completions",
        credential_env="OPENAI_API_KEY",
    )
    cfg = GenerationConfig(
        model="meta-llama/Llama-3.1-8B-Instruct",
        api=api,
        **config_kwargs,
    )
    llm = LLMChatOp([OpMessage(role="user", content=stock)], config=cfg)
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    return RuntimeGraphBuilder().build(compiled), llm.id


def test_api_config_emits_api_task_type() -> None:
    runtime_graph, llm_id = _build_api_graph()
    node = runtime_graph.nodes[llm_id]
    assert node.task_type == "api"
    assert node.backend == "api"
    flowmesh_node = node.to_flowmesh_node()
    spec = flowmesh_node["spec"]
    assert spec["taskType"] == "api"
    assert "model" not in spec
    assert "inference" not in spec
    assert "data" not in spec
    api = spec["api"]
    assert api["method"] == "POST"
    assert api["url"] == "https://api.example.com/v1/chat/completions"
    assert api["auth"]["credential_env"] == "OPENAI_API_KEY"
    body = api["json"]
    assert body["model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert body["messages"] == [{"role": "user", "content": "NVDA"}]


def test_api_config_model_override() -> None:
    runtime_graph, llm_id = _build_api_graph(
        api=ApiConfig(
            url="https://api.example.com/v1/chat/completions",
            model="gpt-4o",
            credential_env="OPENAI_API_KEY",
        )
    )
    node = runtime_graph.nodes[llm_id]
    body = node.api_spec["json"]
    assert body["model"] == "gpt-4o"
    assert node.model == "gpt-4o"


def test_api_samplers_flow_into_body() -> None:
    runtime_graph, llm_id = _build_api_graph(max_tokens=64, temperature=0.5)
    node = runtime_graph.nodes[llm_id]
    body = node.api_spec["json"]
    assert body["max_tokens"] == 64
    assert body["temperature"] == 0.5


def test_api_key_not_in_spec() -> None:
    """The API key is a worker-side secret resolved from ``credential_env`` at
    call time; it must never be embedded in the task spec, which is archived."""
    runtime_graph, llm_id = _build_api_graph()
    node = runtime_graph.nodes[llm_id]
    assert "Authorization" not in node.api_spec["headers"]
    assert "key" not in node.api_spec
    serialized = node.serialize()
    assert "Authorization" not in str(serialized)
    assert "Bearer" not in str(serialized)


def test_api_missing_endpoint_fails_closed() -> None:
    """API mode without a URL fails at build time with a clear message."""
    with pytest.raises(ValueError, match="requires an endpoint"):
        _build_api_graph(api=ApiConfig(url="", credential_env="OPENAI_API_KEY"))


def test_api_missing_credential_ref_fails_closed() -> None:
    """API mode without a credential reference fails at build time with a
    clear message, rather than silently sending an unauthenticated request."""
    with pytest.raises(ValueError, match="credential reference"):
        _build_api_graph(
            api=ApiConfig(url="https://api.example.com/v1/chat/completions")
        )


def test_api_invalid_credential_env_name_fails_closed() -> None:
    """A malformed credential reference (not a valid env var name) is rejected
    at build time."""
    with pytest.raises(ValueError, match="not a valid env var name"):
        _build_api_graph(
            api=ApiConfig(
                url="https://api.example.com/v1/chat/completions",
                credential_env="1BAD NAME",
            )
        )


def test_api_dynamic_column_fails_closed() -> None:
    """A message referencing an upstream node's runtime output cannot be
    rendered at build time and must fail rather than silently fall back."""
    stock = input_placeholder("Stock")
    retrieval = DataRetrievalOp(
        data_spec={
            "type": "lumid",
            "mode": "sql",
            "template": "SELECT * FROM t WHERE symbol = :symbol",
            "params": [{"name": "symbol", "node": stock.id}],
        },
        inputs=[stock],
    )
    llm = LLMChatOp(
        [OpMessage(role="user", content=retrieval)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(
                url="https://api.example.com/v1/chat/completions",
                credential_env="OPENAI_API_KEY",
            ),
        ),
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    with pytest.raises(ValueError, match="cannot render message"):
        RuntimeGraphBuilder().build(compiled)
