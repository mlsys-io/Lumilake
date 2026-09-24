import textwrap
from typing import Any

import pytest
from lumilake import envs

from lumilake_server.common import ApiConfig, GenerationConfig, Message
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
    DataRetrievalOp,
    EmbeddingOp,
    FormatOp,
    ImageGenerationOp,
    LambdaOp,
    LLMChatOp,
    LLMVisionOp,
    OpMessage,
    as_output,
    input_placeholder,
)
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.optimizer.halo import HaloOptimizer
from lumilake_server.runtime.runtime_graph import (
    _API_CREDENTIAL_PLACEHOLDER,
    RuntimeGraph,
    RuntimeGraphBuilder,
    make_node_prefix,
)
from lumilake_server.runtime.runtime_ops import RuntimeOp

_LUMID_URL = "http://lumid-data"
_LUMID_TOKEN = "test-token"
_RUNTIME_TOKEN = "test" + "-pat"
_DEFAULT_API_URL = "https://lum.id/llm/v1/chat/completions"


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", _LUMID_URL)
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", _LUMID_TOKEN)
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", _RUNTIME_TOKEN)


def _build_api_graph(**config_kwargs: Any) -> tuple[RuntimeGraph, str]:
    stock = input_placeholder("Stock")
    api = config_kwargs.pop("api", None) or ApiConfig()
    model = config_kwargs.pop("model", "meta-llama/Llama-3.1-8B-Instruct")
    cfg = GenerationConfig(
        model=model,
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
    assert api["url"] == _DEFAULT_API_URL
    assert "auth" not in api
    assert api["headers"]["Authorization"] == _API_CREDENTIAL_PLACEHOLDER
    body = api["json"]
    assert body["model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert body["messages"] == "{{prompt}}"
    assert "api" not in body


def test_api_config_model_override() -> None:
    runtime_graph, llm_id = _build_api_graph(
        api=ApiConfig(
            url="https://api.example.com/v1/chat/completions",
            model="gpt-4o",
            authorization="Bearer caller-key",
        )
    )
    node = runtime_graph.nodes[llm_id]
    body = node.api_spec["json"]
    assert body["model"] == "gpt-4o"
    assert node.model == "gpt-4o"


def test_api_config_url_override() -> None:
    runtime_graph, llm_id = _build_api_graph(
        api=ApiConfig(
            url="https://api.example.com/v1/chat/completions",
            authorization="Bearer caller-key",
        )
    )
    node = runtime_graph.nodes[llm_id]
    assert node.api_spec["url"] == "https://api.example.com/v1/chat/completions"


def test_api_config_timeout_sec_reaches_emitted_spec() -> None:
    runtime_graph, llm_id = _build_api_graph(
        api=ApiConfig(timeout_sec=120.0),
    )
    node = runtime_graph.nodes[llm_id]
    assert node.api_spec["timeout_sec"] == 120.0


def test_api_config_unset_timeout_emits_no_key() -> None:
    runtime_graph, llm_id = _build_api_graph(api=ApiConfig())
    node = runtime_graph.nodes[llm_id]
    assert "timeout_sec" not in node.api_spec


def test_api_samplers_flow_into_body() -> None:
    runtime_graph, llm_id = _build_api_graph(max_tokens=64, temperature=0.5)
    node = runtime_graph.nodes[llm_id]
    body = node.api_spec["json"]
    assert body["max_tokens"] == 64
    assert body["temperature"] == 0.5


def test_api_key_redacted_on_serialization() -> None:
    """The Authorization header carries only the constant placeholder in the
    graph; the real credential never appears in the serialized spec."""
    runtime_graph, llm_id = _build_api_graph()
    node = runtime_graph.nodes[llm_id]

    assert node.api_spec["headers"]["Authorization"] == _API_CREDENTIAL_PLACEHOLDER

    assert _RUNTIME_TOKEN not in str(runtime_graph.serialize())
    assert "Bearer" not in str(runtime_graph.serialize())


def test_api_missing_pat_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """API mode against a trusted endpoint without a PAT fails at build time."""
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", None)
    with pytest.raises(ValueError, match="LUMILAKE_RUNTIME_TOKEN"):
        _build_api_graph()


def test_api_untrusted_origin_with_caller_credential_passes_through() -> None:
    runtime_graph, llm_id = _build_api_graph(
        api=ApiConfig(
            url="https://api.example.com/v1/chat/completions",
            authorization="Bearer caller-key",
        )
    )
    node = runtime_graph.nodes[llm_id]
    assert node.api_spec["headers"]["Authorization"] == _API_CREDENTIAL_PLACEHOLDER
    assert "caller-key" not in str(runtime_graph.serialize())


def test_api_untrusted_origin_without_credential_fails_closed() -> None:
    with pytest.raises(ValueError, match="untrusted endpoint"):
        _build_api_graph(
            api=ApiConfig(url="https://api.example.com/v1/chat/completions")
        )


def test_api_lookalike_host_not_trusted() -> None:
    with pytest.raises(ValueError, match="untrusted endpoint"):
        _build_api_graph(
            api=ApiConfig(url="https://lum.id.attacker.example/v1/chat/completions")
        )


def test_api_dynamic_column_renders_as_dispatch_placeholder() -> None:
    """A message referencing an upstream node's runtime output renders as a
    FlowMesh ``${node.path}`` dispatch-time placeholder and declares the node
    as a dependency."""
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
            api=ApiConfig(),
        ),
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    node = runtime_graph.nodes[llm.id]
    assert node.api_spec["json"]["messages"] == "{{prompt}}"
    assert node.dependencies == (retrieval.id,)


def test_api_node_downstream_of_api_node_receives_upstream_placeholder() -> None:
    """An API-mode LLMChatOp consuming another API-mode LLMChatOp's output
    (relayed through a ``FormatOp``) renders ``{{prompt}}`` in the request body
    and declares the upstream node as a dependency."""
    stock = input_placeholder("Stock")
    first = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
    )
    relay = FormatOp("{prior}", prior=first)
    second = LLMChatOp(
        [OpMessage(role="user", content=relay)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
    )
    output = as_output("result", second)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (first_row_id,) = runtime_graph.dsl_to_runtime[first.id]
    (second_row_id,) = runtime_graph.dsl_to_runtime[second.id]
    second_node = runtime_graph.nodes[second_row_id]
    assert second_node.api_spec["json"]["messages"] == "{{prompt}}"
    assert second_node.dependencies == (first_row_id,)


def test_node_prefix_remaps_api_placeholder_stage_name() -> None:
    """``with_node_prefix`` must rewrite the ``${node.path}`` placeholder an API
    node emits for an upstream reference to the prefixed node name, or dispatch
    fails with ``Unknown stage reference``."""
    stock = input_placeholder("Stock")
    first = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
    )
    relay = FormatOp("{prior}", prior=first)
    second = LLMChatOp(
        [OpMessage(role="user", content=relay)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
    )
    output = as_output("result", second)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    unprefixed = RuntimeGraphBuilder().build(compiled)
    prefix = make_node_prefix("job1")
    prefixed = RuntimeGraphBuilder().build(compiled, node_prefix="job1")

    (first_row_id,) = unprefixed.dsl_to_runtime[first.id]
    (second_row_id,) = unprefixed.dsl_to_runtime[second.id]
    prefixed_first = f"{prefix}__{first_row_id}"
    prefixed_second = f"{prefix}__{second_row_id}"
    assert prefixed.nodes[prefixed_second].api_spec["json"]["messages"] == "{{prompt}}"
    assert prefixed.nodes[prefixed_second].dependencies == (prefixed_first,)


def test_node_prefix_preserves_literal_placeholder_in_user_content() -> None:
    """``with_node_prefix`` rewrites only placeholders whose node is a
    dependency of the op; a ``${...}`` in message content that references a
    node outside the dependency set (here ``sibling``) is left untouched."""
    stock = input_placeholder("Stock")
    first = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
    )
    relay = FormatOp("{prior}", prior=first)
    sibling = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
    )
    sibling_literal = f"${{{sibling.id}.text}}"
    second = LLMChatOp(
        [
            OpMessage(role="system", content=sibling_literal),
            OpMessage(role="user", content=relay),
        ],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
    )
    output = as_output("result", second)
    compiled = Graph.from_ops([output, as_output("sib", sibling)]).compile(
        Stock=["NVDA"]
    )

    prefixed = RuntimeGraphBuilder().build(compiled, node_prefix="job1")

    (second_row_id,) = prefixed.dsl_to_runtime[second.id]
    node = prefixed.nodes[second_row_id]
    assert node.api_spec["json"]["messages"] == "{{prompt}}"
    template_messages = node.data_spec["template"]["options"]["format"]["messages"]
    assert {"role": "system", "content": sibling_literal} in template_messages


def test_local_node_downstream_of_api_node_uses_text_path() -> None:
    """A local-backend LLMChatOp consuming an API-backed ancestor's output
    (relayed through a ``FormatOp``) reads the API item path, not
    ``items.output``."""
    stock = input_placeholder("Stock")
    api_node = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
    )
    relay = FormatOp("{prior}", prior=api_node)
    local_node = LLMChatOp(
        [OpMessage(role="user", content=relay)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
    )
    output = as_output("result", local_node)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (api_row_id,) = runtime_graph.dsl_to_runtime[api_node.id]
    (local_row_id,) = runtime_graph.dsl_to_runtime[local_node.id]
    columns = runtime_graph.nodes[local_row_id].data_spec["template"]["columns"]
    upstream_columns = [col for col in columns if col.get("node") == api_row_id]
    assert any(
        col.get("path") == "items.json.choices[0].message.content"
        for col in upstream_columns
    )
    assert all(col.get("path") != "items.output" for col in upstream_columns)


def test_api_structural_outputs_flow_into_request_body() -> None:
    """An API-backed LLMChatOp must emit ``structural_outputs`` into the request
    body, matching the local backend's ``inference.templates``."""
    stock = input_placeholder("Stock")
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        structural_outputs=[{"name": "code", "type": "string"}],
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    body = runtime_graph.nodes[llm.id].api_spec["json"]
    assert body["templates"] == [{"name": "code", "type": "string"}]


def test_embedding_op_downstream_of_api_node_uses_text_path() -> None:
    """An EmbeddingOp consuming an API-backed ancestor's output must resolve
    the API item path, not ``items.output``."""
    stock = input_placeholder("Stock")
    api_node = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
    )
    embed = EmbeddingOp(api_node, config=GenerationConfig(model="embed-model"))
    output = as_output("result", embed)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (api_row_id,) = runtime_graph.dsl_to_runtime[api_node.id]
    (embed_row_id,) = runtime_graph.dsl_to_runtime[embed.id]
    data_spec = runtime_graph.nodes[embed_row_id].data_spec
    assert data_spec["node"] == api_row_id
    assert data_spec["path"] == "items.json.choices[0].message.content"


def test_image_generation_op_downstream_of_api_node_uses_text_path() -> None:
    """An ImageGenerationOp consuming an API-backed ancestor's output must
    resolve the API item path instead of ``items.output``."""
    stock = input_placeholder("Stock")
    api_node = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
    )
    image = ImageGenerationOp(api_node, config=GenerationConfig(model="img-model"))
    output = as_output("result", image)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (api_row_id,) = runtime_graph.dsl_to_runtime[api_node.id]
    (image_row_id,) = runtime_graph.dsl_to_runtime[image.id]
    data_spec = runtime_graph.nodes[image_row_id].data_spec
    assert data_spec["node"] == api_row_id
    assert data_spec["path"] == "items.json.choices[0].message.content"


def test_api_lambda_op_message_input_renders_literal() -> None:
    """A LambdaOp message input must render in API mode: the server evaluates
    the function at build time when its inputs are literal."""
    stock = input_placeholder("Stock")
    greeting = FormatOp("Hello, {name}!", name=stock)

    def _shout(inputs: tuple[str | list[Message], ...]) -> str:
        (greeting_text,) = inputs
        assert isinstance(greeting_text, str)
        return greeting_text.upper()

    shout = LambdaOp([greeting], fn=_shout)
    llm = LLMChatOp(
        [OpMessage(role="user", content=shout)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    node = runtime_graph.nodes[llm.id]
    assert node.api_spec["json"]["messages"] == "{{prompt}}"
    columns = node.data_spec["template"]["columns"]
    lambda_cols = [col for col in columns if col.get("label", "").startswith("lambda_")]
    assert lambda_cols and lambda_cols[0]["data"]["items"] == ["HELLO, NVDA!"]


def test_api_lambda_over_runtime_output_builds() -> None:
    """A Lambda message transform over a runtime output must build in API mode:
    the transform renders as a graph_template function step in the data_spec,
    and the executor substitutes the rendered messages into ``{{prompt}}``."""
    stock = input_placeholder("Stock")
    local = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
    )
    relay = FormatOp("{prior}", prior=local)
    shout = LambdaOp(
        [relay],
        fn=lambda inputs: str(inputs[0]).upper(),
    )
    llm = LLMChatOp(
        [OpMessage(role="user", content=shout)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    node = runtime_graph.nodes[llm.id]
    assert node.api_spec["json"]["messages"] == "{{prompt}}"
    assert node.dependencies == (local.id,)
    steps = node.data_spec["template"]["options"]["format"]["steps"]
    assert any("function" in step for step in steps)


def test_api_rowwise_template_emits_single_task() -> None:
    """An API-backed LLMChatOp with ``rowwise_template``/``rowwise_columns``/
    ``system_messages`` must mirror the local rowwise contract: one runtime
    task whose data_spec is a dataframe, with rows resolved by the executor.
    The rowwise builder carries its own copy of the ``timeout_sec`` and
    ``resolved_model`` emission, so both are asserted."""
    stock = input_placeholder("Stock")
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(timeout_sec=300.0, model="gpt-4o"),
        ),
        rowwise_template="Summarize {Stock}.",
        rowwise_columns=[
            {"label": "Stock", "data": {"type": "list", "items": ["NVDA", "AAPL"]}}
        ],
        system_messages=["You are concise."],
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA", "AAPL"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (row_id,) = runtime_graph.dsl_to_runtime[llm.id]
    assert row_id == llm.id
    node = runtime_graph.nodes[row_id]
    assert node.data_spec == {
        "type": "dataframe",
        "columns": [
            {"label": "Stock", "data": {"type": "list", "items": ["NVDA", "AAPL"]}}
        ],
        "messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Summarize {Stock}."},
        ],
    }
    assert node.api_spec["json"]["messages"] == "{{prompt}}"
    assert node.api_spec["timeout_sec"] == 300.0
    assert node.api_spec["json"]["model"] == "gpt-4o"
    assert node.model == "gpt-4o"


def test_api_aggregate_table_renders_df_column() -> None:
    """An API-backed LLMChatOp with ``aggregate_table`` must mirror the local
    aggregate contract: the base template columns are merged with a ``df``
    dataframe column built from ``aggregate_table``. The aggregate builder
    carries its own copy of the ``timeout_sec`` and ``resolved_model``
    emission, so both are asserted here too."""
    stock = input_placeholder("Stock")
    upstream = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
    )
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(timeout_sec=300.0, model="gpt-4o"),
        ),
        aggregate_table=[
            {"label": "summary", "node": upstream.id, "path": "items.output"}
        ],
    )
    llm.inputs.append(upstream)
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (llm_row_id,) = runtime_graph.dsl_to_runtime[llm.id]
    node = runtime_graph.nodes[llm_row_id]
    assert upstream.id in node.dependencies
    template = node.data_spec["template"]
    df_col = next(c for c in template["columns"] if c.get("label") == "df")
    assert df_col["data"]["type"] == "dataframe"
    assert df_col["data"]["columns"] == [
        {
            "label": "summary",
            "node": upstream.id,
            "path": "items.json.choices[0].message.content",
        }
    ]
    assert node.api_spec["json"]["messages"] == "{{prompt}}"
    assert node.api_spec["timeout_sec"] == 300.0
    assert node.api_spec["json"]["model"] == "gpt-4o"
    assert node.model == "gpt-4o"


def test_api_aggregate_df_prompt_renders_into_request_body() -> None:
    """An aggregate API op whose message is a ``FormatOp`` template containing
    ``{df}`` must mirror the local aggregate contract: the dataframe renders in
    the data_spec and the request body carries ``{{prompt}}``."""
    stock = input_placeholder("Stock")
    upstream = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
    )
    digest = FormatOp("Summarize the table:\n{df}", df=upstream)
    llm = LLMChatOp(
        [OpMessage(role="user", content=digest)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        aggregate_table=[
            {"label": "summary", "node": upstream.id, "path": "items.output"}
        ],
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (llm_row_id,) = runtime_graph.dsl_to_runtime[llm.id]
    node = runtime_graph.nodes[llm_row_id]
    assert node.api_spec["json"]["messages"] == "{{prompt}}"
    template = node.data_spec["template"]
    df_col = next(c for c in template["columns"] if c.get("label") == "df")
    assert df_col["data"]["type"] == "dataframe"


def test_api_aggregate_op_emits_condition() -> None:
    """An aggregate API LLMChatOp must emit ``condition`` on its runtime node,
    matching the local aggregate builder."""
    stock = input_placeholder("Stock")
    upstream = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
    )
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        aggregate_table=[
            {"label": "summary", "node": upstream.id, "path": "items.output"}
        ],
        condition={"node": "gate", "expr": "gate == 'on'"},
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (row_id,) = runtime_graph.dsl_to_runtime[llm.id]
    assert runtime_graph.nodes[row_id].condition == {
        "node": "gate",
        "expr": "gate == 'on'",
    }


def test_api_ancestor_with_return_history_feeds_local_downstream() -> None:
    """An API-backed ancestor with ``return_history`` feeding a local
    downstream must mirror the local history contract: the prior prompt is
    inlined as a literal column and the assistant output resolves the API item
    path."""
    stock = input_placeholder("Stock")
    api_node = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        return_history=True,
    )
    downstream = LLMChatOp(
        [OpMessage(role="user", content=api_node)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
    )
    output = as_output("result", downstream)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (api_row_id,) = runtime_graph.dsl_to_runtime[api_node.id]
    (downstream_row_id,) = runtime_graph.dsl_to_runtime[downstream.id]
    columns = runtime_graph.nodes[downstream_row_id].data_spec["template"]["columns"]
    context_cols = [c for c in columns if c.get("label") == f"{api_row_id}_context"]
    assert context_cols and context_cols[0]["data"]["items"] == ["NVDA"]
    output_cols = [c for c in columns if c.get("label") == f"{api_row_id}_output"]
    assert (
        output_cols
        and output_cols[0]["path"] == "items.json.choices[0].message.content"
    )


def test_api_ancestor_with_return_history_feeds_api_downstream() -> None:
    """The same history parity must hold for an API->API edge: the prior prompt
    inlines as a literal and the assistant output resolves the API item path."""
    stock = input_placeholder("Stock")
    api_node = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        return_history=True,
    )
    downstream = LLMChatOp(
        [OpMessage(role="user", content=api_node)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
    )
    output = as_output("result", downstream)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (api_row_id,) = runtime_graph.dsl_to_runtime[api_node.id]
    (downstream_row_id,) = runtime_graph.dsl_to_runtime[downstream.id]
    node = runtime_graph.nodes[downstream_row_id]
    assert node.api_spec["json"]["messages"] == "{{prompt}}"
    columns = node.data_spec["template"]["columns"]
    context_cols = [c for c in columns if c.get("label") == f"{api_row_id}_context"]
    assert context_cols and context_cols[0]["data"]["items"] == ["NVDA"]
    output_cols = [c for c in columns if c.get("label") == f"{api_row_id}_output"]
    assert (
        output_cols
        and output_cols[0]["path"] == "items.json.choices[0].message.content"
    )


def test_api_ancestor_return_history_runtime_prior_prompt_fails_closed() -> None:
    """An API-backed ancestor with ``return_history`` whose prior prompt is
    runtime-derived must fail closed: API mode cannot inline it at build time,
    and an API task result carries no ``metadata.prompt`` at dispatch time, so
    the history cannot be reconstructed."""
    stock = input_placeholder("Stock")
    local = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
    )
    relay = FormatOp("{prior}", prior=local)
    api_node = LLMChatOp(
        [OpMessage(role="user", content=relay)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        return_history=True,
    )
    downstream = LLMChatOp(
        [OpMessage(role="user", content=api_node)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
    )
    output = as_output("result", downstream)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    with pytest.raises(ValueError, match="runtime-derived prior prompt"):
        RuntimeGraphBuilder().build(compiled)


def test_api_ancestor_return_history_multi_message_prior_fails_closed() -> None:
    """An API-backed ancestor with ``return_history`` whose prior prompt is
    more than a single user message must fail closed: API mode cannot replay
    system messages or earlier turns with their roles, so the history cannot
    be reconstructed faithfully like the local ``metadata.prompt`` path."""
    sys_in = input_placeholder("Sys")
    stock = input_placeholder("Stock")
    api_node = LLMChatOp(
        [
            OpMessage(role="system", content=sys_in),
            OpMessage(role="user", content=stock),
        ],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        return_history=True,
    )
    downstream = LLMChatOp(
        [OpMessage(role="user", content=api_node)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
    )
    output = as_output("result", downstream)
    compiled = Graph.from_ops([output]).compile(Sys=["sys1"], Stock=["NVDA"])

    with pytest.raises(ValueError, match="single user message"):
        RuntimeGraphBuilder().build(compiled)


def test_api_scheme_less_url_raises_clean_error() -> None:
    with pytest.raises(ValueError, match="no host"):
        _build_api_graph(api=ApiConfig(url="lum.id/llm/v1/chat/completions"))


def test_api_explicit_default_port_is_trusted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMILAKE_API_TRUSTED_ORIGINS", "https://lum.id")
    runtime_graph, llm_id = _build_api_graph(
        api=ApiConfig(url="https://lum.id:443/v1/chat/completions")
    )
    node = runtime_graph.nodes[llm_id]
    assert node.api_spec["headers"]["Authorization"] == _API_CREDENTIAL_PLACEHOLDER


def test_api_trusted_origins_env_var_is_additive_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LUMILAKE_API_TRUSTED_ORIGINS extends the trusted-origin allowlist; it
    must not replace the always-trusted default https://lum.id."""
    monkeypatch.setattr(
        envs, "LUMILAKE_API_TRUSTED_ORIGINS", "https://vendor.example.com"
    )

    default_graph, default_llm_id = _build_api_graph(api=ApiConfig())
    default_node = default_graph.nodes[default_llm_id]
    assert (
        default_node.api_spec["headers"]["Authorization"] == _API_CREDENTIAL_PLACEHOLDER
    )

    vendor_graph, vendor_llm_id = _build_api_graph(
        api=ApiConfig(url="https://vendor.example.com/v1/chat/completions")
    )
    vendor_node = vendor_graph.nodes[vendor_llm_id]
    assert (
        vendor_node.api_spec["headers"]["Authorization"] == _API_CREDENTIAL_PLACEHOLDER
    )


def test_api_multi_row_input_emits_single_task() -> None:
    """A literal message column with N rows must emit one runtime task: rows are
    resolved by the executor from the upstream result."""
    stock = input_placeholder("Stock")
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct", api=ApiConfig()
        ),
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA", "AAPL"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)

    assert runtime_graph.dsl_to_runtime[llm.id] == [llm.id]
    assert set(runtime_graph.nodes) == {llm.id}
    assert runtime_graph.nodes[llm.id].api_spec["json"]["messages"] == "{{prompt}}"
    assert runtime_graph.output_node_map[llm.id] == "result"


def test_retrieval_param_referencing_single_row_node_builds() -> None:
    """A DataRetrievalOp template param referencing a single-row upstream must
    build: the retrieval param guard must not over-reject a single-row node."""
    stock = input_placeholder("Stock")
    single = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
    )
    retrieval = DataRetrievalOp(
        data_spec={
            "type": "lumid",
            "mode": "sql",
            "template": "SELECT * FROM t WHERE x = :p",
            "params": [{"name": "p", "node": single.id, "path": "items.output"}],
        },
        inputs=[single],
    )
    output = as_output("result", retrieval)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    assert runtime_graph.nodes[retrieval.id].task_type == "data_retrieval"


def test_node_ref_to_single_row_producer_builds() -> None:
    """A node reference to a genuinely single-row producer must build: the
    multi-row node-ref guard must not over-reject a single-row upstream."""
    stock = input_placeholder("Stock")
    single = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
    )
    emb = EmbeddingOp(content=single, config=GenerationConfig(model="bge-m3"))
    output = as_output("result", emb)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    assert runtime_graph.nodes[emb.id].task_type == "embedding"


def test_api_node_consuming_fanned_api_upstream_emits_single_task() -> None:
    """An API-mode LLMChatOp consuming a multi-row API upstream must emit one
    runtime task: the upstream is a single node and the executor resolves rows
    at run time."""
    stock = input_placeholder("Stock")
    draft = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct", api=ApiConfig()
        ),
    )
    relay = FormatOp("{prior}", prior=draft)
    polish = LLMChatOp(
        [OpMessage(role="user", content=relay)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct", api=ApiConfig()
        ),
    )
    output = as_output("result", polish)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA", "AAPL"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (draft_row,) = runtime_graph.dsl_to_runtime[draft.id]
    (polish_row,) = runtime_graph.dsl_to_runtime[polish.id]
    assert draft_row == draft.id
    assert polish_row == polish.id
    assert runtime_graph.nodes[polish_row].api_spec["json"]["messages"] == "{{prompt}}"
    assert runtime_graph.nodes[polish_row].dependencies == (draft_row,)


def test_api_node_consuming_return_history_upstream_builds() -> None:
    """An API-mode LLMChatOp with ``return_history`` consuming a multi-row API
    upstream must build: the upstream is a single node and rows are resolved by
    the executor."""
    stock = input_placeholder("Stock")
    draft = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct", api=ApiConfig()
        ),
        return_history=True,
    )
    polish = LLMChatOp(
        [OpMessage(role="user", content=draft)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct", api=ApiConfig()
        ),
    )
    output = as_output("result", polish)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA", "AAPL"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (draft_row,) = runtime_graph.dsl_to_runtime[draft.id]
    (polish_row,) = runtime_graph.dsl_to_runtime[polish.id]
    assert runtime_graph.nodes[polish_row].api_spec["json"]["messages"] == "{{prompt}}"
    assert runtime_graph.nodes[polish_row].dependencies == (draft_row,)


_YAML_SINGLE_HOP_TWO_ROWS = textwrap.dedent(
    """
    name: yaml-api-two-rows

    inputs:
      Topic:
        - "a database index"
        - "a message queue"

    ops:
      - id: "Summarise"
        op: LLMChatOp
        inputs: [Topic]
        messages:
          - role: system
            content: "Answer in one short sentence."
          - role: user
            content: "Topic"
        config:
          model: meta-llama/Llama-3.1-8B-Instruct
          api:
            url: https://lum.id/llm/v1/chat/completions

    outputs:
      - name: result
        ref: "Summarise"
    """
)


def _build_yaml_two_row_graph() -> tuple[RuntimeGraph, str]:
    specs = parse_yaml_payload(_YAML_SINGLE_HOP_TWO_ROWS)
    spec = specs["yaml-api-two-rows"]
    graph = Graph.from_json(spec["graph"])
    compiled = graph.compile(**spec["inputs"])
    llm_id = next(
        op_id
        for op_id, op_dict in spec["graph"].items()
        if op_dict.get("_op") == "LLMChatOp"
    )
    return RuntimeGraphBuilder().build(compiled), llm_id


def test_yaml_wrapped_bare_reference_emits_single_task() -> None:
    """A bare ``content: "Topic"`` reference in YAML is implicitly wrapped into
    a FormatOp step by the parser; two input rows must still emit one API task,
    with rows resolved by the executor."""
    runtime_graph, llm_id = _build_yaml_two_row_graph()

    row_ids = runtime_graph.dsl_to_runtime[llm_id]
    assert row_ids == [llm_id]

    assert runtime_graph.nodes[llm_id].api_spec["json"]["messages"] == "{{prompt}}"
    assert runtime_graph.output_node_map[llm_id] == "result"


def test_merged_workflow_result_stays_single_task_after_optimize() -> None:
    """The merged/optimized graph must keep the single API output node; two
    input rows resolve to one output entry."""
    runtime_graph, llm_id = _build_yaml_two_row_graph()

    optimized_graph, output_mapping = HaloOptimizer().optimize_graphs(
        {"yaml-api-two-rows": runtime_graph}
    )

    row_ids = runtime_graph.dsl_to_runtime[llm_id]
    assert len(row_ids) == 1

    result_nodes = [
        node_id
        for node_id, name in optimized_graph.output_node_map.items()
        if name == "result"
    ]
    assert sorted(result_nodes) == sorted(row_ids)
    for node_id in result_nodes:
        assert output_mapping[node_id] == ("yaml-api-two-rows", "result")

    assert optimized_graph.nodes[llm_id].api_spec["json"]["messages"] == "{{prompt}}"


_YAML_API_NODE_FEEDS_LOCAL_NODE = textwrap.dedent(
    """
    name: api-node-feeds-local-node

    inputs:
      Topic:
        - "a database index"
        - "a message queue"

    ops:
      - id: "Summarise"
        op: LLMChatOp
        inputs: [Topic]
        messages:
          - role: system
            content: "Answer in one short sentence."
          - role: user
            content: "Topic"
        config:
          model: model-a
          api:
            url: https://lum.id/llm/v1/chat/completions

      - id: "Critique"
        op: LLMChatOp
        inputs: ["Summarise"]
        messages:
          - role: system
            content: "Reply with one word."
          - role: user
            content: "Summarise"
        config:
          model: meta-llama/Llama-3.1-8B-Instruct

    outputs:
      - name: result
        ref: "Critique"
    """
)


def test_api_node_feeding_local_node_builds() -> None:
    """A local (non-API) LLM node consuming an API node's output must build: the
    API upstream is a single node and the local consumer reads its item path."""
    specs = parse_yaml_payload(_YAML_API_NODE_FEEDS_LOCAL_NODE)
    spec = specs["api-node-feeds-local-node"]
    graph = Graph.from_json(spec["graph"])
    compiled = graph.compile(**spec["inputs"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)
    local_node = next(
        node for node in runtime_graph.nodes.values() if node.task_type == "inference"
    )
    columns = local_node.data_spec["template"]["columns"]
    upstream_cols = [c for c in columns if c.get("node") is not None]
    assert upstream_cols
    assert upstream_cols[0]["path"] == "items.json.choices[0].message.content"


_YAML_LOCAL_NODE_FEEDS_API_NODE = textwrap.dedent(
    """
    name: local-node-feeds-api-node

    inputs:
      Topic:
        - "a database index"
        - "a message queue"

    ops:
      - id: "Local"
        op: LLMChatOp
        inputs: [Topic]
        messages:
          - role: user
            content: "Topic"
        config:
          model: model-a

      - id: "Api"
        op: LLMChatOp
        inputs: [Local]
        messages:
          - role: user
            content: "Local"
        config:
          model: model-b
          api:
            url: https://lum.id/llm/v1/chat/completions

    outputs:
      - name: result
        ref: "Api"
    """
)


def test_api_node_feeding_local_multi_row_upstream_fails_closed() -> None:
    """An API node cannot consume a local upstream that produces multiple
    rows: API mode message columns can only carry one row per node reference,
    so wiring this downstream node to the single unsuffixed output would
    silently drop every row but the first. The graph build fails closed
    instead."""
    specs = parse_yaml_payload(_YAML_LOCAL_NODE_FEEDS_API_NODE)
    spec = specs["local-node-feeds-api-node"]
    graph = Graph.from_json(spec["graph"])
    compiled = graph.compile(**spec["inputs"])

    with pytest.raises(ValueError, match="produces multiple rows"):
        RuntimeGraphBuilder().build(compiled)


def test_api_node_feeding_single_row_local_upstream_builds() -> None:
    """An API node consuming a local upstream that produces a single row must
    build, reading the local ``items.output`` path."""
    specs = parse_yaml_payload(_YAML_LOCAL_NODE_FEEDS_API_NODE)
    spec = specs["local-node-feeds-api-node"]
    graph = Graph.from_json(spec["graph"])
    compiled = graph.compile(Topic=["a database index"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)
    api_node = next(
        node for node in runtime_graph.nodes.values() if node.task_type == "api"
    )
    assert api_node.api_spec["json"]["messages"] == "{{prompt}}"
    columns = api_node.data_spec["template"]["columns"]
    upstream_cols = [c for c in columns if c.get("node") is not None]
    assert upstream_cols and upstream_cols[0]["path"] == "items.output"


def test_api_aggregate_node_feeding_local_multi_row_upstream_builds() -> None:
    """An aggregate API op whose ``aggregate_table`` references a multi-row
    local upstream must build: the upstream is a single node and rows are
    resolved by the executor."""
    stock = input_placeholder("Stock")
    local = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
    )
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        aggregate_table=[
            {"label": "summary", "node": local.id, "path": "items.output"}
        ],
    )
    llm.inputs.append(local)
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA", "AAPL"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)
    (llm_row,) = runtime_graph.dsl_to_runtime[llm.id]
    node = runtime_graph.nodes[llm_row]
    assert node.api_spec["json"]["messages"] == "{{prompt}}"
    assert node.dependencies == (local.id,)


def test_api_aggregate_node_feeding_fanned_api_upstream_builds() -> None:
    """An aggregate API op whose ``aggregate_table`` references a rowwise API
    upstream must build: the upstream is a single node and rows are resolved by
    the executor."""
    stock = input_placeholder("Stock")
    api_up = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        rowwise_template="Summarize {S}.",
        rowwise_columns=[{"label": "S", "data": {"type": "list", "items": ["a", "b"]}}],
    )
    llm = LLMChatOp(
        [OpMessage(role="user", content="Summarize the table.")],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        aggregate_table=[
            {"label": "summary", "node": api_up.id, "path": "items.output"}
        ],
    )
    llm.inputs.append(api_up)
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)
    (llm_row,) = runtime_graph.dsl_to_runtime[llm.id]
    node = runtime_graph.nodes[llm_row]
    assert node.api_spec["json"]["messages"] == "{{prompt}}"
    assert node.dependencies == (api_up.id,)
    df_col = next(
        c for c in node.data_spec["template"]["columns"] if c.get("label") == "df"
    )
    assert df_col["data"]["columns"][0]["path"] == (
        "items.rows.json.choices[0].message.content"
    )


def test_api_rowwise_node_feeding_fanned_api_upstream_builds() -> None:
    """A rowwise API op whose node-ref column references a rowwise API upstream
    must build, reading the upstream at the row-wise API path."""
    stock = input_placeholder("Stock")
    api_up = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        rowwise_template="Summarize {S}.",
        rowwise_columns=[{"label": "S", "data": {"type": "list", "items": ["a", "b"]}}],
    )
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        rowwise_template="Summarize {Prior}.",
        rowwise_columns=[{"label": "Prior", "node": api_up.id, "path": "items.output"}],
    )
    llm.inputs.append(api_up)
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)
    (llm_row,) = runtime_graph.dsl_to_runtime[llm.id]
    node = runtime_graph.nodes[llm_row]
    assert node.api_spec["json"]["messages"] == "{{prompt}}"
    assert node.dependencies == (api_up.id,)
    assert node.data_spec["columns"][0]["path"] == (
        "items.rows.json.choices[0].message.content"
    )


def test_api_rowwise_node_ref_source_mapping_is_authoritative() -> None:
    """A rowwise API consumer whose node-ref column references an already-built
    API source with a ``node:`` column must use the runtime mapping (one node),
    reading the source at the row-wise API path."""
    stock = input_placeholder("Stock")
    src = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        rowwise_template="Summarize {Prior}.",
        rowwise_columns=[{"label": "Prior", "node": stock.id, "path": "items.output"}],
    )
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        rowwise_template="Summarize {Prior}.",
        rowwise_columns=[{"label": "Prior", "node": src.id, "path": "items.output"}],
    )
    llm.inputs.append(src)
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA", "AAPL"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (src_row,) = runtime_graph.dsl_to_runtime[src.id]
    (llm_row,) = runtime_graph.dsl_to_runtime[llm.id]
    assert src_row == src.id
    assert llm_row == llm.id
    node = runtime_graph.nodes[llm_row]
    assert node.dependencies == (src_row,)
    assert node.data_spec["columns"][0]["path"] == (
        "items.rows.json.choices[0].message.content"
    )


def test_api_node_consuming_upstream_feeding_local_vlm_builds() -> None:
    """A local VLM consumer of an API upstream must build: the API upstream is a
    single node and the VLM reads its item path."""
    stock = input_placeholder("Stock")
    draft = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct", api=ApiConfig()
        ),
    )
    relay = FormatOp("{prior}", prior=draft)
    vision = LLMVisionOp(
        [OpMessage(role="user", content=relay)],
        image_source=stock.id,
        image_source_op=stock,
        config=GenerationConfig(model="llava-hf/llava-1.5-7b-hf", api=ApiConfig()),
    )
    output = as_output("result", vision)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA", "AAPL"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)
    (draft_row,) = runtime_graph.dsl_to_runtime[draft.id]
    vision_rows = runtime_graph.dsl_to_runtime[vision.id]
    vision_node = runtime_graph.nodes[vision_rows[-1]]
    assert draft_row in vision_node.dependencies
    columns = vision_node.data_spec["template"]["columns"]
    upstream_cols = [c for c in columns if c.get("node") == draft_row]
    assert upstream_cols
    assert upstream_cols[0]["path"] == "items.json.choices[0].message.content"


def test_api_node_consuming_local_rowwise_literal_upstream_fails_closed() -> None:
    """An API-mode consumer cannot reference a local rowwise LLMChatOp whose
    literal ``rowwise_columns`` produce multiple rows from one node."""
    stock = input_placeholder("Stock")
    local_rw = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
        rowwise_template="Summarize {S}.",
        rowwise_columns=[{"label": "S", "data": {"type": "list", "items": ["a", "b"]}}],
    )
    relay = FormatOp("{prior}", prior=local_rw)
    api = LLMChatOp(
        [OpMessage(role="user", content=relay)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct", api=ApiConfig()
        ),
    )
    output = as_output("result", api)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    with pytest.raises(ValueError, match="produces multiple rows"):
        RuntimeGraphBuilder().build(compiled)


def test_embedding_op_consuming_fanned_api_upstream_fails_closed() -> None:
    """An EmbeddingOp consuming a fanned API producer must fail closed: it binds
    a single ``${node.path}`` reference to the unsuffixed row-0 node, silently
    dropping every row but the first."""
    stock = input_placeholder("Stock")
    draft = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct", api=ApiConfig()
        ),
    )
    emb = EmbeddingOp(content=draft, config=GenerationConfig(model="bge-m3"))
    output = as_output("result", emb)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA", "AAPL"])

    with pytest.raises(ValueError, match="produces multiple rows"):
        RuntimeGraphBuilder().build(compiled)


def test_image_generation_op_consuming_fanned_api_upstream_fails_closed() -> None:
    """An ImageGenerationOp consuming a fanned API producer must fail closed: it
    binds a single ``${node.path}`` reference to the unsuffixed row-0 node,
    silently dropping every row but the first."""
    stock = input_placeholder("Stock")
    draft = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct", api=ApiConfig()
        ),
    )
    relay = FormatOp("{ref0}", draft)
    img = ImageGenerationOp(content=relay, config=GenerationConfig(model="sdxl"))
    output = as_output("result", img)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA", "AAPL"])

    with pytest.raises(ValueError, match="produces multiple rows"):
        RuntimeGraphBuilder().build(compiled)


def test_fanned_consumer_condition_fails_closed() -> None:
    """A condition on a consumer whose source is multi-row must fail closed
    when the consumer is also multi-row: a single gate cannot select which
    row's value to test."""
    stock = input_placeholder("Stock")
    gate = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        rowwise_template="Summarize {S}.",
        rowwise_columns=[{"label": "S", "data": {"type": "list", "items": ["a", "b"]}}],
    )
    relay = FormatOp("{prior}", prior=gate)
    consumer = LLMChatOp(
        [OpMessage(role="user", content=relay)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        condition={"node": gate.id, "expr": "gate == on"},
    )
    output = as_output("result", consumer)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    with pytest.raises(ValueError, match="can only gate a whole task"):
        RuntimeGraphBuilder().build(compiled)


def test_local_rowwise_node_feeding_fanned_api_upstream_builds() -> None:
    """A local rowwise LLMChatOp whose node-ref column references a rowwise API
    producer must build, reading the producer at the row-wise API path."""
    stock = input_placeholder("Stock")
    api_up = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        rowwise_template="Summarize {S}.",
        rowwise_columns=[{"label": "S", "data": {"type": "list", "items": ["a", "b"]}}],
    )
    local_rw = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
        rowwise_template="Summarize {Prior}.",
        rowwise_columns=[{"label": "Prior", "node": api_up.id, "path": "items.output"}],
    )
    local_rw.inputs.append(api_up)
    output = as_output("result", local_rw)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)
    (local_row,) = runtime_graph.dsl_to_runtime[local_rw.id]
    node = runtime_graph.nodes[local_row]
    assert node.dependencies == (api_up.id,)
    assert node.data_spec["columns"][0]["path"] == (
        "items.rows.json.choices[0].message.content"
    )


def test_local_aggregate_node_feeding_fanned_api_upstream_builds() -> None:
    """A local aggregate LLMChatOp whose ``aggregate_table`` references a
    rowwise API producer must build, reading the producer at the row-wise API
    path."""
    stock = input_placeholder("Stock")
    api_up = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        rowwise_template="Summarize {S}.",
        rowwise_columns=[{"label": "S", "data": {"type": "list", "items": ["a", "b"]}}],
    )
    local_agg = LLMChatOp(
        [OpMessage(role="user", content="Summarize the table.")],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
        aggregate_table=[
            {"label": "summary", "node": api_up.id, "path": "items.output"}
        ],
    )
    local_agg.inputs.append(api_up)
    output = as_output("result", local_agg)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)
    (local_row,) = runtime_graph.dsl_to_runtime[local_agg.id]
    node = runtime_graph.nodes[local_row]
    assert node.dependencies == (api_up.id,)
    df_col = next(
        c for c in node.data_spec["template"]["columns"] if c.get("label") == "df"
    )
    assert df_col["data"]["columns"][0]["path"] == (
        "items.rows.json.choices[0].message.content"
    )


def test_vlm_rowwise_column_feeding_fanned_api_upstream_builds() -> None:
    """A VLM rowwise column referencing a rowwise API producer must build,
    reading the producer at the row-wise API path."""
    stock = input_placeholder("Stock")
    api_up = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        rowwise_template="Summarize {S}.",
        rowwise_columns=[{"label": "S", "data": {"type": "list", "items": ["a", "b"]}}],
    )
    vision = LLMVisionOp(
        [OpMessage(role="user", content="ignored")],
        image_source=stock.id,
        image_source_op=stock,
        config=GenerationConfig(model="llava-hf/llava-1.5-7b-hf"),
        rowwise_template="Summarize {Prior}.",
        rowwise_columns=[{"label": "Prior", "node": api_up.id, "path": "items.output"}],
    )
    vision.inputs.append(api_up)
    output = as_output("result", vision)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)
    vision_rows = runtime_graph.dsl_to_runtime[vision.id]
    vision_node = runtime_graph.nodes[vision_rows[-1]]
    columns = vision_node.data_spec["template"]["columns"]
    upstream_cols = [c for c in columns if c.get("node") == api_up.id]
    assert upstream_cols
    assert upstream_cols[0]["path"] == "items.rows.json.choices[0].message.content"


def test_retrieval_param_feeding_multi_row_upstream_fails_closed() -> None:
    """A DataRetrievalOp template param referencing a multi-row upstream must
    fail closed: a retrieval param binds a single node reference, so wiring it
    to the unsuffixed output would silently drop every row but the first."""
    stock = input_placeholder("Stock")
    local_rw = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
        rowwise_template="Summarize {S}.",
        rowwise_columns=[{"label": "S", "data": {"type": "list", "items": ["a", "b"]}}],
    )
    retrieval = DataRetrievalOp(
        data_spec={
            "type": "lumid",
            "mode": "sql",
            "template": "SELECT * FROM t WHERE x = :p",
            "params": [{"name": "p", "node": local_rw.id, "path": "items.output"}],
        },
        inputs=[local_rw],
    )
    output = as_output("result", retrieval)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    with pytest.raises(ValueError, match="produces multiple rows"):
        RuntimeGraphBuilder().build(compiled)


def test_retrieval_param_feeding_multi_row_embedding_fails_closed() -> None:
    """A DataRetrievalOp template param referencing an EmbeddingOp with a
    multi-row input must fail closed: the embedding emits one output row per
    input text, so a retrieval param binding only the unsuffixed node would
    silently drop every row but the first."""
    stock = input_placeholder("Stock")
    emb = EmbeddingOp(
        content=stock,
        config=GenerationConfig(model="bge-m3", api=ApiConfig()),
    )
    retrieval = DataRetrievalOp(
        data_spec={
            "type": "lumid",
            "mode": "sql",
            "template": "SELECT * FROM t WHERE x = :p",
            "params": [{"name": "p", "node": emb.id, "path": "items.output"}],
        },
        inputs=[emb],
    )
    output = as_output("result", retrieval)
    compiled = Graph.from_ops([output]).compile(Stock=["a", "b"])

    with pytest.raises(ValueError, match="produces multiple rows"):
        RuntimeGraphBuilder().build(compiled)


def test_fanned_condition_cardinality_mismatch_fails_closed() -> None:
    """A condition on a multi-row consumer whose ``node`` is a multi-row source
    must fail closed regardless of the row counts, rather than silently gating
    on the wrong source row."""
    stock = input_placeholder("Stock")

    def build(source_rows: int, consumer_rows: int) -> None:
        gate = LLMChatOp(
            [OpMessage(role="user", content=stock)],
            config=GenerationConfig(
                model="meta-llama/Llama-3.1-8B-Instruct",
                api=ApiConfig(),
            ),
            rowwise_template="Summarize {S}.",
            rowwise_columns=[
                {
                    "label": "S",
                    "data": {
                        "type": "list",
                        "items": [str(i) for i in range(source_rows)],
                    },
                }
            ],
        )
        consumer = LLMChatOp(
            [OpMessage(role="user", content=stock)],
            config=GenerationConfig(
                model="meta-llama/Llama-3.1-8B-Instruct",
                api=ApiConfig(),
            ),
            rowwise_template="Summarize {Stock}.",
            rowwise_columns=[
                {
                    "label": "Stock",
                    "data": {
                        "type": "list",
                        "items": [str(i) for i in range(consumer_rows)],
                    },
                }
            ],
            condition={"node": gate.id, "expr": "gate == on"},
        )
        gate_out = as_output("gate_out", gate)
        consumer_out = as_output("result", consumer)
        compiled = Graph.from_ops([gate_out, consumer_out]).compile(Stock=["NVDA"])
        RuntimeGraphBuilder().build(compiled)

    with pytest.raises(ValueError, match="can only gate a whole task"):
        build(source_rows=3, consumer_rows=2)
    with pytest.raises(ValueError, match="can only gate a whole task"):
        build(source_rows=2, consumer_rows=3)


def test_one_row_consumer_of_fanned_source_builds() -> None:
    """A one-row API consumer conditioned on a two-row API source must build:
    a single-row consumer is not multi-row, so the fail-closed condition does
    not fire and the gate references the source's single node."""
    stock = input_placeholder("Stock")
    gate = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        rowwise_template="Summarize {S}.",
        rowwise_columns=[{"label": "S", "data": {"type": "list", "items": ["a", "b"]}}],
    )
    consumer = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        condition={"node": gate.id, "expr": "gate == on"},
    )
    compiled = Graph.from_ops(
        [as_output("gate_out", gate), as_output("result", consumer)]
    ).compile(Stock=["NVDA"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)
    (gate_row,) = runtime_graph.dsl_to_runtime[gate.id]
    (consumer_row,) = runtime_graph.dsl_to_runtime[consumer.id]
    assert runtime_graph.nodes[consumer_row].condition == {
        "node": gate_row,
        "expr": "gate == on",
    }


def test_condition_source_max_cardinality_across_messages_fails_closed() -> None:
    """A condition source with multi-row cardinality across its messages feeding
    a multi-row consumer must fail closed."""
    stock = input_placeholder("Stock")
    topic = input_placeholder("Topic")
    consumer = LLMChatOp(
        [OpMessage(role="user", content=topic)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        condition={"node": "gate", "expr": "gate == on"},
    )
    gate = LLMChatOp(
        [
            OpMessage(role="user", content=stock),
            OpMessage(role="user", content=topic),
        ],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
    )
    consumer.condition = {"node": gate.id, "expr": "gate == on"}
    compiled = Graph.from_ops(
        [as_output("gate_out", gate), as_output("result", consumer)]
    ).compile(Stock=["NVDA"], Topic=["x", "y"])

    with pytest.raises(ValueError, match="can only gate a whole task"):
        RuntimeGraphBuilder().build(compiled)


def test_condition_on_local_rowwise_source_fails_closed() -> None:
    """A condition on a local rowwise producer feeding a multi-row consumer must
    fail closed: both produce multiple rows, so a single gate cannot select
    which row's value to test."""
    stock = input_placeholder("Stock")
    local_rw = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
        rowwise_template="Summarize {S}.",
        rowwise_columns=[{"label": "S", "data": {"type": "list", "items": ["a", "b"]}}],
    )
    consumer = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        rowwise_template="Summarize {Stock}.",
        rowwise_columns=[
            {
                "label": "Stock",
                "data": {"type": "list", "items": ["x", "y"]},
            }
        ],
        condition={"node": local_rw.id, "expr": "gate == on"},
    )
    compiled = Graph.from_ops(
        [as_output("gate_out", local_rw), as_output("result", consumer)]
    ).compile(Stock=["NVDA"])

    with pytest.raises(ValueError, match="can only gate a whole task"):
        RuntimeGraphBuilder().build(compiled)


def test_condition_source_rowwise_node_ref_fails_closed() -> None:
    """A condition on a rowwise API source whose ``node:`` column references a
    multi-row InputOp, feeding a multi-row consumer, must fail closed."""
    stock = input_placeholder("Stock")
    gate = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        rowwise_template="Summarize {Prior}.",
        rowwise_columns=[{"label": "Prior", "node": stock.id, "path": "items.output"}],
    )
    relay = FormatOp("{prior}", prior=gate)
    consumer = LLMChatOp(
        [OpMessage(role="user", content=relay)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        condition={"node": gate.id, "expr": "gate == on"},
    )
    compiled = Graph.from_ops(
        [as_output("gate_out", gate), as_output("result", consumer)]
    ).compile(Stock=["NVDA", "AAPL"])

    with pytest.raises(ValueError, match="can only gate a whole task"):
        RuntimeGraphBuilder().build(compiled)


def test_condition_source_rowwise_node_ref_consumer_first_builds() -> None:
    """A single-row consumer built before its rowwise API condition source must
    build: a single-row consumer is not multi-row, so the fail-closed condition
    does not fire and the gate references the source's single node."""
    stock = input_placeholder("Stock")
    gate = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        rowwise_template="Summarize {Prior}.",
        rowwise_columns=[{"label": "Prior", "node": stock.id, "path": "items.output"}],
    )
    consumer = LLMChatOp(
        [OpMessage(role="user", content="hi")],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        condition={"node": gate.id, "expr": "gate == on"},
    )
    compiled = Graph.from_ops(
        [as_output("gate_out", gate), as_output("result", consumer)]
    ).compile(Stock=["NVDA", "AAPL"])

    graph_order = list(compiled.graph.as_dict().keys())
    assert graph_order.index(consumer.id) < graph_order.index(gate.id)

    runtime_graph = RuntimeGraphBuilder().build(compiled)
    (gate_row,) = runtime_graph.dsl_to_runtime[gate.id]
    (consumer_row,) = runtime_graph.dsl_to_runtime[consumer.id]
    assert runtime_graph.nodes[consumer_row].condition == {
        "node": gate_row,
        "expr": "gate == on",
    }


def test_local_consumer_of_fanned_source_builds() -> None:
    """A local (non-API) single-row consumer conditioned on a multi-row API
    source must build: a single-row consumer is not multi-row, so the
    fail-closed condition does not fire."""
    stock = input_placeholder("Stock")
    gate = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        rowwise_template="Summarize {S}.",
        rowwise_columns=[{"label": "S", "data": {"type": "list", "items": ["a", "b"]}}],
    )
    local = LLMChatOp(
        [OpMessage(role="user", content="hi")],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
        condition={"node": gate.id, "expr": "gate == on"},
    )
    compiled = Graph.from_ops(
        [as_output("gate_out", gate), as_output("result", local)]
    ).compile(Stock=["NVDA"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)
    (gate_row,) = runtime_graph.dsl_to_runtime[gate.id]
    (local_row,) = runtime_graph.dsl_to_runtime[local.id]
    assert runtime_graph.nodes[local_row].condition == {
        "node": gate_row,
        "expr": "gate == on",
    }


def test_condition_on_unfanned_vlm_source_builds() -> None:
    """A condition on an unfanned VLM source must build successfully: a VLM
    maps to ``[<id>_embedding, <id>]`` — two implementation stages of one
    logical op, not fanned rows — so it must not be treated as a two-row
    source. The consumer's condition references the VLM's logical node."""
    stock = input_placeholder("Stock")
    vlm = LLMVisionOp(
        [OpMessage(role="user", content="Describe.")],
        image_source=stock.id,
        image_source_op=stock,
        config=GenerationConfig(model="llava-hf/llava-1.5-7b-hf"),
    )
    consumer = LLMChatOp(
        [OpMessage(role="user", content="hi")],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        condition={"node": vlm.id, "expr": "vlm == on"},
    )
    compiled = Graph.from_ops(
        [as_output("vlm_out", vlm), as_output("result", consumer)]
    ).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (consumer_row,) = runtime_graph.dsl_to_runtime[consumer.id]
    assert runtime_graph.nodes[consumer_row].condition == {
        "node": vlm.id,
        "expr": "vlm == on",
    }


def test_vlm_through_format_op_feeds_api_llm() -> None:
    """A VLM -> FormatOp -> API-LLM chain must build, referencing the VLM's
    local ``items.output`` path (the LLMVisionOp exclusion must hold)."""
    stock = input_placeholder("Stock")
    vlm = LLMVisionOp(
        [OpMessage(role="user", content="Describe.")],
        image_source=stock.id,
        image_source_op=stock,
        config=GenerationConfig(
            model="llava-hf/llava-1.5-7b-hf",
            api=ApiConfig(),
        ),
    )
    relay = FormatOp("{prior}", prior=vlm)
    api = LLMChatOp(
        [OpMessage(role="user", content=relay)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
    )
    output = as_output("result", api)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    assert runtime_graph.nodes[api.id].task_type == "api"
    assert runtime_graph.nodes[api.id].api_spec["json"]["messages"] == "{{prompt}}"
    columns = runtime_graph.nodes[api.id].data_spec["template"]["columns"]
    upstream_cols = [c for c in columns if c.get("node") == vlm.id]
    assert upstream_cols and upstream_cols[0]["path"] == "items.output"


def test_condition_source_fanned_through_lambda_builds() -> None:
    """A single-row consumer conditioned on a source whose multi-row input comes
    through a LambdaOp must build: the consumer is not multi-row, so the
    fail-closed condition does not fire."""
    stock = input_placeholder("Stock")

    def _shout(inputs: tuple[str | list[Message], ...]) -> str:
        (text,) = inputs
        return str(text)

    lam = LambdaOp([stock], fn=_shout)
    src = LLMChatOp(
        [OpMessage(role="user", content=lam)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
    )
    consumer = LLMChatOp(
        [OpMessage(role="user", content="hi")],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        condition={"node": src.id, "expr": "src == on"},
    )
    compiled = Graph.from_ops(
        [as_output("src_out", src), as_output("result", consumer)]
    ).compile(Stock=["a", "b"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)
    (src_row,) = runtime_graph.dsl_to_runtime[src.id]
    (consumer_row,) = runtime_graph.dsl_to_runtime[consumer.id]
    assert runtime_graph.nodes[consumer_row].condition == {
        "node": src_row,
        "expr": "src == on",
    }


def test_row_fanned_api_nodes_distinguished_by_api_spec_in_dedupe() -> None:
    """Two API nodes that are not output-mapped (they only feed a downstream
    consumer) but share an identical data_spec must still be kept distinct by
    the optimizer's dedupe pass when their api_spec differs - dedupe must key
    on api_spec, not just data_spec."""
    shared_data_spec = {"type": "graph_template", "template": {"columns": []}}
    row0 = RuntimeOp(
        node_id="Summarise",
        task_type="api",
        backend="api",
        model="model-a",
        data_spec=shared_data_spec,
        model_spec={},
        inference_spec={},
        api_spec={
            "method": "POST",
            "url": "https://lum.id/llm/v1/chat/completions",
            "json": {"messages": [{"role": "user", "content": "a database index"}]},
        },
    )
    row1 = RuntimeOp(
        node_id="Summarise__row1",
        task_type="api",
        backend="api",
        model="model-a",
        data_spec=shared_data_spec,
        model_spec={},
        inference_spec={},
        api_spec={
            "method": "POST",
            "url": "https://lum.id/llm/v1/chat/completions",
            "json": {"messages": [{"role": "user", "content": "a message queue"}]},
        },
    )
    consumer = RuntimeOp(
        node_id="Critique",
        task_type="inference",
        backend="local",
        model="meta-llama/Llama-3.1-8B-Instruct",
        data_spec={},
        model_spec={},
        inference_spec={},
        dependencies=(row0.node_id,),
    )
    graph = RuntimeGraph(
        nodes={row0.node_id: row0, row1.node_id: row1, consumer.node_id: consumer},
        node_order=[row0.node_id, row1.node_id, consumer.node_id],
        output_node_map={consumer.node_id: "result"},
        dsl_to_runtime={
            "Summarise": [row0.node_id, row1.node_id],
            "Critique": [consumer.node_id],
        },
    )

    optimized_graph, _ = HaloOptimizer().optimize_graphs({"wf": graph})

    assert row0.node_id in optimized_graph.nodes
    assert row1.node_id in optimized_graph.nodes
    contents = [
        optimized_graph.nodes[node_id].api_spec["json"]["messages"][-1]["content"]
        for node_id in (row0.node_id, row1.node_id)
    ]
    assert contents == ["a database index", "a message queue"]


def test_api_node_ref_column_accepts_items_output_path() -> None:
    """A node-ref column on an API upstream declaring ``items.output`` must map
    to the upstream's read path (the API item path for a plain API task)."""
    stock = input_placeholder("Stock")
    upstream = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
    )
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        rowwise_template="Summarize {Prior}.",
        rowwise_columns=[
            {"label": "Prior", "node": upstream.id, "path": "items.output"}
        ],
    )
    llm.inputs.append(upstream)
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (llm_row,) = runtime_graph.dsl_to_runtime[llm.id]
    node = runtime_graph.nodes[llm_row]
    assert node.data_spec["columns"][0]["path"] == (
        "items.json.choices[0].message.content"
    )


def test_api_node_ref_column_rejects_other_paths() -> None:
    """A node-ref column on an API upstream declaring any path other than
    ``items.output`` must fail closed, naming the consumer, the column label
    and the upstream."""
    stock = input_placeholder("Stock")
    upstream = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
    )
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        rowwise_template="Summarize {Prior}.",
        rowwise_columns=[
            {"label": "Prior", "node": upstream.id, "path": "items.output.statement"}
        ],
    )
    llm.inputs.append(upstream)
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    with pytest.raises(
        ValueError,
        match=f"Column 'Prior' of consumer '{llm.id}' reads 'items.output.statement'"
        f" from API upstream '{upstream.id}'",
    ):
        RuntimeGraphBuilder().build(compiled)


def test_api_aggregate_upstream_read_at_item_path() -> None:
    """An aggregate API op feeding a row-wise API op and a list Lambda input
    must be read at the aggregate (plain item) path, not the row-wise path."""
    stock = input_placeholder("Stock")
    aggregate = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        aggregate_table=[
            {"label": "summary", "node": stock.id, "path": "items.output"}
        ],
    )
    aggregate.inputs.append(stock)
    rowwise = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        rowwise_template="Summarize {Prior}.",
        rowwise_columns=[
            {"label": "Prior", "node": aggregate.id, "path": "items.output"}
        ],
    )
    rowwise.inputs.append(aggregate)
    output = as_output("result", rowwise)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (rowwise_row,) = runtime_graph.dsl_to_runtime[rowwise.id]
    node = runtime_graph.nodes[rowwise_row]
    assert node.dependencies == (aggregate.id,)
    assert node.data_spec["columns"][0]["path"] == (
        "items.json.choices[0].message.content"
    )


def test_api_rowwise_fanout_aggregate_refanout() -> None:
    """A rowwise API op fanning out, feeding an aggregate over all its rows,
    feeding a rowwise re-fanout over the aggregate's output must build with one
    task per op, and the re-fanout column references the aggregate node at the
    aggregate (plain item) path."""
    stock = input_placeholder("Stock")
    fanout = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        rowwise_template="Summarize {S}.",
        rowwise_columns=[{"label": "S", "data": {"type": "list", "items": ["a", "b"]}}],
    )
    aggregate = LLMChatOp(
        [OpMessage(role="user", content="Summarize the table.")],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        aggregate_table=[
            {"label": "summary", "node": fanout.id, "path": "items.output"}
        ],
    )
    aggregate.inputs.append(fanout)
    refanout = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        rowwise_template="Summarize {Prior}.",
        rowwise_columns=[
            {"label": "Prior", "node": aggregate.id, "path": "items.output"}
        ],
    )
    refanout.inputs.append(aggregate)
    output = as_output("result", refanout)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (fanout_row,) = runtime_graph.dsl_to_runtime[fanout.id]
    (aggregate_row,) = runtime_graph.dsl_to_runtime[aggregate.id]
    (refanout_row,) = runtime_graph.dsl_to_runtime[refanout.id]
    assert fanout_row == fanout.id
    assert aggregate_row == aggregate.id
    assert refanout_row == refanout.id
    assert runtime_graph.nodes[fanout_row].data_spec["type"] == "dataframe"
    assert runtime_graph.nodes[aggregate_row].data_spec["type"] == "graph_template"
    assert runtime_graph.nodes[refanout_row].data_spec["type"] == "dataframe"
    refanout_node = runtime_graph.nodes[refanout_row]
    assert refanout_node.dependencies == (aggregate_row,)
    assert refanout_node.data_spec["columns"][0]["path"] == (
        "items.json.choices[0].message.content"
    )
    aggregate_node = runtime_graph.nodes[aggregate_row]
    df_col = next(
        c
        for c in aggregate_node.data_spec["template"]["columns"]
        if c.get("label") == "df"
    )
    assert df_col["data"]["columns"][0]["path"] == (
        "items.rows.json.choices[0].message.content"
    )


def test_api_rowwise_data_spec_matches_local_path() -> None:
    """A row-wise op built with and without API config must emit the same
    data_spec and dependencies: the API and local paths differ only in
    task_type, backend and api_spec."""
    stock = input_placeholder("Stock")
    upstream = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
    )

    def build(api: bool) -> tuple[dict, tuple | None]:
        cfg = GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig() if api else None,
        )
        llm = LLMChatOp(
            [OpMessage(role="user", content=stock)],
            config=cfg,
            rowwise_template="Summarize {Prior}.",
            rowwise_columns=[
                {"label": "Prior", "node": upstream.id, "path": "items.output"}
            ],
        )
        llm.inputs.append(upstream)
        output = as_output("result", llm)
        compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
        runtime_graph = RuntimeGraphBuilder().build(compiled)
        (row,) = runtime_graph.dsl_to_runtime[llm.id]
        node = runtime_graph.nodes[row]
        return node.data_spec, node.dependencies

    api_data, api_deps = build(api=True)
    local_data, local_deps = build(api=False)
    assert api_data == local_data
    assert api_deps == local_deps
