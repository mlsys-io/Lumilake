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
    assert api["headers"]["Authorization"] == f"Bearer {_RUNTIME_TOKEN}"
    body = api["json"]
    assert body["model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert body["messages"] == [{"role": "user", "content": "NVDA"}]
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
    """The Authorization header is redacted from graph-level serialize() (the
    stored form) but kept at op level, which graph building reads from."""
    runtime_graph, llm_id = _build_api_graph()
    node = runtime_graph.nodes[llm_id]

    assert node.api_spec["headers"]["Authorization"] == f"Bearer {_RUNTIME_TOKEN}"

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
    assert node.api_spec["headers"]["Authorization"] == "Bearer caller-key"


def test_api_trusted_origin_ignores_caller_credential() -> None:
    """For a trusted origin the server PAT always wins; a caller-supplied
    config.api.authorization is not honored."""
    runtime_graph, llm_id = _build_api_graph(
        api=ApiConfig(authorization="Bearer caller-key")
    )
    node = runtime_graph.nodes[llm_id]
    assert node.api_spec["headers"]["Authorization"] == f"Bearer {_RUNTIME_TOKEN}"


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


def test_api_url_omitted_defaults_to_lumid() -> None:
    runtime_graph, llm_id = _build_api_graph(api=ApiConfig())
    node = runtime_graph.nodes[llm_id]
    assert node.api_spec["url"] == _DEFAULT_API_URL


def test_api_model_omitted_is_rejected() -> None:
    """API mode is a backend switch, not a model source: an empty top-level
    ``model`` must be rejected at config construction, exactly like local
    mode, so the workflow spec reads the same either way."""
    with pytest.raises(ValueError, match="model is required"):
        GenerationConfig(model="", api=ApiConfig())


def test_yaml_api_llm_op_without_config_model_is_rejected() -> None:
    """A YAML LLMChatOp that sets ``config.api`` but omits ``config.model``
    must be rejected at parse time, exactly like local mode: ``config.api`` is
    a backend switch and must not relax the model requirement, so the workflow
    spec reads the same either way."""
    yaml_text = textwrap.dedent(
        """
        name: yaml-api-no-model

        ops:
          - id: "Ask"
            op: LLMChatOp
            messages:
              - role: user
                content: "hello"
            config:
              api: {}

        outputs:
          - name: result
            ref: "Ask"
        """
    )
    with pytest.raises(ValueError, match="requires 'config' with a 'model' field"):
        parse_yaml_payload(yaml_text)


def test_yaml_api_config_string_fails_at_graph_build_not_runtime_build() -> None:
    """A YAML ``config.api: "x"`` (a scalar, not a mapping) must be rejected
    where ``GenerationConfig`` is constructed -- during ``Graph.from_json`` --
    rather than reaching ``RuntimeGraphBuilder`` and crashing there with
    ``AttributeError: 'str' object has no attribute 'url'`` from
    ``_build_api_llm_op``."""
    yaml_text = textwrap.dedent(
        """
        name: yaml-api-bad-type

        ops:
          - id: "Ask"
            op: LLMChatOp
            messages:
              - role: user
                content: "hello"
            config:
              model: "local-model"
              api: "x"

        outputs:
          - name: result
            ref: "Ask"
        """
    )
    specs = parse_yaml_payload(yaml_text)
    spec = specs["yaml-api-bad-type"]

    with pytest.raises(ValueError, match="api must be a mapping or ApiConfig"):
        Graph.from_json(spec["graph"])


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
    assert node.api_spec["json"]["messages"] == [
        {"role": "user", "content": f"${{{retrieval.id}.items.0.table}}"}
    ]
    assert node.dependencies == (retrieval.id,)


def test_api_node_downstream_of_api_node_receives_upstream_placeholder() -> None:
    """An API-mode LLMChatOp consuming another API-mode LLMChatOp's output
    (relayed through a ``FormatOp``) renders a ``${node.text}`` placeholder
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
    assert second_node.api_spec["json"]["messages"] == [
        {"role": "user", "content": f"${{{first_row_id}.text}}"}
    ]
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
    assert prefixed.nodes[prefixed_second].api_spec["json"]["messages"] == [
        {"role": "user", "content": f"${{{prefixed_first}.text}}"}
    ]


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
    messages = prefixed.nodes[second_row_id].api_spec["json"]["messages"]
    assert {"role": "system", "content": sibling_literal} in messages


def test_local_node_downstream_of_api_node_uses_text_path() -> None:
    """A local-backend LLMChatOp consuming an API-backed ancestor's output
    (relayed through a ``FormatOp``) renders a ``text`` column, not
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
    assert any(col.get("path") == "text" for col in upstream_columns)
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
    ``text`` (the ``APIResult`` shape), not ``items.output``."""
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
    assert data_spec["path"] == "text"


def test_image_generation_op_downstream_of_api_node_uses_text_path() -> None:
    """An ImageGenerationOp consuming an API-backed ancestor's output must
    resolve ``text`` (the ``APIResult`` shape) instead of ``items.output``."""
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
    assert data_spec["path"] == "text"


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

    messages = runtime_graph.nodes[llm.id].api_spec["json"]["messages"]
    assert messages == [{"role": "user", "content": "HELLO, NVDA!"}]


def test_api_lambda_op_message_input_renders_literal_with_lambda() -> None:
    """A LambdaOp whose function is a real ``lambda`` must render in API mode
    too: the server evaluates the callable directly, so a lambda behaves
    identically to a named def."""
    stock = input_placeholder("Stock")
    greeting = FormatOp("Hello, {name}!", name=stock)
    shout = LambdaOp(
        [greeting],
        fn=lambda inputs: (
            inputs[0].upper() if isinstance(inputs[0], str) else inputs[0][0].content
        ),
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

    messages = runtime_graph.nodes[llm.id].api_spec["json"]["messages"]
    assert messages == [{"role": "user", "content": "HELLO, NVDA!"}]


def test_api_lambda_over_runtime_output_fails_closed() -> None:
    """A Lambda message transform over a runtime output must fail closed in API
    mode with a self-explaining error: the API request body cannot carry a
    graph_template function step, so the transform cannot be evaluated at
    dispatch time. This pins the rejection so it cannot silently become a
    wrong-answer path."""
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

    with pytest.raises(ValueError, match="Lambda message transform over a runtime"):
        RuntimeGraphBuilder().build(compiled)


def test_api_rowwise_template_fans_out_row_aligned_nodes() -> None:
    """An API-backed LLMChatOp with ``rowwise_template``/``rowwise_columns``/
    ``system_messages`` must mirror the local rowwise contract: the template is
    formatted per row and the op fans out into one API task per row."""
    stock = input_placeholder("Stock")
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
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

    row_ids = runtime_graph.dsl_to_runtime[llm.id]
    assert row_ids == [llm.id, f"{llm.id}__row1"]
    assert runtime_graph.nodes[row_ids[0]].api_spec["json"]["messages"] == [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Summarize NVDA."},
    ]
    assert runtime_graph.nodes[row_ids[1]].api_spec["json"]["messages"] == [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Summarize AAPL."},
    ]


def test_api_rowwise_timeout_sec_reaches_emitted_spec() -> None:
    """A rowwise API op with ``timeout_sec`` must emit it on every fanned-out
    row."""
    stock = input_placeholder("Stock")
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(timeout_sec=300.0),
        ),
        rowwise_template="Summarize {Stock}.",
        rowwise_columns=[
            {"label": "Stock", "data": {"type": "list", "items": ["NVDA", "AAPL"]}}
        ],
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA", "AAPL"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    row_ids = runtime_graph.dsl_to_runtime[llm.id]
    assert row_ids == [llm.id, f"{llm.id}__row1"]
    assert runtime_graph.nodes[row_ids[0]].api_spec["timeout_sec"] == 300.0
    assert runtime_graph.nodes[row_ids[1]].api_spec["timeout_sec"] == 300.0


def test_api_rowwise_model_override_reaches_emitted_spec() -> None:
    """A rowwise API op must resolve ``config.api.model`` into every fanned-out
    row's request body, not fall back to top-level ``config.model``."""
    stock = input_placeholder("Stock")
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(model="gpt-4o"),
        ),
        rowwise_template="Summarize {Stock}.",
        rowwise_columns=[
            {"label": "Stock", "data": {"type": "list", "items": ["NVDA", "AAPL"]}}
        ],
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA", "AAPL"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    row_ids = runtime_graph.dsl_to_runtime[llm.id]
    assert row_ids == [llm.id, f"{llm.id}__row1"]
    assert runtime_graph.nodes[row_ids[0]].api_spec["json"]["model"] == "gpt-4o"
    assert runtime_graph.nodes[row_ids[1]].api_spec["json"]["model"] == "gpt-4o"
    assert runtime_graph.nodes[row_ids[0]].model == "gpt-4o"


def test_api_aggregate_table_renders_df_column() -> None:
    """An API-backed LLMChatOp with ``aggregate_table`` must mirror the local
    aggregate contract: the base template columns are merged with a ``df``
    dataframe column built from ``aggregate_table``."""
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
        aggregate_table=[{"label": "summary", "node": upstream.id, "path": "text"}],
    )
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
        {"label": "summary", "node": upstream.id, "path": "text"}
    ]


def test_api_aggregate_timeout_sec_reaches_emitted_spec() -> None:
    """An aggregate API op with ``timeout_sec`` must emit it."""
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
            api=ApiConfig(timeout_sec=300.0),
        ),
        aggregate_table=[{"label": "summary", "node": upstream.id, "path": "text"}],
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (llm_row_id,) = runtime_graph.dsl_to_runtime[llm.id]
    node = runtime_graph.nodes[llm_row_id]
    assert node.api_spec["timeout_sec"] == 300.0


def test_api_aggregate_df_prompt_renders_into_request_body() -> None:
    """An aggregate API op whose message is a ``FormatOp`` template containing
    ``{df}`` must render the dataframe into the request body, mirroring the
    local aggregate contract; dropping the format steps leaves the message
    referencing an unresolved step label."""
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
        aggregate_table=[{"label": "summary", "node": upstream.id, "path": "text"}],
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (llm_row_id,) = runtime_graph.dsl_to_runtime[llm.id]
    node = runtime_graph.nodes[llm_row_id]
    messages = node.api_spec["json"]["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "Summarize the table:" in messages[0]["content"]


def test_api_aggregate_model_override_reaches_emitted_spec() -> None:
    """An aggregate API op must resolve ``config.api.model`` into its request
    body, not fall back to top-level ``config.model``."""
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
            api=ApiConfig(model="gpt-4o"),
        ),
        aggregate_table=[{"label": "summary", "node": upstream.id, "path": "text"}],
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (llm_row_id,) = runtime_graph.dsl_to_runtime[llm.id]
    node = runtime_graph.nodes[llm_row_id]
    assert node.api_spec["json"]["model"] == "gpt-4o"
    assert node.model == "gpt-4o"


def test_api_generic_op_emits_condition() -> None:
    """A plain (non-rowwise, non-aggregate) API LLMChatOp must emit ``condition``
    on its runtime node, matching the local builder."""
    stock = input_placeholder("Stock")
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
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


def test_api_rowwise_op_emits_condition() -> None:
    """A rowwise API LLMChatOp must emit ``condition`` on every fanned-out
    row, matching the local rowwise builder."""
    stock = input_placeholder("Stock")
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        rowwise_template="Summarize {Stock}.",
        rowwise_columns=[
            {"label": "Stock", "data": {"type": "list", "items": ["NVDA", "AAPL"]}}
        ],
        condition={"node": "gate", "expr": "gate == 'on'"},
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA", "AAPL"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    for row_id in runtime_graph.dsl_to_runtime[llm.id]:
        assert runtime_graph.nodes[row_id].condition == {
            "node": "gate",
            "expr": "gate == 'on'",
        }


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
        aggregate_table=[{"label": "summary", "node": upstream.id, "path": "text"}],
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
    inlined as a literal column and the assistant output resolves ``text``."""
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
    assert output_cols and output_cols[0]["path"] == "text"


def test_api_ancestor_with_return_history_feeds_api_downstream() -> None:
    """The same history parity must hold for an API->API edge: the prior prompt
    inlines as a literal and the assistant output renders ``${node.text}``."""
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
    messages = runtime_graph.nodes[downstream_row_id].api_spec["json"]["messages"]
    assert messages == [
        {"role": "user", "content": "NVDA"},
        {"role": "assistant", "content": f"${{{api_row_id}.text}}"},
    ]


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


def test_api_scheme_less_url_raises_clean_error() -> None:
    with pytest.raises(ValueError, match="no host"):
        _build_api_graph(api=ApiConfig(url="lum.id/llm/v1/chat/completions"))


def test_api_explicit_default_port_is_trusted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMILAKE_API_TRUSTED_ORIGINS", "https://lum.id")
    runtime_graph, llm_id = _build_api_graph(
        api=ApiConfig(url="https://lum.id:443/v1/chat/completions")
    )
    node = runtime_graph.nodes[llm_id]
    assert node.api_spec["headers"]["Authorization"] == f"Bearer {_RUNTIME_TOKEN}"


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
        default_node.api_spec["headers"]["Authorization"] == f"Bearer {_RUNTIME_TOKEN}"
    )

    vendor_graph, vendor_llm_id = _build_api_graph(
        api=ApiConfig(url="https://vendor.example.com/v1/chat/completions")
    )
    vendor_node = vendor_graph.nodes[vendor_llm_id]
    assert (
        vendor_node.api_spec["headers"]["Authorization"] == f"Bearer {_RUNTIME_TOKEN}"
    )


def test_api_multi_row_input_fans_out_row_aligned_nodes() -> None:
    """A literal message column with N rows must fan out into N nodes, one
    per row, in row order."""
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

    row1_id = f"{llm.id}__row1"
    assert runtime_graph.dsl_to_runtime[llm.id] == [llm.id, row1_id]
    assert set(runtime_graph.nodes) == {llm.id, row1_id}
    assert runtime_graph.nodes[llm.id].api_spec["json"]["messages"] == [
        {"role": "user", "content": "NVDA"}
    ]
    assert runtime_graph.nodes[row1_id].api_spec["json"]["messages"] == [
        {"role": "user", "content": "AAPL"}
    ]
    assert runtime_graph.output_node_map[llm.id] == "result"
    assert runtime_graph.output_node_map[row1_id] == "result"


def test_api_node_consuming_fanned_api_upstream_builds() -> None:
    """An API-mode LLMChatOp consuming a row-fanned API upstream must build:
    the structural guard permits a fanned upstream only when both the upstream
    and the consumer are API mode, because API message columns can carry the
    per-row alignment. The consumer fans out into one node per upstream row."""
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

    draft_row1 = f"{draft.id}__row1"
    polish_row1 = f"{polish.id}__row1"
    assert runtime_graph.dsl_to_runtime[draft.id] == [draft.id, draft_row1]
    assert runtime_graph.dsl_to_runtime[polish.id] == [polish.id, polish_row1]


def test_structural_consumer_of_unfanned_api_upstream_builds() -> None:
    """An API-mode consumer of an unfanned API upstream must build: the
    structural fanned-consumer guard must not over-reject a single-row
    upstream."""
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
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (draft_row,) = runtime_graph.dsl_to_runtime[draft.id]
    (polish_row,) = runtime_graph.dsl_to_runtime[polish.id]
    assert runtime_graph.nodes[polish_row].api_spec["json"]["messages"] == [
        {"role": "user", "content": f"${{{draft_row}.text}}"}
    ]


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


def test_condition_matching_cardinality_builds_and_remaps() -> None:
    """A condition where the consumer and source fan out to the same number of
    rows must build and remap each consumer row to the matching source row: the
    condition cardinality guard must not over-reject a matching pair."""
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
        rowwise_template="Summarize {Stock}.",
        rowwise_columns=[
            {
                "label": "Stock",
                "data": {"type": "list", "items": ["x", "y"]},
            }
        ],
        condition={"node": gate.id, "expr": "gate == on"},
    )
    compiled = Graph.from_ops(
        [as_output("gate_out", gate), as_output("result", consumer)]
    ).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    gate_rows = runtime_graph.dsl_to_runtime[gate.id]
    consumer_rows = runtime_graph.dsl_to_runtime[consumer.id]
    assert len(gate_rows) == 2
    assert len(consumer_rows) == 2
    for row_index, consumer_row in enumerate(consumer_rows):
        assert runtime_graph.nodes[consumer_row].condition == {
            "node": gate_rows[row_index],
            "expr": "gate == on",
        }


def test_api_node_consuming_fanned_api_upstream_emits_per_row_placeholders() -> None:
    """An API-mode LLMChatOp consuming a row-fanned API upstream must emit one
    ``${<row>.text}`` placeholder per upstream row node and declare every
    upstream row node as a dependency. Row i's downstream node must consume
    upstream row i."""
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

    draft_row1 = f"{draft.id}__row1"
    polish_row1 = f"{polish.id}__row1"

    assert runtime_graph.nodes[polish.id].api_spec["json"]["messages"] == [
        {"role": "user", "content": f"${{{draft.id}.text}}"}
    ]
    assert runtime_graph.nodes[polish_row1].api_spec["json"]["messages"] == [
        {"role": "user", "content": f"${{{draft_row1}.text}}"}
    ]

    assert set(runtime_graph.nodes[polish.id].dependencies) == {
        draft.id,
        draft_row1,
    }
    assert set(runtime_graph.nodes[polish_row1].dependencies) == {
        draft.id,
        draft_row1,
    }


def test_api_node_consuming_fanned_return_history_upstream_fails_closed() -> None:
    """An API-mode LLMChatOp with ``return_history`` cannot consume a row-fanned
    API upstream: API mode cannot reconstruct per-row history for a fanned
    upstream, so the graph build fails closed."""
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

    with pytest.raises(ValueError, match="cannot reconstruct per-row history"):
        RuntimeGraphBuilder().build(compiled)


def test_node_prefix_preserves_api_spec_on_row_fanned_nodes() -> None:
    """``with_node_prefix`` must carry a row-fanned API node's api_spec through
    unchanged, or the prefixed node dispatches with no URL/headers/body."""
    stock = input_placeholder("Stock")
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct", api=ApiConfig()
        ),
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA", "AAPL"])

    unprefixed = RuntimeGraphBuilder().build(compiled)
    prefix = make_node_prefix("job1")
    prefixed = RuntimeGraphBuilder().build(compiled, node_prefix="job1")

    row_ids = unprefixed.dsl_to_runtime[llm.id]
    assert len(row_ids) == 2
    for row_id in row_ids:
        prefixed_id = f"{prefix}__{row_id}"
        assert prefixed_id in prefixed.nodes
        assert prefixed.nodes[prefixed_id].api_spec == unprefixed.nodes[row_id].api_spec
        assert prefixed.nodes[prefixed_id].api_spec != {}


def test_api_fanout_row_order_matches_input_across_two_nodes() -> None:
    """Two independent API nodes fed by the same multi-row input must both
    preserve row order identically."""
    stock = input_placeholder("Stock")
    first = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(model="model-a", api=ApiConfig()),
    )
    second = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(model="model-b", api=ApiConfig()),
    )
    compiled = Graph.from_ops(
        [as_output("first_result", first), as_output("second_result", second)]
    ).compile(Stock=["NVDA", "AAPL", "MSFT"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)

    for llm in (first, second):
        row_ids = runtime_graph.dsl_to_runtime[llm.id]
        assert row_ids == [llm.id, f"{llm.id}__row1", f"{llm.id}__row2"]
        contents = [
            runtime_graph.nodes[row_id].api_spec["json"]["messages"][0]["content"]
            for row_id in row_ids
        ]
        assert contents == ["NVDA", "AAPL", "MSFT"]


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


def test_yaml_wrapped_bare_reference_fans_out_row_aligned_nodes() -> None:
    """A bare ``content: "Topic"`` reference in YAML is implicitly wrapped into
    a FormatOp step by the parser; two input rows must still produce two
    row-aligned API nodes through that step, not collapse to one."""
    runtime_graph, llm_id = _build_yaml_two_row_graph()

    row_ids = runtime_graph.dsl_to_runtime[llm_id]
    assert row_ids == [llm_id, f"{llm_id}__row1"]

    contents = [
        runtime_graph.nodes[row_id].api_spec["json"]["messages"][-1]["content"]
        for row_id in row_ids
    ]
    assert contents == ["a database index", "a message queue"]

    assert runtime_graph.output_node_map[row_ids[0]] == "result"
    assert runtime_graph.output_node_map[row_ids[1]] == "result"


def test_merged_workflow_result_stays_row_aligned_after_optimize() -> None:
    """The merged/optimized graph must keep both per-row output nodes distinct
    and row-ordered; two input rows must resolve to two output entries."""
    runtime_graph, llm_id = _build_yaml_two_row_graph()

    optimized_graph, output_mapping = HaloOptimizer().optimize_graphs(
        {"yaml-api-two-rows": runtime_graph}
    )

    row_ids = runtime_graph.dsl_to_runtime[llm_id]
    assert len(row_ids) == 2

    result_nodes = [
        node_id
        for node_id, name in optimized_graph.output_node_map.items()
        if name == "result"
    ]
    assert sorted(result_nodes) == sorted(row_ids)
    for node_id in result_nodes:
        assert output_mapping[node_id] == ("yaml-api-two-rows", "result")

    contents = [
        optimized_graph.nodes[row_id].api_spec["json"]["messages"][-1]["content"]
        for row_id in row_ids
    ]
    assert contents == ["a database index", "a message queue"]


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


def test_api_row_fanned_node_feeding_local_node_fails_closed() -> None:
    """A local (non-API) LLM node cannot consume a row-fanned API node's
    output: the structural-message wiring only knows the unsuffixed node id,
    so a downstream local node would silently see only the first row. The
    graph build fails closed instead."""
    specs = parse_yaml_payload(_YAML_API_NODE_FEEDS_LOCAL_NODE)
    spec = specs["api-node-feeds-local-node"]
    graph = Graph.from_json(spec["graph"])
    compiled = graph.compile(**spec["inputs"])

    with pytest.raises(ValueError, match="fanned out into multiple row-aligned"):
        RuntimeGraphBuilder().build(compiled)


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
    rows: the structural-message wiring rewrites the reference to the single
    unsuffixed output (``items.0.output``), which would silently drop every
    row but the first. The graph build fails closed instead."""
    specs = parse_yaml_payload(_YAML_LOCAL_NODE_FEEDS_API_NODE)
    spec = specs["local-node-feeds-api-node"]
    graph = Graph.from_json(spec["graph"])
    compiled = graph.compile(**spec["inputs"])

    with pytest.raises(ValueError, match="produces multiple rows"):
        RuntimeGraphBuilder().build(compiled)


def test_api_node_feeding_single_row_local_upstream_builds() -> None:
    """An API node consuming a local upstream that produces a single row must
    still build: the guard fires only for multi-row upstreams, never for a
    single-row one."""
    specs = parse_yaml_payload(_YAML_LOCAL_NODE_FEEDS_API_NODE)
    spec = specs["local-node-feeds-api-node"]
    graph = Graph.from_json(spec["graph"])
    compiled = graph.compile(Topic=["a database index"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)
    api_node = next(
        node for node in runtime_graph.nodes.values() if node.task_type == "api"
    )
    assert api_node.api_spec["json"]["messages"] == [
        {"role": "user", "content": f"${{{api_node.dependencies[0]}.items.0.output}}"}
    ]


def test_api_rowwise_node_feeding_local_multi_row_upstream_fails_closed() -> None:
    """A rowwise API op whose node-ref column references a multi-row local
    upstream must fail closed (the rowwise guard, not the structural one)."""
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
        rowwise_template="Summarize {Prior}.",
        rowwise_columns=[{"label": "Prior", "node": local.id, "path": "items.output"}],
    )
    llm.inputs.append(local)
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA", "AAPL"])

    with pytest.raises(ValueError, match="rowwise column 'Prior' references"):
        RuntimeGraphBuilder().build(compiled)


def test_api_rowwise_node_feeding_multi_row_api_upstream_fails_closed() -> None:
    """A rowwise API op whose node-ref column references a fanned-out API
    upstream (multiple rows) must fail closed too: the rowwise branch binds
    only the unsuffixed row-0 node, silently dropping every row but the first.
    The guard must cover API upstreams, not just local ones."""
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
        rowwise_columns=[{"label": "Prior", "node": api_up.id, "path": "text"}],
    )
    llm.inputs.append(api_up)
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA", "AAPL"])

    with pytest.raises(ValueError, match="rowwise column 'Prior' references"):
        RuntimeGraphBuilder().build(compiled)


def test_api_aggregate_node_feeding_local_multi_row_upstream_fails_closed() -> None:
    """An aggregate API op whose ``aggregate_table`` references a multi-row
    local upstream must fail closed (the aggregate guard, not the structural
    one)."""
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

    with pytest.raises(ValueError, match="aggregate column 'summary' references"):
        RuntimeGraphBuilder().build(compiled)


def test_api_aggregate_node_feeding_multi_row_api_upstream_fails_closed() -> None:
    """An aggregate API op whose ``aggregate_table`` references a fanned-out API
    upstream (multiple rows) must fail closed too: the aggregate branch binds
    only the unsuffixed row-0 node, silently dropping every row but the first.
    The guard must cover API upstreams, not just local ones."""
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
        aggregate_table=[{"label": "summary", "node": api_up.id, "path": "text"}],
    )
    llm.inputs.append(api_up)
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA", "AAPL"])

    with pytest.raises(ValueError, match="aggregate column 'summary' references"):
        RuntimeGraphBuilder().build(compiled)


def test_api_aggregate_node_feeding_fanned_api_upstream_fails_closed() -> None:
    """An aggregate API op whose ``aggregate_table`` references a fanned API
    upstream must fail closed even with a single-row input."""
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
        aggregate_table=[{"label": "summary", "node": api_up.id, "path": "text"}],
    )
    llm.inputs.append(api_up)
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    with pytest.raises(ValueError, match="aggregate column 'summary' references"):
        RuntimeGraphBuilder().build(compiled)


def test_api_rowwise_node_feeding_fanned_api_upstream_fails_closed() -> None:
    """A rowwise API op whose node-ref column references a fanned API upstream
    must fail closed even with a single-row input."""
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
        rowwise_columns=[{"label": "Prior", "node": api_up.id, "path": "text"}],
    )
    llm.inputs.append(api_up)
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    with pytest.raises(ValueError, match="rowwise column 'Prior' references"):
        RuntimeGraphBuilder().build(compiled)


def test_api_node_consuming_fanned_upstream_feeding_local_vlm_fails_closed() -> None:
    """A local VLM consumer cannot consume a row-fanned API upstream: a VLM is
    a local embedding + inference pair even with ``config.api`` set."""
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

    with pytest.raises(ValueError, match="fanned out into multiple row-aligned"):
        RuntimeGraphBuilder().build(compiled)


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


def test_api_aggregate_node_feeding_local_rowwise_literal_upstream_fails_closed() -> (
    None
):
    """An aggregate API op whose ``aggregate_table`` references a local rowwise
    LLMChatOp fanned by literal ``rowwise_columns`` must fail closed: the
    upstream emits multiple rows from one runtime node, so the static row-count
    estimate must see the rowwise columns."""
    stock = input_placeholder("Stock")
    local_rw = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
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
            {"label": "summary", "node": local_rw.id, "path": "items.output"}
        ],
    )
    llm.inputs.append(local_rw)
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    with pytest.raises(ValueError, match="aggregate column 'summary' references"):
        RuntimeGraphBuilder().build(compiled)


def test_api_rowwise_node_feeding_local_rowwise_literal_upstream_fails_closed() -> None:
    """A rowwise API op whose node-ref column references a local rowwise
    LLMChatOp fanned by literal ``rowwise_columns`` must fail closed: the
    upstream emits multiple rows from one runtime node, so the static row-count
    estimate must see the rowwise columns."""
    stock = input_placeholder("Stock")
    local_rw = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
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
        rowwise_columns=[
            {"label": "Prior", "node": local_rw.id, "path": "items.output"}
        ],
    )
    llm.inputs.append(local_rw)
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    with pytest.raises(ValueError, match="rowwise column 'Prior' references"):
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


def test_fanned_consumer_condition_remaps_to_matching_source_row() -> None:
    """A condition on a fanned consumer whose ``node`` is a fanned source must
    gate each consumer row against the matching source row, not against row 0 of
    the source."""
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
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    gate_rows = runtime_graph.dsl_to_runtime[gate.id]
    consumer_rows = runtime_graph.dsl_to_runtime[consumer.id]
    assert len(gate_rows) == 2
    assert len(consumer_rows) == 2
    for row_index, consumer_row in enumerate(consumer_rows):
        assert runtime_graph.nodes[consumer_row].condition == {
            "node": gate_rows[row_index],
            "expr": "gate == on",
        }


def test_local_rowwise_node_feeding_fanned_api_upstream_fails_closed() -> None:
    """A local rowwise LLMChatOp whose node-ref column references a fanned API
    producer must fail closed: the local rowwise branch binds a single
    ``node:`` reference to the unsuffixed row-0 node, silently dropping every
    row but the first."""
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
        rowwise_columns=[{"label": "Prior", "node": api_up.id, "path": "text"}],
    )
    local_rw.inputs.append(api_up)
    output = as_output("result", local_rw)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    with pytest.raises(ValueError, match="rowwise column references"):
        RuntimeGraphBuilder().build(compiled)


def test_local_aggregate_node_feeding_fanned_api_upstream_fails_closed() -> None:
    """A local aggregate LLMChatOp whose ``aggregate_table`` references a fanned
    API producer must fail closed: the local aggregate branch binds a single
    ``node:`` reference to the unsuffixed row-0 node, silently dropping every
    row but the first."""
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
        aggregate_table=[{"label": "summary", "node": api_up.id, "path": "text"}],
    )
    local_agg.inputs.append(api_up)
    output = as_output("result", local_agg)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    with pytest.raises(ValueError, match="aggregate column references"):
        RuntimeGraphBuilder().build(compiled)


def test_vlm_rowwise_column_feeding_fanned_api_upstream_fails_closed() -> None:
    """A VLM rowwise column referencing a fanned API producer must fail closed:
    the VLM rowwise branch binds a single ``node:`` reference to the unsuffixed
    row-0 node, silently dropping every row but the first."""
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
        rowwise_columns=[{"label": "Prior", "node": api_up.id, "path": "text"}],
    )
    vision.inputs.append(api_up)
    output = as_output("result", vision)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    with pytest.raises(ValueError, match="rowwise column references"):
        RuntimeGraphBuilder().build(compiled)


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
    """A condition on a fanned consumer whose ``node`` is a fanned source must
    fail closed when the consumer and source fan out to different row counts —
    in either direction — rather than silently gating on the wrong source row
    or escaping as an IndexError."""
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

    with pytest.raises(ValueError, match="fanned out into"):
        build(source_rows=3, consumer_rows=2)
    with pytest.raises(ValueError, match="fanned out into"):
        build(source_rows=2, consumer_rows=3)


def test_one_row_consumer_of_fanned_source_fails_closed() -> None:
    """A one-row API consumer conditioned on a two-row API source must fail
    closed: the cardinality check must run for row 0 too, not only for rows
    after the first. A one-row consumer is entirely row 0, so skipping the
    check there would silently gate it on source row 0 only."""
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

    with pytest.raises(ValueError, match="fanned out into"):
        RuntimeGraphBuilder().build(compiled)


def test_condition_source_max_cardinality_across_messages() -> None:
    """A condition source must fan out over the max cardinality across all
    messages; the consumer is built first to force the static fallback."""
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
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    gate_rows = runtime_graph.dsl_to_runtime[gate.id]
    consumer_rows = runtime_graph.dsl_to_runtime[consumer.id]
    assert len(gate_rows) == 2
    assert len(consumer_rows) == 2
    for row_index, consumer_row in enumerate(consumer_rows):
        assert runtime_graph.nodes[consumer_row].condition == {
            "node": gate_rows[row_index],
            "expr": "gate == on",
        }


def test_condition_on_local_rowwise_source_broadcasts_single_node() -> None:
    """A condition on a local rowwise producer must reference the producer's
    single runtime node for every consumer row, not a synthesized
    ``__rowN`` node: a local rowwise op emits multiple items from one node, so
    it has no per-row nodes to gate on."""
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
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (local_row,) = runtime_graph.dsl_to_runtime[local_rw.id]
    for consumer_row in runtime_graph.dsl_to_runtime[consumer.id]:
        assert runtime_graph.nodes[consumer_row].condition == {
            "node": local_row,
            "expr": "gate == on",
        }


def test_condition_source_fans_over_non_user_role_message() -> None:
    """A condition source whose fanout comes from a non-user-role message (a
    system message with a multi-row InputOp) must fan out over that message."""
    sys_in = input_placeholder("Sys")
    src = LLMChatOp(
        [
            OpMessage(role="system", content=sys_in),
            OpMessage(role="user", content="hi"),
        ],
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
    ).compile(Sys=["a", "b"])

    with pytest.raises(ValueError, match="fanned out into"):
        RuntimeGraphBuilder().build(compiled)


def test_local_consumer_of_fanned_source_fails_closed() -> None:
    """A local (non-API) consumer conditioned on a fanned API source must fail
    closed: the fail-closed condition invariant applies to local consumers too,
    not just API consumers. A single-node local consumer gated on a two-row
    source would otherwise gate only on source row 0."""
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

    with pytest.raises(ValueError, match="fanned out into"):
        RuntimeGraphBuilder().build(compiled)


def test_local_rowwise_consumer_of_fanned_source_fails_closed() -> None:
    """A local rowwise consumer conditioned on a fanned API source must fail
    closed, matching the local generic consumer."""
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
    local_rw = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
        rowwise_template="Summarize {S}.",
        rowwise_columns=[{"label": "S", "data": {"type": "list", "items": ["x"]}}],
        condition={"node": gate.id, "expr": "gate == on"},
    )
    compiled = Graph.from_ops(
        [as_output("gate_out", gate), as_output("result", local_rw)]
    ).compile(Stock=["NVDA"])

    with pytest.raises(ValueError, match="fanned out into"):
        RuntimeGraphBuilder().build(compiled)


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
    local ``items.0.output`` path (the LLMVisionOp exclusion must hold)."""
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
    assert runtime_graph.nodes[api.id].api_spec["json"]["messages"] == [
        {"role": "user", "content": f"${{{vlm.id}.items.0.output}}"}
    ]


def test_condition_source_fanned_through_lambda_fails_closed() -> None:
    """A one-row consumer conditioned on a source whose API fanout comes through
    a LambdaOp over a multi-row input must fail closed: the static row-count
    fallback must count a LambdaOp's inputs, so the consumer is not left gated
    on source row 0 while the source subsequently builds multiple rows."""
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

    with pytest.raises(ValueError, match="fanned out into"):
        RuntimeGraphBuilder().build(compiled)


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


def test_api_nodes_distinguished_by_timeout_sec_in_dedupe() -> None:
    """Two API nodes differing only in ``timeout_sec`` must not be merged by
    the optimizer's dedupe pass."""
    shared = {
        "method": "POST",
        "url": "https://lum.id/llm/v1/chat/completions",
        "json": {"messages": [{"role": "user", "content": "a database index"}]},
    }
    row0 = RuntimeOp(
        node_id="Summarise",
        task_type="api",
        backend="api",
        model="model-a",
        data_spec={"type": "graph_template", "template": {"columns": []}},
        model_spec={},
        inference_spec={},
        api_spec={**shared, "timeout_sec": 60.0},
    )
    row1 = RuntimeOp(
        node_id="Summarise__row1",
        task_type="api",
        backend="api",
        model="model-a",
        data_spec={"type": "graph_template", "template": {"columns": []}},
        model_spec={},
        inference_spec={},
        api_spec={**shared, "timeout_sec": 300.0},
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
    timeouts = sorted(
        optimized_graph.nodes[node_id].api_spec["timeout_sec"]
        for node_id in (row0.node_id, row1.node_id)
    )
    assert timeouts == [60.0, 300.0]
