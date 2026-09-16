import textwrap

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
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.optimizer.halo import HaloOptimizer
from lumilake_server.runtime.runtime_graph import RuntimeGraph, RuntimeGraphBuilder

_LUMID_URL = "http://lumid-data"
_LUMID_TOKEN = "test-token"
_RUNTIME_TOKEN = "test" + "-pat"
_DEFAULT_API_URL = "https://lum.id/llm/v1/chat/completions"
_DEFAULT_API_MODEL = "deepseek-v4-flash"


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", _LUMID_URL)
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", _LUMID_TOKEN)
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", _RUNTIME_TOKEN)


def _build_api_graph(**config_kwargs) -> tuple[RuntimeGraph, str]:
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


def test_api_samplers_flow_into_body() -> None:
    runtime_graph, llm_id = _build_api_graph(max_tokens=64, temperature=0.5)
    node = runtime_graph.nodes[llm_id]
    body = node.api_spec["json"]
    assert body["max_tokens"] == 64
    assert body["temperature"] == 0.5


def test_api_key_redacted_on_serialization() -> None:
    """The Authorization header carries the PAT in the emitted spec, but must
    be redacted whenever the spec is serialized for storage or logging."""
    runtime_graph, llm_id = _build_api_graph()
    node = runtime_graph.nodes[llm_id]

    # The live spec must carry the real credential - it is what reaches FlowMesh.
    assert node.api_spec["headers"]["Authorization"] == f"Bearer {_RUNTIME_TOKEN}"

    # Graph-level serialization is the form that gets stored, and it must not.
    # Op-level serialize() is an internal step that graph-level builds from, so
    # it deliberately still carries the token.
    assert _RUNTIME_TOKEN not in str(runtime_graph.serialize())
    assert "Bearer" not in str(runtime_graph.serialize())


def test_api_missing_pat_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """API mode against a trusted endpoint without a PAT fails at build time
    with a clear error rather than emitting a spec that fails later."""
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


def test_api_model_omitted_defaults_to_deepseek() -> None:
    runtime_graph, llm_id = _build_api_graph(api=ApiConfig(), model="")
    node = runtime_graph.nodes[llm_id]
    assert node.api_spec["json"]["model"] == _DEFAULT_API_MODEL


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
            api=ApiConfig(),
        ),
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    with pytest.raises(ValueError, match="cannot render message"):
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


def test_api_multi_row_input_fans_out_row_aligned_nodes() -> None:
    """A literal message column with N rows must fan out into N nodes, one
    per row, in row order - not reject the graph as it did previously."""
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


def test_api_fanout_row_order_matches_input_across_two_nodes() -> None:
    """Row alignment must be a per-node property of the fan-out itself, not
    an accident of a single node under test: two independent API nodes fed
    by the same multi-row input must both preserve row order identically."""
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
    """A bare ``content: "Topic"`` reference in YAML is implicitly wrapped
    into a FormatOp step by the parser (mirroring n8n's prompt wrapping), so
    it never appears as a literal column to the runtime graph builder - only
    as a format step. Two input rows must still produce two row-aligned API
    nodes through that step, not collapse to one."""
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
    """The merged/optimized graph that scheduling actually runs against must
    keep both per-row output nodes distinct and row-ordered; a job with two
    input rows must resolve to two output entries, not collapse to one."""
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


def test_api_row_fanned_non_terminal_node_survives_dedupe() -> None:
    """A row-fanned API node that is not itself output-mapped (it feeds a
    downstream local op instead) reuses the same unresolved data_spec for
    every row - only api_spec differs per row. The optimizer's dedupe
    signature must include api_spec, or it will incorrectly collapse
    distinct rows of a non-terminal API node into one."""
    specs = parse_yaml_payload(_YAML_API_NODE_FEEDS_LOCAL_NODE)
    spec = specs["api-node-feeds-local-node"]
    graph = Graph.from_json(spec["graph"])
    compiled = graph.compile(**spec["inputs"])
    summarise_id = next(
        op_id
        for op_id, op_dict in spec["graph"].items()
        if op_dict.get("_op") == "LLMChatOp" and "api" in op_dict.get("config", {})
    )

    runtime_graph = RuntimeGraphBuilder().build(compiled)
    summarise_row_ids = runtime_graph.dsl_to_runtime[summarise_id]
    assert len(summarise_row_ids) == 2

    optimized_graph, _ = HaloOptimizer().optimize_graphs({"wf": runtime_graph})

    optimized_summarise_ids = {
        node_id for node_id in summarise_row_ids if node_id in optimized_graph.nodes
    }
    assert len(optimized_summarise_ids) == 2

    contents = [
        optimized_graph.nodes[row_id].api_spec["json"]["messages"][-1]["content"]
        for row_id in summarise_row_ids
    ]
    assert contents == ["a database index", "a message queue"]
