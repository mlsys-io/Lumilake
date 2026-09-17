import textwrap

import pytest
from lumilake import envs

from lumilake_server.common import ApiConfig, GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
    DataRetrievalOp,
    FormatOp,
    LLMChatOp,
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


def test_api_samplers_flow_into_body() -> None:
    runtime_graph, llm_id = _build_api_graph(max_tokens=64, temperature=0.5)
    node = runtime_graph.nodes[llm_id]
    body = node.api_spec["json"]
    assert body["max_tokens"] == 64
    assert body["temperature"] == 0.5


def test_api_key_redacted_on_serialization() -> None:
    """The Authorization header carries the PAT in the runtime spec (what
    reaches FlowMesh) but must be redacted from graph-level serialize(), the
    form that gets stored; op-level serialize() deliberately still carries
    it since graph-level building reads from it."""
    runtime_graph, llm_id = _build_api_graph()
    node = runtime_graph.nodes[llm_id]

    assert node.api_spec["headers"]["Authorization"] == f"Bearer {_RUNTIME_TOKEN}"

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


def test_api_trusted_origin_ignores_caller_credential() -> None:
    """For a trusted origin the server PAT always wins; a caller-supplied
    config.api.authorization is not honored (common.py's ApiConfig
    docstring and OPS.md both say the credential is resolved server-side
    for a trusted origin, not supplied by the caller)."""
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


def test_api_model_omitted_defaults_to_deepseek() -> None:
    runtime_graph, llm_id = _build_api_graph(api=ApiConfig(), model="")
    node = runtime_graph.nodes[llm_id]
    assert node.api_spec["json"]["model"] == _DEFAULT_API_MODEL


def test_yaml_api_llm_op_without_config_model_reaches_the_default() -> None:
    """A YAML LLMChatOp that sets ``config.api`` and omits ``config.model``
    entirely (not merely an empty string) must parse and resolve to the
    typed API default end-to-end. Pins yaml_parser.py's
    ``_emit_llm_like_op`` relaxation (the ``is_api_chat_op`` check) that
    lets ``config`` omit the ``model`` key when ``config.api`` is set;
    without it, ``parse_yaml_payload`` raises before this op ever reaches
    the runtime graph."""
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
    specs = parse_yaml_payload(yaml_text)
    spec = specs["yaml-api-no-model"]
    graph = Graph.from_json(spec["graph"])
    compiled = graph.compile()
    llm_id = next(
        op_id
        for op_id, op_dict in spec["graph"].items()
        if op_dict.get("_op") == "LLMChatOp"
    )

    runtime_graph = RuntimeGraphBuilder().build(compiled)

    assert runtime_graph.nodes[llm_id].api_spec["json"]["model"] == _DEFAULT_API_MODEL


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
    FlowMesh ``${node.path}`` dispatch-time placeholder (resolved by
    FlowMesh's dispatcher before the api executor runs), and the referenced
    node is declared as a FlowMesh dependency."""
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
    (relayed through a ``FormatOp``, the same idiom ``hello-world.yaml`` uses
    to carry an upstream op's output into a message) renders the FlowMesh
    ``${node.text}`` placeholder that FlowMesh's dispatcher resolves against
    the upstream node's real ``APIResult`` before dispatch, and declares
    that upstream node as a dependency."""
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
    """Real job dispatch renames every node via ``RuntimeGraph.with_node_prefix``;
    the FlowMesh dispatcher keys its stage context by that prefixed graph node
    name, so the ``${node.path}`` placeholder an API node emits for an upstream
    runtime reference must be rewritten to the prefixed name too, or dispatch
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


def test_local_node_downstream_of_api_node_uses_text_path() -> None:
    """A local-backend LLMChatOp consuming an API-backed ancestor's output
    (relayed through a ``FormatOp``) renders a ``text`` column (matching
    ``APIResult``'s shape) instead of ``items.output`` (which only
    ``InferenceResult`` carries), so the local worker's graph-template
    renderer does not fail at execution time."""
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


def test_api_ancestor_with_return_history_fails_closed() -> None:
    """``return_history`` needs each item's ``metadata.prompt``, which only
    ``InferenceResult`` (local backend) carries; an API-backed ancestor with
    ``return_history`` enabled must fail closed instead of silently omitting
    chat-history context."""
    stock = input_placeholder("Stock")
    api_node = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        return_history=True,
    )
    relay = FormatOp("{prior}", prior=api_node)
    downstream = LLMChatOp(
        [OpMessage(role="user", content=relay)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
    )
    output = as_output("result", downstream)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    with pytest.raises(ValueError, match="return_history"):
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
    must not replace the always-trusted default https://lum.id (ENV.md's
    documented contract for the env var)."""
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


def test_node_prefix_preserves_api_spec_on_row_fanned_nodes() -> None:
    """RuntimeGraphBuilder.build's node_prefix path (used for real job
    dispatch, e.g. routes/jobs.py) renames every node via
    RuntimeGraph.with_node_prefix; that rename must carry a row-fanned API
    node's api_spec through unchanged, or the prefixed node dispatches with
    no URL/headers/body and the request never goes out."""
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
