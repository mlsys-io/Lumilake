import pytest
from lumilake import envs

from lumilake_server.common import GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
    DataRetrievalOp,
    ImageGenerationOp,
    LLMChatOp,
    LLMVisionOp,
    OpMessage,
    as_output,
    input_placeholder,
)
from lumilake_server.parser.n8n import parse_n8n_payload
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder

_LUMID_URL = "http://lumid-data"
_LUMID_TOKEN = "test-token"


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", _LUMID_URL)
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", _LUMID_TOKEN)


def _sql_spec(template: str, params: list) -> dict:
    return {
        "type": "lumid",
        "mode": "sql",
        "template": template,
        "params": params,
    }


def _s3_spec(template: str, params: list) -> dict:
    return {
        "type": "lumid",
        "mode": "s3",
        "template": template,
        "params": params,
    }


def test_runtime_graph_builder_accepts_input_used_only_by_retrieval() -> None:
    stock = input_placeholder("Stock")
    retrieval = DataRetrievalOp(
        data_spec=_sql_spec(
            "SELECT * FROM t WHERE symbol = :symbol",
            [{"name": "symbol", "node": stock.id}],
        ),
        inputs=[stock],
    )
    llm = LLMChatOp(
        [OpMessage(role="user", content=retrieval)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)

    assert len(runtime_graph.nodes) == 2
    assert retrieval.id in runtime_graph.nodes
    assert llm.id in runtime_graph.nodes
    # Verify lumid wire format
    ds = runtime_graph.nodes[retrieval.id].data_spec
    assert ds["type"] == "lumid"
    assert ds["mode"] == "sql"
    assert ds["lumid_data_url"] == _LUMID_URL
    assert ds["lumid_data_token"] == _LUMID_TOKEN
    assert "connection_string" not in ds


def test_runtime_graph_builder_emits_topological_runtime_node_order() -> None:
    stock = input_placeholder("Stock")
    planner = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
    )
    retrieval = DataRetrievalOp(
        data_spec=_sql_spec(
            "SELECT * FROM t WHERE query = :query",
            [{"label": "query", "node": planner.id, "path": "items.output"}],
        ),
        inputs=[planner],
    )
    report = LLMChatOp(
        [OpMessage(role="user", content=retrieval)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
    )
    output = as_output("result", report)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)

    order_index = {node_id: idx for idx, node_id in enumerate(runtime_graph.node_order)}
    assert runtime_graph.node_order == runtime_graph.topological_order()
    assert order_index[planner.id] < order_index[retrieval.id]
    assert order_index[retrieval.id] < order_index[report.id]


def test_runtime_graph_builder_supports_s3_retrieval_as_vlm_image_source() -> None:
    stock = input_placeholder("Stock")
    retrieval = DataRetrievalOp(
        data_spec=_s3_spec(
            "example-data/news/images/{symbol}.png",
            [{"name": "symbol", "node": stock.id}],
        ),
        inputs=[stock],
    )
    vision = LLMVisionOp(
        [OpMessage(role="user", content="Describe the image briefly.")],
        image_source=retrieval.id,
        image_source_op=retrieval,
        config=GenerationConfig(model="llava-hf/llava-1.5-7b-hf"),
    )
    output = as_output("result", vision)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)

    embedding_id = f"{vision.id}_embedding"
    assert embedding_id in runtime_graph.nodes
    embedding = runtime_graph.nodes[embedding_id]
    assert embedding.task_type == "embedding"
    assert embedding.data_spec["node"] == retrieval.id
    assert embedding.data_spec["path"] == "items.content"
    assert embedding.dependencies == (retrieval.id,)


def test_runtime_graph_builder_supports_rowwise_vlm_template() -> None:
    stock = input_placeholder("Stock")
    news_sql = DataRetrievalOp(
        data_spec=_sql_spec(
            "SELECT title FROM news WHERE symbol = :symbol",
            [{"name": "symbol", "node": stock.id}],
        ),
        inputs=[stock],
    )
    news_s3 = DataRetrievalOp(
        data_spec=_s3_spec(
            "example-data/news/images/{symbol}.png",
            [{"name": "symbol", "node": stock.id}],
        ),
        inputs=[stock],
    )
    vision = LLMVisionOp(
        [OpMessage(role="user", content="ignored")],
        image_source=news_s3.id,
        image_source_op=news_s3,
        config=GenerationConfig(model="llava-hf/llava-1.5-7b-hf"),
        rowwise_template="Summarize {Stock} with title {title}.",
        rowwise_columns=[
            {"label": "Stock", "data": {"type": "list", "items": ["NVDA"]}},
            {
                "label": "title",
                "node": news_sql.id,
                "path": "items.table.title",
            },
        ],
        system_messages=["You are concise."],
    )
    output = as_output("result", vision)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)

    embedding_id = f"{vision.id}_embedding"
    vlm = runtime_graph.nodes[vision.id]
    template = vlm.data_spec["template"]
    columns = template["columns"]
    assert {"label": "Stock", "data": {"type": "list", "items": ["NVDA"]}} in columns
    assert {
        "label": "title",
        "node": news_sql.id,
        "path": "items.table.title",
    } in columns
    assert {"role": "user", "content": "Summarize {Stock} with title {title}."} in (
        template["options"]["format"]["messages"]
    )
    assert embedding_id in vlm.dependencies
    assert news_sql.id in vlm.dependencies


def test_runtime_graph_builder_omits_unused_upstream_context_column() -> None:
    workflow = {
        "nodes": [
            {
                "parameters": {"options": {}},
                "type": "@n8n/n8n-nodes-langchain.chatTrigger",
                "name": "Input",
                "id": "input-node",
                "typeVersion": 1.4,
            },
            {
                "parameters": {
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "options": {"maxTokens": 64},
                },
                "type": "@n8n/n8n-nodes-langchain.lmOpenHuggingFaceInference",
                "name": "Model",
                "id": "model-node",
                "typeVersion": 1,
            },
            {
                "parameters": {
                    "promptType": "define",
                    "text": "=First stage for {{ $('Input') }}",
                    "messages": {"messageValues": []},
                },
                "type": "@n8n/n8n-nodes-langchain.chainLlm",
                "name": "Upstream",
                "id": "upstream-node",
                "typeVersion": 1.9,
                "notes": '{"op-type":"text-generation","is-output":false}',
            },
            {
                "parameters": {
                    "promptType": "define",
                    "text": "=Second stage for {{ $('Upstream') }}",
                    "messages": {"messageValues": []},
                },
                "type": "@n8n/n8n-nodes-langchain.chainLlm",
                "name": "Downstream",
                "id": "downstream-node",
                "typeVersion": 1.9,
                "notes": '{"op-type":"text-generation","is-output":false}',
            },
        ],
        "connections": {
            "Input": {"main": [[{"node": "Upstream", "type": "main", "index": 0}]]},
            "Upstream": {
                "main": [[{"node": "Downstream", "type": "main", "index": 0}]]
            },
            "Model": {
                "ai_languageModel": [
                    [
                        {"node": "Upstream", "type": "ai_languageModel", "index": 0},
                        {"node": "Downstream", "type": "ai_languageModel", "index": 0},
                    ]
                ]
            },
        },
    }
    payload = {
        "graphs": [
            {"name": "graph", "workflow": workflow, "inputs": {"Input": ["NVDA"]}}
        ]
    }
    parsed = parse_n8n_payload(payload)
    spec = parsed["graph"]
    compiled = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    upstream_id = next(nid for nid in runtime_graph.nodes if "Upstream" in nid)
    downstream_id = next(nid for nid in runtime_graph.nodes if "Downstream" in nid)
    columns = runtime_graph.nodes[downstream_id].data_spec["template"]["columns"]
    upstream_columns = [col for col in columns if col.get("node") == upstream_id]
    assert any(col.get("path") == "items.output" for col in upstream_columns)
    assert all(col.get("path") != "items.metadata.prompt" for col in upstream_columns)


def test_attach_lumid_cfg_injected_for_s3_image_inputs() -> None:
    images = input_placeholder("Images")
    vision = LLMVisionOp(
        [OpMessage(role="user", content="Describe the image.")],
        image_source=images.id,
        image_source_op=images,
        config=GenerationConfig(model="llava-hf/llava-1.5-7b-hf"),
    )
    output = as_output("result", vision)
    compiled = Graph.from_ops([output]).compile(
        Images=["s3://bucket/img/a.png", "s3://bucket/img/b.png"]
    )

    runtime_graph = RuntimeGraphBuilder().build(compiled)

    embedding_id = f"{vision.id}_embedding"
    embedding = runtime_graph.nodes[embedding_id]
    ds = embedding.data_spec
    assert ds["type"] == "list"
    assert "lumid_cfg" in ds
    assert ds["lumid_cfg"]["lumid_data_url"] == _LUMID_URL
    assert ds["lumid_cfg"]["lumid_data_token"] == _LUMID_TOKEN
    assert ds["lumid_cfg"]["encoding"] == "utf-8"
    assert "s3_cfg" not in ds


def test_attach_lumid_cfg_not_injected_for_non_s3_list_inputs() -> None:
    texts = input_placeholder("Texts")
    llm = LLMChatOp(
        [OpMessage(role="user", content=texts)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Texts=["hello", "world"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)

    llm_node = runtime_graph.nodes[llm.id]
    ds = llm_node.data_spec
    assert "lumid_cfg" not in ds
    assert "s3_cfg" not in ds


def test_runtime_graph_propagates_extended_sampler_fields() -> None:
    stock = input_placeholder("Stock")
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            max_tokens=256,
            min_tokens=4,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5,
            frequency_penalty=0.5,
            repetition_penalty=1.0,
            seed=42,
            chat_template_kwargs={"enable_thinking": False},
            extra_sampling_params={"length_penalty": 1.1},
        ),
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    spec = runtime_graph.nodes[llm.id].inference_spec
    assert spec["max_tokens"] == 256
    assert spec["min_tokens"] == 4
    assert spec["temperature"] == 0.7
    assert spec["top_p"] == 0.8
    assert spec["top_k"] == 20
    assert spec["min_p"] == 0.0
    assert spec["presence_penalty"] == 1.5
    assert spec["frequency_penalty"] == 0.5
    assert spec["repetition_penalty"] == 1.0
    assert spec["seed"] == 42
    assert spec["chat_template_kwargs"] == {"enable_thinking": False}
    # extra_sampling_params keys are merged into the spec as flat keys.
    assert spec["length_penalty"] == 1.1
    # model never appears in inference_spec — it's carried separately.
    assert "model" not in spec
    # The escape-hatch wrapper key itself is not leaked into the spec.
    assert "extra_sampling_params" not in spec


def _build_llm_graph(**config_kwargs):
    stock = input_placeholder("Stock")
    cfg = GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct", **config_kwargs)
    llm = LLMChatOp([OpMessage(role="user", content=stock)], config=cfg)
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    return RuntimeGraphBuilder().build(compiled), llm.id


_ENGINE_FIELDS = (
    "max_model_len",
    "gpu_memory_utilization",
    "tensor_parallel_size",
    "dtype",
)


def test_engine_overlay_lands_in_vllm_backend() -> None:
    runtime_graph, llm_id = _build_llm_graph(
        max_model_len=8192,
        gpu_memory_utilization=0.75,
        tensor_parallel_size=2,
        dtype="bf16",
        extra_engine_kwargs={"quantization": "fp8", "kv_cache_dtype": "fp8"},
    )
    node = runtime_graph.nodes[llm_id]
    vllm_cfg = node.model_spec["vllm"]
    assert vllm_cfg["max_model_len"] == 8192
    assert vllm_cfg["gpu_memory_utilization"] == 0.75
    assert vllm_cfg["tensor_parallel_size"] == 2
    assert vllm_cfg["dtype"] == "bf16"
    assert vllm_cfg["quantization"] == "fp8"
    assert vllm_cfg["kv_cache_dtype"] == "fp8"
    # Engine fields must NOT leak into inference_spec.
    for k in _ENGINE_FIELDS + ("quantization", "kv_cache_dtype"):
        assert k not in node.inference_spec


def test_engine_overlay_omitted_when_unset() -> None:
    # gpu_memory_utilization has an env-var default in the baseline vLLM
    # config, so unset means "stays at env default" not "absent". The other
    # three fields have no default and should not appear when unset.
    runtime_graph, llm_id = _build_llm_graph()
    vllm_cfg = runtime_graph.nodes[llm_id].model_spec["vllm"]
    for k in ("max_model_len", "tensor_parallel_size", "dtype"):
        assert k not in vllm_cfg


def test_engine_overlay_overrides_baseline_default() -> None:
    # gpu_memory_utilization is set in the baseline config; an explicit
    # value on GenerationConfig must override it.
    runtime_graph, llm_id = _build_llm_graph(gpu_memory_utilization=0.6)
    vllm_cfg = runtime_graph.nodes[llm_id].model_spec["vllm"]
    assert vllm_cfg["gpu_memory_utilization"] == 0.6


def test_vllm_max_model_len_env_caps_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # LUMILAKE_VLLM_MAX_MODEL_LEN > 0 caps max_model_len in the baseline vLLM
    # config when GenerationConfig sets none, so a model's native context can't
    # demand a KV cache larger than a smaller-VRAM GPU allows.
    monkeypatch.setattr(envs, "LUMILAKE_VLLM_MAX_MODEL_LEN", 8192)
    runtime_graph, llm_id = _build_llm_graph()
    vllm_cfg = runtime_graph.nodes[llm_id].model_spec["vllm"]
    assert vllm_cfg["max_model_len"] == 8192


def test_vllm_max_model_len_env_overridden_by_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An explicit GenerationConfig max_model_len wins over the env cap.
    monkeypatch.setattr(envs, "LUMILAKE_VLLM_MAX_MODEL_LEN", 8192)
    runtime_graph, llm_id = _build_llm_graph(max_model_len=4096)
    vllm_cfg = runtime_graph.nodes[llm_id].model_spec["vllm"]
    assert vllm_cfg["max_model_len"] == 4096


def test_extra_sampling_params_conflict_rejected() -> None:
    stock = input_placeholder("Stock")
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            temperature=0.7,
            extra_sampling_params={"temperature": 0.0},
        ),
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    with pytest.raises(ValueError, match="extra_sampling_params conflict"):
        RuntimeGraphBuilder().build(compiled)


def test_extra_engine_kwargs_conflict_rejected() -> None:
    stock = input_placeholder("Stock")
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            max_model_len=8192,
            extra_engine_kwargs={"max_model_len": 4096},
        ),
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    with pytest.raises(ValueError, match="extra_engine_kwargs conflict"):
        RuntimeGraphBuilder().build(compiled)


def test_inference_spec_omits_unset_defaults() -> None:
    runtime_graph, llm_id = _build_llm_graph()
    spec = runtime_graph.nodes[llm_id].inference_spec
    # n/stream/ignore_eos must not be emitted when the user did not set them.
    assert "n" not in spec
    assert "stream" not in spec
    assert "stream_options" not in spec
    assert "ignore_eos" not in spec


def test_inference_spec_omits_stream_even_when_set() -> None:
    # stream is a request-mode flag, not a sampler knob; it must never leak
    # into the runtime inference spec sent to the worker.
    runtime_graph, llm_id = _build_llm_graph(stream=True)
    spec = runtime_graph.nodes[llm_id].inference_spec
    assert "stream" not in spec


def test_inference_spec_emits_explicit_n_and_ignore_eos() -> None:
    runtime_graph, llm_id = _build_llm_graph(n=2, ignore_eos=True)
    spec = runtime_graph.nodes[llm_id].inference_spec
    assert spec["n"] == 2
    assert spec["ignore_eos"] is True


def test_diffusers_backend_applies_dtype_and_extra_engine_kwargs() -> None:
    builder = RuntimeGraphBuilder()
    config = GenerationConfig(
        model="stabilityai/sdxl",
        dtype="fp16",
        extra_engine_kwargs={"use_safetensors": False, "variant": "fp16"},
    )
    spec = builder._build_model_spec(config, backend="diffusers")
    diffusers_cfg = spec["diffusers"]
    assert diffusers_cfg["dtype"] == "fp16"
    assert diffusers_cfg["use_safetensors"] is False
    assert diffusers_cfg["variant"] == "fp16"


def test_diffusers_backend_rejects_vllm_only_engine_fields() -> None:
    builder = RuntimeGraphBuilder()
    config = GenerationConfig(
        model="stabilityai/sdxl",
        max_model_len=8192,
    )
    with pytest.raises(
        ValueError, match="diffusers backend does not accept typed engine fields"
    ):
        builder._build_model_spec(config, backend="diffusers")


def _build_image_gen_graph(**config_kwargs):
    prompt = input_placeholder("Prompt")
    cfg = GenerationConfig(model="stabilityai/sdxl", **config_kwargs)
    img = ImageGenerationOp(content=prompt, config=cfg)
    output = as_output("result", img)
    compiled = Graph.from_ops([output]).compile(Prompt=["a cat"])
    return RuntimeGraphBuilder().build(compiled), img.id


def test_image_gen_defaults_emitted_when_user_omits() -> None:
    runtime_graph, img_id = _build_image_gen_graph()
    spec = runtime_graph.nodes[img_id].inference_spec
    assert spec["num_inference_steps"] == 8
    assert spec["guidance_scale"] == 1.0
    assert spec["height"] == 1024
    assert spec["width"] == 1024


def test_image_gen_user_extras_override_defaults() -> None:
    runtime_graph, img_id = _build_image_gen_graph(
        extra_sampling_params={
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "height": 768,
            "width": 768,
        },
    )
    spec = runtime_graph.nodes[img_id].inference_spec
    assert spec["num_inference_steps"] == 30
    assert spec["guidance_scale"] == 7.5
    assert spec["height"] == 768
    assert spec["width"] == 768
