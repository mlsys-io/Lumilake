"""Focused tests for raw string-prompt LLM generation."""

import textwrap

import pytest

from lumilake_server.common import ApiConfig, GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import LambdaOp, LLMCompletionOp, as_output, input_placeholder
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder


def test_completion_op_compiles_to_string_prompt_list_without_chat_template() -> None:
    op = LLMCompletionOp(
        prompt=["first raw prompt", "second raw prompt"],
        config=GenerationConfig(
            model="Qwen/Qwen3-8B",
            temperature=0.7,
            max_tokens=3000,
            extra_sampling_params={"stop_token_ids": [128009]},
        ),
    )
    compiled = Graph.from_ops([as_output("answer", op)]).compile()
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    node = runtime_graph.nodes[op.id]

    assert node.data_spec == {
        "type": "list",
        "items": ["first raw prompt", "second raw prompt"],
    }
    assert node.inference_spec["apply_chat_template"] is False
    assert node.inference_spec["temperature"] == 0.7
    assert node.inference_spec["max_tokens"] == 3000
    assert node.inference_spec["stop_token_ids"] == [128009]


def test_completion_yaml_uses_string_prompt_and_no_chat_template() -> None:
    yaml_text = textwrap.dedent("""
        name: raw_prompts
        inputs:
          Prompt: ["first raw prompt", "second raw prompt"]
        ops:
          - id: Generate
            op: LLMCompletionOp
            inputs: [Prompt]
            prompt: Prompt
            config:
              model: Qwen/Qwen3-8B
              temperature: 0.7
              max_tokens: 3000
              extra_sampling_params:
                stop_token_ids: [128009]
        outputs:
          - name: answer
            ref: Generate
        """)
    spec = parse_yaml_payload(yaml_text)["raw_prompts"]
    compiled = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    [node] = [
        node for node in runtime_graph.nodes.values() if node.task_type == "inference"
    ]

    assert node.data_spec == {
        "type": "list",
        "items": ["first raw prompt", "second raw prompt"],
    }
    assert node.inference_spec["apply_chat_template"] is False
    assert node.inference_spec["stop_token_ids"] == [128009]


def test_completion_prompt_from_upstream_lambda_binds_exact_runtime_path() -> None:
    prompt = input_placeholder("Prompt")
    render = LambdaOp([prompt], fn=lambda inputs: str(inputs[0]).upper())
    completion = LLMCompletionOp(
        prompt=render,
        config=GenerationConfig(model="Qwen/Qwen3-8B"),
    )
    compiled = Graph.from_ops([as_output("answer", completion)]).compile(
        Prompt=["first", "second"]
    )
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    node = runtime_graph.nodes[completion.id]

    assert node.data_spec == {
        "type": "list",
        "node": render.id,
        "path": "value.items.output",
    }
    assert render.id in node.dependencies
    assert node.inference_spec["apply_chat_template"] is False


def test_completion_prompt_from_list_mode_lambda_skips_message_inference() -> None:
    prompt = input_placeholder("Prompt")
    render = LambdaOp(
        [prompt],
        fn=lambda inputs: [f"{value}?" for value in inputs[0]],
        mode="list",
    )
    completion = LLMCompletionOp(
        prompt=render,
        config=GenerationConfig(model="Qwen/Qwen3-8B"),
    )
    compiled = Graph.from_ops([as_output("answer", completion)]).compile(
        Prompt=["first", "second"]
    )
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    node = runtime_graph.nodes[completion.id]

    assert node.data_spec == {
        "type": "list",
        "node": render.id,
        "path": "value.items.output",
    }
    assert node.inference_spec["apply_chat_template"] is False


def test_completion_op_rejects_api_mode() -> None:
    with pytest.raises(ValueError, match="API mode"):
        LLMCompletionOp(
            prompt=["raw"],
            config=GenerationConfig(
                model="org/model",
                api=ApiConfig(url="https://example.invalid/v1"),
            ),
        )


def test_completion_op_rejects_chat_template_kwargs() -> None:
    with pytest.raises(ValueError, match="chat_template_kwargs"):
        LLMCompletionOp(
            prompt=["raw"],
            config=GenerationConfig(
                model="Qwen/Qwen3-8B",
                chat_template_kwargs={"enable_thinking": False},
            ),
        )
