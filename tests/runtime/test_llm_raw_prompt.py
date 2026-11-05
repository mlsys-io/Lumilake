"""Focused tests for raw string-prompt LLM generation."""

import textwrap

import pytest

from lumilake_server.common import GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import LLMRawChatOp, as_output
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder


def test_raw_chat_op_compiles_to_string_prompt_list_without_chat_template() -> None:
    op = LLMRawChatOp(
        prompt=["first raw prompt", "second raw prompt"],
        config=GenerationConfig(
            model="OpenSciLM/Llama-3.1_OpenScholar-8B",
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


def test_raw_chat_yaml_uses_string_prompt_and_no_chat_template() -> None:
    yaml_text = textwrap.dedent("""
        name: raw_prompts
        inputs:
          Prompt: ["first raw prompt", "second raw prompt"]
        ops:
          - id: Generate
            op: LLMRawChatOp
            inputs: [Prompt]
            prompt: Prompt
            config:
              model: OpenSciLM/Llama-3.1_OpenScholar-8B
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


def test_raw_chat_op_rejects_api_mode() -> None:
    with pytest.raises(ValueError, match="API mode"):
        LLMRawChatOp(
            prompt=["raw"],
            config=GenerationConfig(model="org/model", api={}),  # type: ignore[arg-type]
        )
