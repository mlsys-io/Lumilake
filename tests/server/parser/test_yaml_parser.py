import textwrap

import pytest
from lumilake import envs

from lumilake_server.graphs import Graph
from lumilake_server.ops import InputOp, LLMChatOp, MessageOp, OutputOp
from lumilake_server.parser import parse_yaml_payload


def _ensure_envs() -> None:
    if envs.DATABASE_URL is None:
        envs.DATABASE_URL = "sqlite://"
    if envs.S3_URL is None:
        envs.S3_URL = "s3://dummy"


def test_single_data_op_compiles() -> None:
    yaml_text = textwrap.dedent(
        """
        name: single_data
        ops:
          - id: constant
            op: DataOp
            data: ["hello", "world"]
        outputs:
          - name: out
            ref: constant
        """
    )
    specs = parse_yaml_payload(yaml_text)
    assert set(specs) == {"single_data"}
    spec = specs["single_data"]
    graph = Graph.from_json(spec["graph"])
    assert any(isinstance(op, OutputOp) for op in graph.iter_ops(OutputOp))


def test_inputs_and_outputs_wiring() -> None:
    yaml_text = textwrap.dedent(
        """
        name: echo
        inputs:
          query: ["NVDA"]
        ops:
          - id: formatted
            op: FormatOp
            inputs: [query]
            template: "Q: {q}"
            format_kwargs:
              q: query
        outputs:
          - name: result
            ref: formatted
        """
    )
    specs = parse_yaml_payload(yaml_text)
    spec = specs["echo"]
    graph = Graph.from_json(spec["graph"])

    input_ops = list(graph.iter_ops(InputOp))
    output_ops = list(graph.iter_ops(OutputOp))
    assert len(input_ops) == 1
    assert input_ops[0].name == "query"
    assert len(output_ops) == 1
    assert output_ops[0].name == "result"

    compiled = graph.compile(**spec["inputs"])
    assert compiled is not None


def test_multi_op_topology_inputs() -> None:
    yaml_text = textwrap.dedent(
        """
        name: multi
        inputs:
          q: ["x"]
        ops:
          - id: fmt
            op: FormatOp
            inputs: [q]
            template: "{v}"
            format_kwargs:
              v: q
          - id: summarize
            op: LLMChatOp
            inputs: [fmt]
            messages:
              - role: user
                content: fmt
            config:
              model: dummy-model
              temperature: 0.1
        outputs:
          - name: answer
            ref: summarize
        """
    )
    specs = parse_yaml_payload(yaml_text)
    spec = specs["multi"]

    # Find the LLM op dict (there should be exactly one). Each LLMChatOp has an
    # implicit MessageOp input.
    graph_dict = spec["graph"]
    llm_ops = [op for op in graph_dict.values() if op["_op"] == "LLMChatOp"]
    msg_ops = [op for op in graph_dict.values() if op["_op"] == "MessageOp"]
    fmt_ops = [op for op in graph_dict.values() if op["_op"] == "FormatOp"]
    assert len(llm_ops) == 1
    assert len(msg_ops) == 1
    assert len(fmt_ops) == 1

    llm_op = llm_ops[0]
    msg_op = msg_ops[0]
    fmt_op = fmt_ops[0]

    # LLM -> Message -> Format -> Input.
    assert llm_op["_inputs"] == [msg_op["_id"]]
    assert fmt_op["_id"] in msg_op["_inputs"]
    input_op_ids = [op["_id"] for op in graph_dict.values() if op["_op"] == "InputOp"]
    assert len(input_op_ids) == 1
    assert input_op_ids[0] in fmt_op["_inputs"]

    graph = Graph.from_json(graph_dict)
    assert list(graph.iter_ops(LLMChatOp))
    assert list(graph.iter_ops(MessageOp))


def test_data_retrieval_op_resolves_node_refs() -> None:
    _ensure_envs()
    yaml_text = textwrap.dedent(
        """
        name: retrieve
        inputs:
          entity: ["NVDA"]
        ops:
          - id: retrieval
            op: DataRetrievalOp
            inputs: [entity]
            data_spec:
              type: s3
              connection_string: s3://bucket
              template: "docs/{name}.json"
              params:
                - label: name
                  node: entity
                  path: data
              encoding: utf-8
        outputs:
          - name: docs
            ref: retrieval
        """
    )
    specs = parse_yaml_payload(yaml_text)
    spec = specs["retrieve"]
    graph_dict = spec["graph"]

    retrieval_ops = [op for op in graph_dict.values() if op["_op"] == "DataRetrievalOp"]
    assert len(retrieval_ops) == 1
    retrieval_op = retrieval_ops[0]
    # node ref rewritten from 'entity' to the internal input op id.
    internal_node = retrieval_op["data_spec"]["params"][0]["node"]
    input_ops = [op for op in graph_dict.values() if op["_op"] == "InputOp"]
    assert internal_node == input_ops[0]["_id"]

    graph = Graph.from_json(graph_dict)
    compiled = graph.compile(**spec["inputs"])
    assert compiled is not None


def test_unknown_op_type_raises() -> None:
    yaml_text = textwrap.dedent(
        """
        name: bad
        ops:
          - id: wat
            op: NoSuchOp
        """
    )
    with pytest.raises(ValueError, match="unsupported op type"):
        parse_yaml_payload(yaml_text)


def test_dangling_reference_raises() -> None:
    yaml_text = textwrap.dedent(
        """
        name: dangling
        ops:
          - id: fmt
            op: FormatOp
            inputs: [missing]
            template: "{v}"
            format_kwargs:
              v: missing
        """
    )
    with pytest.raises(ValueError, match="unknown id"):
        parse_yaml_payload(yaml_text)


def test_duplicate_id_raises() -> None:
    yaml_text = textwrap.dedent(
        """
        name: dup
        ops:
          - id: same
            op: DataOp
            data: ["a"]
          - id: same
            op: DataOp
            data: ["b"]
        """
    )
    with pytest.raises(ValueError, match="duplicate id"):
        parse_yaml_payload(yaml_text)


def test_duplicate_id_across_input_and_op_raises() -> None:
    yaml_text = textwrap.dedent(
        """
        name: collision
        inputs:
          shared: ["a"]
        ops:
          - id: shared
            op: DataOp
            data: ["b"]
        """
    )
    with pytest.raises(ValueError, match="duplicate id"):
        parse_yaml_payload(yaml_text)


def test_accepts_dict_payload() -> None:
    payload = {
        "name": "from_dict",
        "ops": [
            {"id": "d", "op": "DataOp", "data": ["x"]},
        ],
        "outputs": [{"name": "out", "ref": "d"}],
    }
    specs = parse_yaml_payload(payload)
    assert set(specs) == {"from_dict"}


def test_output_refers_unknown_id_raises() -> None:
    yaml_text = textwrap.dedent(
        """
        name: out_bad
        ops:
          - id: d
            op: DataOp
            data: ["x"]
        outputs:
          - name: out
            ref: nope
        """
    )
    with pytest.raises(ValueError, match="output .* unknown id"):
        parse_yaml_payload(yaml_text)


def test_data_spec_param_node_must_reference_known_id() -> None:
    """A dangling ``data_spec.params[*].node`` ref raises loudly, not silently."""
    _ensure_envs()
    yaml_text = textwrap.dedent(
        """
        name: bad_param
        inputs:
          query: ["x"]
        ops:
          - id: retrieve
            op: DataRetrievalOp
            inputs: [query]
            data_spec:
              type: s3
              connection_string: s3://bucket
              template: docs/{entity}.json
              params:
                - { label: entity, node: does_not_exist, path: data }
        """
    )
    with pytest.raises(ValueError, match="references unknown input/op id"):
        parse_yaml_payload(yaml_text)


def test_messages_ref_must_point_at_message_op() -> None:
    """``messages_ref`` pointing at a non-MessageOp must raise."""
    _ensure_envs()
    yaml_text = textwrap.dedent(
        """
        name: wrong_ref_type
        inputs:
          q: ["hi"]
        ops:
          - id: not_a_message
            op: DataOp
            data: ["x"]
          - id: answer
            op: LLMChatOp
            inputs: [q]
            messages_ref: not_a_message
            config:
              model: meta-llama/Llama-3.1-8B-Instruct
        """
    )
    with pytest.raises(ValueError, match="must reference a MessageOp"):
        parse_yaml_payload(yaml_text)


def test_llm_chat_bare_user_id_content_wraps_in_implicit_format_op() -> None:
    """A user-role message with content = bare upstream id auto-wraps into a FormatOp.

    Pins the behaviour the module docstring promises: ``content: retrieve``
    does NOT pass the literal string "retrieve" through — it detects that
    "retrieve" is a user-facing op id and synthesizes an implicit FormatOp
    whose ``_inputs`` point at that op. The MessageOp's content then points
    at the FormatOp, not the original op or the literal string.
    """
    _ensure_envs()
    yaml_text = textwrap.dedent(
        """
        name: ex
        inputs:
          query: ["NVDA"]
        ops:
          - id: retrieve
            op: DataRetrievalOp
            inputs: [query]
            data_spec:
              type: s3
              connection_string: s3://bucket
              template: docs/{entity}.json
              params:
                - { label: entity, node: query, path: data }
          - id: summarize
            op: LLMChatOp
            inputs: [retrieve]
            messages:
              - { role: system, content: "You summarize documents." }
              - { role: user, content: retrieve }
            config: { model: meta-llama/Llama-3.1-8B-Instruct }
        outputs:
          - { name: out, ref: summarize }
        """
    )
    specs = parse_yaml_payload(yaml_text)
    graph = specs["ex"]["graph"]
    op_types = {op["_op"] for op in graph.values()}
    # An implicit FormatOp must have been synthesized for the bare ref.
    assert "FormatOp" in op_types
    format_ops = [op for op in graph.values() if op["_op"] == "FormatOp"]
    assert len(format_ops) == 1
    fmt = format_ops[0]
    retrieve_id = next(
        op["_id"] for op in graph.values() if op["_op"] == "DataRetrievalOp"
    )
    assert fmt["_inputs"] == [retrieve_id]
    assert fmt["template"] == "{ref0}"
    assert fmt["format_kwargs"] == {"ref0": retrieve_id}

    # MessageOp's user message now points at the FormatOp (not the literal
    # "retrieve" string, and not the DataRetrievalOp directly).
    msg_op = next(op for op in graph.values() if op["_op"] == "MessageOp")
    user_msg = next(m for m in msg_op["messages"] if m["role"] == "user")
    assert user_msg["content"] == fmt["_id"]
    # The literal system-message stays as-is, no wrapping.
    sys_msg = next(m for m in msg_op["messages"] if m["role"] == "system")
    assert sys_msg["content"] == "You summarize documents."
