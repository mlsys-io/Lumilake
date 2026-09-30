from lumilake_server.runtime.runtime_ops import RuntimeOp


def _build_runtime_op(task_type: str) -> RuntimeOp:
    return RuntimeOp(
        node_id="n1",
        task_type=task_type,
        backend="transformers",
        model="test-model",
        data_spec={"type": "list", "items": ["hello"]},
        model_spec={"source": {"identifier": "test-model"}},
        inference_spec={"max_tokens": 16},
    )


def test_data_retrieval_excludes_model_and_inference() -> None:
    node = _build_runtime_op("data_retrieval").to_flowmesh_node()

    spec = node["spec"]
    assert spec["taskType"] == "data_retrieval"
    assert "model" not in spec
    assert "inference" not in spec
    assert "state" not in spec


def test_embedding_includes_model_and_excludes_inference() -> None:
    node = _build_runtime_op("embedding").to_flowmesh_node()

    spec = node["spec"]
    assert spec["taskType"] == "embedding"
    assert "model" in spec
    assert "inference" not in spec


def test_inference_includes_model_and_inference() -> None:
    node = _build_runtime_op("inference").to_flowmesh_node()

    spec = node["spec"]
    assert spec["taskType"] == "inference"
    assert "model" in spec
    assert "inference" in spec


def test_diffusion_includes_model_and_inference() -> None:
    node = _build_runtime_op("diffusion").to_flowmesh_node()

    spec = node["spec"]
    assert spec["taskType"] == "diffusion"
    assert "model" in spec
    assert "inference" in spec


def test_condition_node_is_added_to_dependsOn() -> None:
    op = RuntimeOp(
        node_id="map-gated",
        task_type="inference",
        backend="vllm",
        model="Qwen/Qwen3-32B",
        data_spec={"type": "list", "items": ["hello"]},
        model_spec={"source": {"identifier": "Qwen/Qwen3-32B"}},
        inference_spec={"max_tokens": 1},
        dependencies=("some-upstream",),
        condition={
            "node": "classifier-op",
            "field": "routing_key",
            "equals": "global-1",
        },
    )
    node = op.to_flowmesh_node()

    deps = node["dependsOn"]
    assert "some-upstream" in deps
    assert (
        "classifier-op" in deps
    ), f"condition.node must appear in dependsOn; got: {deps!r}"
    # Condition spec itself is still emitted.
    assert node["spec"]["condition"]["node"] == "classifier-op"


def test_condition_node_already_in_dependencies_is_not_duplicated() -> None:
    op = RuntimeOp(
        node_id="map-gated",
        task_type="inference",
        backend="vllm",
        model="Qwen/Qwen3-32B",
        data_spec={"type": "list", "items": ["x"]},
        model_spec={"source": {"identifier": "Qwen/Qwen3-32B"}},
        inference_spec={"max_tokens": 1},
        dependencies=("classifier-op", "other-upstream"),
        condition={"node": "classifier-op", "field": "routing_key", "equals": "local"},
    )
    node = op.to_flowmesh_node()

    deps = node["dependsOn"]
    assert deps.count("classifier-op") == 1
    assert "other-upstream" in deps
