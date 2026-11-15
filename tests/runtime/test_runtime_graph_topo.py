"""Guard for ``RuntimeGraph.topological_order`` filtering out non-node
deps, so callers that index back into ``graph.nodes`` don't see KeyError.

Runtime nodes can carry dependencies on DSL-layer op ids that never
materialize as runtime nodes (e.g. an ``InputOp`` that a retrieval op's
``data_spec.params[*].node`` references). The topo sort needs those
edges to preserve ordering, but the returned list must only contain
real node ids.
"""

from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp


def _make_op(node_id: str, dependencies: tuple[str, ...] = ()) -> RuntimeOp:
    return RuntimeOp(
        node_id=node_id,
        task_type="data_retrieval",
        backend="data_retrieval",
        model="data_retrieval",
        data_spec={"type": "s3", "connection_string": "s3://b", "template": "x"},
        model_spec={},
        inference_spec={},
        dependencies=dependencies,
    )


class TestTopologicalOrderFilter:
    def test_filters_out_non_node_deps(self) -> None:
        """A dependency that names a DSL op id with no matching runtime
        node must not appear in the returned list — even though the
        topo-sort graph knows about it for ordering."""
        graph = RuntimeGraph(
            nodes={
                "a": _make_op("a", dependencies=("external_input_op",)),
                "b": _make_op("b", dependencies=("a",)),
            },
            node_order=["a", "b"],
            output_node_map={"answer": "b"},
            dsl_to_runtime={},
        )
        order = graph.topological_order()
        assert "external_input_op" not in order
        assert set(order) == {"a", "b"}
        # Canonical ordering: dep first.
        assert order.index("a") < order.index("b")

    def test_no_external_deps_returns_all_nodes(self) -> None:
        graph = RuntimeGraph(
            nodes={"a": _make_op("a"), "b": _make_op("b", dependencies=("a",))},
            node_order=["a", "b"],
            output_node_map={},
            dsl_to_runtime={},
        )
        order = graph.topological_order()
        assert order == ["a", "b"]


class TestFlowmeshDepsFilter:
    """``to_flowmesh_nodes`` must strip deps that reference DSL ops never
    materialized as FlowMesh tasks — FlowMesh holds a task in PENDING
    forever when any listed dep is unknown to it."""

    def test_strips_non_runtime_deps_from_flowmesh_payload(self) -> None:
        graph = RuntimeGraph(
            nodes={
                "a": _make_op("a", dependencies=("external_input",)),
                "b": _make_op("b", dependencies=("a", "external_input")),
            },
            node_order=["a", "b"],
            output_node_map={"answer": "b"},
            dsl_to_runtime={},
        )
        flowmesh_nodes = graph.to_flowmesh_nodes()
        a_payload = next(n for n in flowmesh_nodes if n["name"] == "a")
        b_payload = next(n for n in flowmesh_nodes if n["name"] == "b")
        # ``a``'s only dep was the DSL op → removed, no ``dependsOn`` key.
        assert "dependsOn" not in a_payload
        # ``b`` keeps the real dep on ``a`` and drops ``external_input``.
        assert b_payload["dependsOn"] == ["a"]

    def test_preserves_real_runtime_deps(self) -> None:
        graph = RuntimeGraph(
            nodes={
                "a": _make_op("a"),
                "b": _make_op("b", dependencies=("a",)),
            },
            node_order=["a", "b"],
            output_node_map={},
            dsl_to_runtime={},
        )
        payloads = graph.to_flowmesh_nodes()
        b_payload = next(n for n in payloads if n["name"] == "b")
        assert b_payload["dependsOn"] == ["a"]
