"""Coverage for data-profile placeholder resolution in
``RuntimeGraphBuilder``.

Exercises the construction-time fallback for ``LLMChatOp`` structural
outputs missing ``min``/``max``/``candidates``, the upstream
``DataRetrievalOp`` ``sample_value`` short-circuit, the live-sample
path with the SQL sampler hook, the recursive resolver across two
``DataRetrievalOp`` hops, the cycle guard, and the explicit rejection
for unsupported upstream op kinds.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import psycopg.errors
import pytest
from lumilake import envs
from lumilake.envs import _positive_float

from lumilake_server.common import GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
    DataRetrievalOp,
    FormatOp,
    LLMChatOp,
    OpMessage,
    as_output,
    input_placeholder,
)
from lumilake_server.runtime import runtime_graph as rg
from lumilake_server.runtime.runtime_graph import _default_sql_sampler
from lumilake_server.utils.data_profile_offload import (
    _build_sample_data_profile_queries,
    _collect_data_profile_param_candidates,
)


def _build_profile_node(compiled: Any, op_id: str) -> Any:
    builder = rg.RuntimeGraphBuilder()
    runtime = builder.build(compiled, task_type_override="data_profile")
    return runtime.nodes[op_id]


class TestStructuralOutputTypeDefault:
    def test_string_field_without_constraints_synthesizes_empty_string(self) -> None:
        stock = input_placeholder("Stock")
        planner = LLMChatOp(
            [OpMessage(role="user", content=stock)],
            config=GenerationConfig(model="x"),
            structural_outputs=[{"name": "code", "type": "string"}],
        )
        retrieval = DataRetrievalOp(
            data_spec={
                "type": "sql",
                "connection_string": "postgresql://example",
                "template": (
                    "SELECT * FROM t WHERE symbol='{symbol}' " "AND code='{code}'"
                ),
                "params": [
                    {"label": "symbol", "node": stock.id},
                    {
                        "label": "code",
                        "node": planner.id,
                        "path": "items.output.code",
                    },
                ],
            },
            inputs=[stock, planner],
        )
        report = LLMChatOp(
            [OpMessage(role="user", content=retrieval)],
            config=GenerationConfig(model="x"),
        )
        compiled = Graph.from_ops([as_output("r", report)]).compile(Stock=["NVDA"])
        node = _build_profile_node(compiled, retrieval.id)
        queries = _build_sample_data_profile_queries(
            template=node.data_spec["template"],
            params=node.data_spec["params"],
            constraints=node.data_spec.get("constraints"),
            num_samples=1,
            node_id=retrieval.id,
        )
        assert queries == ["SELECT * FROM t WHERE symbol='NVDA' AND code=''"]

    def test_int_field_defaults_to_zero(self) -> None:
        stock = input_placeholder("Stock")
        planner = LLMChatOp(
            [OpMessage(role="user", content=stock)],
            config=GenerationConfig(model="x"),
            structural_outputs=[{"name": "n", "type": "int"}],
        )
        retrieval = DataRetrievalOp(
            data_spec={
                "type": "sql",
                "connection_string": "postgresql://example",
                "template": "SELECT * FROM t WHERE n={n}",
                "params": [
                    {
                        "label": "n",
                        "node": planner.id,
                        "path": "items.output.n",
                    }
                ],
            },
            inputs=[stock, planner],
        )
        report = LLMChatOp(
            [OpMessage(role="user", content=retrieval)],
            config=GenerationConfig(model="x"),
        )
        compiled = Graph.from_ops([as_output("r", report)]).compile(Stock=["NVDA"])
        node = _build_profile_node(compiled, retrieval.id)
        queries = _build_sample_data_profile_queries(
            template=node.data_spec["template"],
            params=node.data_spec["params"],
            constraints=node.data_spec.get("constraints"),
            num_samples=1,
            node_id=retrieval.id,
        )
        assert queries == ["SELECT * FROM t WHERE n=0"]

    def test_candidates_take_precedence(self) -> None:
        stock = input_placeholder("Stock")
        planner = LLMChatOp(
            [OpMessage(role="user", content=stock)],
            config=GenerationConfig(model="x"),
            structural_outputs=[
                {
                    "name": "side",
                    "type": "string",
                    "candidates": ["BUY", "SELL"],
                }
            ],
        )
        retrieval = DataRetrievalOp(
            data_spec={
                "type": "sql",
                "connection_string": "postgresql://example",
                "template": "SELECT * FROM t WHERE side='{side}'",
                "params": [
                    {
                        "label": "side",
                        "node": planner.id,
                        "path": "items.output.side",
                    }
                ],
            },
            inputs=[stock, planner],
        )
        report = LLMChatOp(
            [OpMessage(role="user", content=retrieval)],
            config=GenerationConfig(model="x"),
        )
        compiled = Graph.from_ops([as_output("r", report)]).compile(Stock=["NVDA"])
        node = _build_profile_node(compiled, retrieval.id)
        sample = node.data_spec["params"][0]["data"]["items"][0]
        assert sample == "BUY"


class TestSampleValueShortCircuit:
    def test_upstream_data_retrieval_sample_value_is_used_directly(self) -> None:
        stock = input_placeholder("Stock")
        upstream = DataRetrievalOp(
            data_spec={
                "type": "sql",
                "connection_string": "postgresql://example",
                "template": "SELECT key FROM lookup WHERE s='{s}'",
                "params": [{"label": "s", "node": stock.id}],
                "sample_value": "K-123",
            },
            inputs=[stock],
        )
        downstream = DataRetrievalOp(
            data_spec={
                "type": "sql",
                "connection_string": "postgresql://example",
                "template": "SELECT * FROM facts WHERE key='{key}'",
                "params": [
                    {
                        "label": "key",
                        "node": upstream.id,
                        "path": "items.table.key",
                    }
                ],
            },
            inputs=[upstream],
        )
        llm = LLMChatOp(
            [OpMessage(role="user", content=downstream)],
            config=GenerationConfig(model="x"),
        )
        compiled = Graph.from_ops([as_output("r", llm)]).compile(Stock=["NVDA"])
        node = _build_profile_node(compiled, downstream.id)
        sample = node.data_spec["params"][0]["data"]["items"][0]
        assert sample == "K-123"

    def test_sample_value_projects_by_path_when_record_shaped(self) -> None:
        stock = input_placeholder("Stock")
        upstream = DataRetrievalOp(
            data_spec={
                "type": "sql",
                "connection_string": "postgresql://example",
                "template": "SELECT key, label FROM lookup WHERE s='{s}'",
                "params": [{"label": "s", "node": stock.id}],
                "sample_value": {"key": "K-7", "label": "alpha"},
            },
            inputs=[stock],
        )
        downstream = DataRetrievalOp(
            data_spec={
                "type": "sql",
                "connection_string": "postgresql://example",
                "template": "SELECT * FROM facts WHERE key='{key}'",
                "params": [
                    {
                        "label": "key",
                        "node": upstream.id,
                        "path": "items.table.key",
                    }
                ],
            },
            inputs=[upstream],
        )
        llm = LLMChatOp(
            [OpMessage(role="user", content=downstream)],
            config=GenerationConfig(model="x"),
        )
        compiled = Graph.from_ops([as_output("r", llm)]).compile(Stock=["NVDA"])
        node = _build_profile_node(compiled, downstream.id)
        sample = node.data_spec["params"][0]["data"]["items"][0]
        assert sample == "K-7"


class TestLiveSqlSampler:
    def test_sql_sampler_invoked_with_limit_when_no_sample_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(envs, "LUMILAKE_DATA_PROFILE_ENABLE_LIVE_SAMPLING", True)

        calls: list[tuple[str, str]] = []

        def fake_sampler(connection_string: str, query: str) -> list[Any]:
            calls.append((connection_string, query))
            return [{"key": "fetched-key"}]

        monkeypatch.setattr(rg._RuntimeProfileSamplers, "sql", fake_sampler)

        stock = input_placeholder("Stock")
        upstream = DataRetrievalOp(
            data_spec={
                "type": "sql",
                "connection_string": "postgresql://upstream",
                "template": "SELECT key FROM lookup WHERE s='{s}'",
                "params": [{"label": "s", "node": stock.id}],
            },
            inputs=[stock],
        )
        downstream = DataRetrievalOp(
            data_spec={
                "type": "sql",
                "connection_string": "postgresql://downstream",
                "template": "SELECT * FROM facts WHERE key='{key}'",
                "params": [
                    {
                        "label": "key",
                        "node": upstream.id,
                        "path": "items.table.key",
                    }
                ],
            },
            inputs=[upstream],
        )
        llm = LLMChatOp(
            [OpMessage(role="user", content=downstream)],
            config=GenerationConfig(model="x"),
        )
        compiled = Graph.from_ops([as_output("r", llm)]).compile(Stock=["NVDA"])
        node = _build_profile_node(compiled, downstream.id)
        sample = node.data_spec["params"][0]["data"]["items"][0]
        assert sample == "fetched-key"
        assert len(calls) == 1
        assert calls[0][0] == "postgresql://upstream"
        assert "LIMIT" in calls[0][1].upper()
        assert "SELECT key FROM lookup WHERE s='NVDA'" in calls[0][1]

    def test_empty_sampler_result_raises_explicit_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(envs, "LUMILAKE_DATA_PROFILE_ENABLE_LIVE_SAMPLING", True)
        monkeypatch.setattr(
            rg._RuntimeProfileSamplers,
            "sql",
            lambda *_a, **_kw: [],
        )

        stock = input_placeholder("Stock")
        upstream = DataRetrievalOp(
            data_spec={
                "type": "sql",
                "connection_string": "postgresql://upstream",
                "template": "SELECT key FROM lookup WHERE s='{s}'",
                "params": [{"label": "s", "node": stock.id}],
            },
            inputs=[stock],
        )
        downstream = DataRetrievalOp(
            data_spec={
                "type": "sql",
                "connection_string": "postgresql://downstream",
                "template": "SELECT * FROM facts WHERE key='{key}'",
                "params": [
                    {
                        "label": "key",
                        "node": upstream.id,
                        "path": "items.table.key",
                    }
                ],
            },
            inputs=[upstream],
        )
        llm = LLMChatOp(
            [OpMessage(role="user", content=downstream)],
            config=GenerationConfig(model="x"),
        )
        compiled = Graph.from_ops([as_output("r", llm)]).compile(Stock=["NVDA"])
        with pytest.raises(ValueError, match="returned 0 rows"):
            _build_profile_node(compiled, downstream.id)


class TestDefaultDenyLiveSampling:
    def _make_upstream_downstream(self) -> tuple[Any, Any, Any]:
        """Return (stock, upstream, downstream) chain with no sample_value."""
        stock = input_placeholder("Stock")
        upstream = DataRetrievalOp(
            data_spec={
                "type": "sql",
                "connection_string": "postgresql://upstream",
                "template": "SELECT key FROM lookup WHERE s='{s}'",
                "params": [{"label": "s", "node": stock.id}],
            },
            inputs=[stock],
        )
        downstream = DataRetrievalOp(
            data_spec={
                "type": "sql",
                "connection_string": "postgresql://downstream",
                "template": "SELECT * FROM facts WHERE key='{key}'",
                "params": [
                    {
                        "label": "key",
                        "node": upstream.id,
                        "path": "items.table.key",
                    }
                ],
            },
            inputs=[upstream],
        )
        return stock, upstream, downstream

    def test_live_sampling_default_denies_without_sample_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(envs, "LUMILAKE_DATA_PROFILE_ENABLE_LIVE_SAMPLING", False)

        stock, upstream, downstream = self._make_upstream_downstream()
        llm = LLMChatOp(
            [OpMessage(role="user", content=downstream)],
            config=GenerationConfig(model="x"),
        )
        compiled = Graph.from_ops([as_output("r", llm)]).compile(Stock=["NVDA"])
        with pytest.raises(ValueError) as excinfo:
            _build_profile_node(compiled, downstream.id)
        message = str(excinfo.value)
        assert "sample_value" in message
        assert "LUMILAKE_DATA_PROFILE_ENABLE_LIVE_SAMPLING" in message

    def test_live_sampling_proceeds_when_env_var_is_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(envs, "LUMILAKE_DATA_PROFILE_ENABLE_LIVE_SAMPLING", True)

        connect_calls: list[str] = []
        mock_cursor = MagicMock()
        mock_cursor.description = [MagicMock(name="key")]
        mock_cursor.fetchall.return_value = [("live-key",)]
        mock_cursor.__enter__ = lambda s: s
        mock_cursor.__exit__ = MagicMock(return_value=False)

        class FakeConn:
            read_only = False

            def cursor(self) -> Any:
                return mock_cursor

            def __enter__(self) -> "FakeConn":
                return self

            def __exit__(self, *_: Any) -> None:
                pass

        def fake_connect(conn_str: str, **_kwargs: Any) -> FakeConn:
            connect_calls.append(conn_str)
            return FakeConn()

        monkeypatch.setattr(
            "lumilake_server.runtime.runtime_graph.psycopg.connect", fake_connect
        )

        stock, upstream, downstream = self._make_upstream_downstream()
        llm = LLMChatOp(
            [OpMessage(role="user", content=downstream)],
            config=GenerationConfig(model="x"),
        )
        compiled = Graph.from_ops([as_output("r", llm)]).compile(Stock=["NVDA"])
        node = _build_profile_node(compiled, downstream.id)
        sample = node.data_spec["params"][0]["data"]["items"][0]
        assert sample == "live-key"
        assert connect_calls == ["postgresql://upstream"]
        # call_args returns the most recent execute call, which is the actual
        # sample query (the SET LOCAL fires first as a separate execute).
        executed_query: str = mock_cursor.execute.call_args[0][0]
        assert "LIMIT" in executed_query.upper()

    def test_sample_value_short_circuit_works_regardless_of_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(envs, "LUMILAKE_DATA_PROFILE_ENABLE_LIVE_SAMPLING", False)

        stock = input_placeholder("Stock")
        upstream = DataRetrievalOp(
            data_spec={
                "type": "sql",
                "connection_string": "postgresql://upstream",
                "template": "SELECT key FROM lookup WHERE s='{s}'",
                "params": [{"label": "s", "node": stock.id}],
                "sample_value": "static-key",
            },
            inputs=[stock],
        )
        downstream = DataRetrievalOp(
            data_spec={
                "type": "sql",
                "connection_string": "postgresql://downstream",
                "template": "SELECT * FROM facts WHERE key='{key}'",
                "params": [
                    {
                        "label": "key",
                        "node": upstream.id,
                        "path": "items.table.key",
                    }
                ],
            },
            inputs=[upstream],
        )
        llm = LLMChatOp(
            [OpMessage(role="user", content=downstream)],
            config=GenerationConfig(model="x"),
        )
        compiled = Graph.from_ops([as_output("r", llm)]).compile(Stock=["NVDA"])
        with patch(
            "lumilake_server.runtime.runtime_graph.psycopg.connect"
        ) as mock_connect:
            node = _build_profile_node(compiled, downstream.id)
            mock_connect.assert_not_called()

        sample = node.data_spec["params"][0]["data"]["items"][0]
        assert sample == "static-key"


class TestRecursiveResolverAndCycleGuard:
    def test_two_hop_recursion_resolves_via_chained_sample_values(self) -> None:
        stock = input_placeholder("Stock")
        hop1 = DataRetrievalOp(
            data_spec={
                "type": "sql",
                "connection_string": "postgresql://hop1",
                "template": "SELECT a FROM t1 WHERE s='{s}'",
                "params": [{"label": "s", "node": stock.id}],
                "sample_value": "hop1-a",
            },
            inputs=[stock],
        )
        hop2 = DataRetrievalOp(
            data_spec={
                "type": "sql",
                "connection_string": "postgresql://hop2",
                "template": "SELECT b FROM t2 WHERE a='{a}'",
                "params": [{"label": "a", "node": hop1.id, "path": "items.table.a"}],
                "sample_value": "hop2-b",
            },
            inputs=[hop1],
        )
        downstream = DataRetrievalOp(
            data_spec={
                "type": "sql",
                "connection_string": "postgresql://downstream",
                "template": "SELECT * FROM t3 WHERE b='{b}'",
                "params": [{"label": "b", "node": hop2.id, "path": "items.table.b"}],
            },
            inputs=[hop2],
        )
        llm = LLMChatOp(
            [OpMessage(role="user", content=downstream)],
            config=GenerationConfig(model="x"),
        )
        compiled = Graph.from_ops([as_output("r", llm)]).compile(Stock=["NVDA"])
        node = _build_profile_node(compiled, downstream.id)
        sample = node.data_spec["params"][0]["data"]["items"][0]
        assert sample == "hop2-b"

    def test_cycle_in_upstream_chain_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the live sampler is invoked recursively against a cyclic
        DAG, the visited-set guard must trip and raise."""
        # Build a fake graph_dict directly and call into the resolver,
        # because constructing a cyclic DAG through the public Op API
        # is intentionally blocked at graph compile time.
        stock = input_placeholder("Stock")
        op_a = DataRetrievalOp(
            data_spec={
                "type": "sql",
                "connection_string": "postgresql://a",
                "template": "SELECT x FROM ta WHERE k='{k}'",
                "params": [{"label": "k", "node": "OP_B", "path": "items.table.x"}],
            },
            inputs=[stock],
        )
        op_b = DataRetrievalOp(
            data_spec={
                "type": "sql",
                "connection_string": "postgresql://b",
                "template": "SELECT x FROM tb WHERE k='{k}'",
                "params": [{"label": "k", "node": op_a.id, "path": "items.table.x"}],
            },
            inputs=[stock],
        )
        graph_dict: dict[str, Any] = {
            stock.id: stock,
            op_a.id: op_a,
            "OP_B": op_b,
        }
        builder = rg.RuntimeGraphBuilder()
        with pytest.raises(ValueError, match="cycle"):
            builder._resolve_profile_param(
                owner_node_id=op_a.id,
                param={
                    "label": "k",
                    "node": "OP_B",
                    "path": "items.table.x",
                },
                graph_dict=graph_dict,
                inputs_dict={stock.name: ["NVDA"]},
                visited={op_a.id, "OP_B"},
            )


class TestUnsupportedUpstreamRejection:
    def test_format_op_upstream_is_rejected_with_actionable_message(
        self,
    ) -> None:
        stock = input_placeholder("Stock")
        fmt = FormatOp("prefix-{s}", s=stock)
        retrieval = DataRetrievalOp(
            data_spec={
                "type": "sql",
                "connection_string": "postgresql://example",
                "template": "SELECT * FROM t WHERE k='{k}'",
                "params": [{"label": "k", "node": fmt.id, "path": "items.output"}],
            },
            inputs=[fmt],
        )
        llm = LLMChatOp(
            [OpMessage(role="user", content=retrieval)],
            config=GenerationConfig(model="x"),
        )
        compiled = Graph.from_ops([as_output("r", llm)]).compile(Stock=["NVDA"])
        with pytest.raises(ValueError) as excinfo:
            _build_profile_node(compiled, retrieval.id)
        message = str(excinfo.value)
        assert "FormatOp" in message
        assert "sample_value" in message
        assert "LUMILAKE_DISABLE_DATA_PROFILE" in message


class TestApplyLimitToSampleSql:
    def test_for_update_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="sample_value"):
            rg.RuntimeGraphBuilder._apply_limit_to_sample_sql(
                "SELECT id FROM t FOR UPDATE", 5
            )

    def test_for_share_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="sample_value"):
            rg.RuntimeGraphBuilder._apply_limit_to_sample_sql(
                "SELECT id FROM t FOR SHARE", 5
            )

    def test_for_update_lowercase_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="sample_value"):
            rg.RuntimeGraphBuilder._apply_limit_to_sample_sql(
                "select id from t for update", 5
            )

    def test_normal_query_gets_outer_wrap(self) -> None:
        result = rg.RuntimeGraphBuilder._apply_limit_to_sample_sql(
            "SELECT id FROM t", 5
        )
        assert result.startswith("SELECT * FROM (")
        assert result.endswith(") AS _lumilake_sample LIMIT 5")

    def test_query_with_existing_limit_is_wrapped(self) -> None:
        result = rg.RuntimeGraphBuilder._apply_limit_to_sample_sql(
            "SELECT id FROM t LIMIT 10", 5
        )
        assert result.startswith("SELECT * FROM (")
        assert result.endswith(") AS _lumilake_sample LIMIT 5")

    def test_limit_inside_string_literal_does_not_bypass_wrap(self) -> None:
        result = rg.RuntimeGraphBuilder._apply_limit_to_sample_sql(
            "SELECT * FROM t WHERE note = 'LIMIT 100'", 5
        )
        assert result.startswith("SELECT * FROM (")
        assert result.endswith(") AS _lumilake_sample LIMIT 5")

    def test_limit_inside_comment_does_not_bypass_wrap(self) -> None:
        result = rg.RuntimeGraphBuilder._apply_limit_to_sample_sql(
            "SELECT * FROM t -- LIMIT 100", 5
        )
        assert result.startswith("SELECT * FROM (")
        assert result.endswith(") AS _lumilake_sample LIMIT 5")

    def test_large_inner_limit_is_capped_by_outer_wrap(self) -> None:
        result = rg.RuntimeGraphBuilder._apply_limit_to_sample_sql(
            "SELECT * FROM t LIMIT 1000000", 5
        )
        assert result.startswith("SELECT * FROM (")
        assert result.endswith(") AS _lumilake_sample LIMIT 5")

    def test_trailing_semicolon_is_stripped_before_wrap(self) -> None:
        result = rg.RuntimeGraphBuilder._apply_limit_to_sample_sql(
            "SELECT * FROM t;", 5
        )
        assert result.startswith("SELECT * FROM (")
        assert result.endswith(") AS _lumilake_sample LIMIT 5")
        assert ";)" not in result


class TestSampleQueryReadonlyValidation:
    def test_sample_query_with_cte_dml_is_rejected_statically(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            rg.RuntimeGraphBuilder._apply_limit_to_sample_sql(
                "WITH x AS (DELETE FROM t RETURNING id) SELECT id FROM x", 5
            )
        msg = str(excinfo.value)
        assert "DELETE" in msg
        assert "sample_value" in msg

    def test_sample_query_with_top_level_dml_is_rejected_statically(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            rg.RuntimeGraphBuilder._apply_limit_to_sample_sql(
                "DELETE FROM t WHERE id < 5", 5
            )
        msg = str(excinfo.value)
        assert "DELETE" in msg
        assert "sample_value" in msg

    def test_sample_query_with_multistatement_is_rejected_statically(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            rg.RuntimeGraphBuilder._apply_limit_to_sample_sql(
                "SELECT 1; DELETE FROM t", 5
            )
        assert "sample_value" in str(excinfo.value)

    def test_sample_query_plain_select_passes_static_check(self) -> None:
        result = rg.RuntimeGraphBuilder._apply_limit_to_sample_sql(
            "SELECT id FROM t WHERE id = 1", 5
        )
        assert "LIMIT" in result.upper()
        assert "SELECT id FROM t WHERE id = 1" in result

    def test_sql_sampler_uses_read_only_transaction(self) -> None:
        mock_cursor = MagicMock()
        mock_cursor.description = [MagicMock(name="id")]
        mock_cursor.fetchall.return_value = [(42,)]
        mock_cursor.__enter__ = lambda s: s
        mock_cursor.__exit__ = MagicMock(return_value=False)

        read_only_set_at: list[bool] = []
        execute_called_after_readonly: list[bool] = []

        class TrackingConnection:
            def __init__(self) -> None:
                self._read_only = False

            @property
            def read_only(self) -> bool:
                return self._read_only

            @read_only.setter
            def read_only(self, value: bool) -> None:
                self._read_only = value
                read_only_set_at.append(value)

            def cursor(self) -> Any:
                execute_called_after_readonly.append(self._read_only)
                return mock_cursor

            def __enter__(self) -> "TrackingConnection":
                return self

            def __exit__(self, *_: Any) -> None:
                pass

        with patch(
            "lumilake_server.runtime.runtime_graph.psycopg.connect"
        ) as mock_connect:
            mock_connect.return_value = TrackingConnection()
            _default_sql_sampler("postgresql://test", "SELECT id FROM t LIMIT 1")

        assert read_only_set_at == [
            True
        ], "read_only must be set to True before cursor use"
        assert execute_called_after_readonly == [
            True
        ], "cursor opened after read_only=True"


class TestStructuralOutputsMissingOrFieldAbsent:
    def test_llm_without_structural_outputs_is_rejected_with_actionable_message(
        self,
    ) -> None:
        stock = input_placeholder("Stock")
        planner = LLMChatOp(
            [OpMessage(role="user", content=stock)],
            config=GenerationConfig(model="x"),
        )
        retrieval = DataRetrievalOp(
            data_spec={
                "type": "sql",
                "connection_string": "postgresql://example",
                "template": "SELECT * FROM t WHERE code='{code}'",
                "params": [
                    {
                        "label": "code",
                        "node": planner.id,
                        "path": "items.output.code",
                    }
                ],
            },
            inputs=[stock, planner],
        )
        report = LLMChatOp(
            [OpMessage(role="user", content=retrieval)],
            config=GenerationConfig(model="x"),
        )
        compiled = Graph.from_ops([as_output("r", report)]).compile(Stock=["NVDA"])
        with pytest.raises(ValueError) as excinfo:
            _build_profile_node(compiled, retrieval.id)
        message = str(excinfo.value)
        assert planner.id in message
        assert "items.output.code" in message
        assert "structural_outputs" in message

    def test_llm_with_structural_outputs_missing_referenced_field_is_rejected(
        self,
    ) -> None:
        stock = input_placeholder("Stock")
        planner = LLMChatOp(
            [OpMessage(role="user", content=stock)],
            config=GenerationConfig(model="x"),
            structural_outputs=[{"name": "foo", "type": "string"}],
        )
        retrieval = DataRetrievalOp(
            data_spec={
                "type": "sql",
                "connection_string": "postgresql://example",
                "template": "SELECT * FROM t WHERE code='{code}'",
                "params": [
                    {
                        "label": "code",
                        "node": planner.id,
                        "path": "items.output.code",
                    }
                ],
            },
            inputs=[stock, planner],
        )
        report = LLMChatOp(
            [OpMessage(role="user", content=retrieval)],
            config=GenerationConfig(model="x"),
        )
        compiled = Graph.from_ops([as_output("r", report)]).compile(Stock=["NVDA"])
        with pytest.raises(ValueError) as excinfo:
            _build_profile_node(compiled, retrieval.id)
        message = str(excinfo.value)
        assert "foo" in message
        assert "code" in message


class TestSqlSamplerTimeouts:
    def _make_tracking_conn(
        self,
    ) -> tuple[Any, list[str]]:
        """Return (FakeConn class, execute_log).

        execute_log records each SQL string passed to cursor.execute in order.
        """
        execute_log: list[str] = []
        mock_cursor = MagicMock()
        mock_cursor.description = [MagicMock(name="id")]
        mock_cursor.fetchall.return_value = [(42,)]
        mock_cursor.__enter__ = lambda s: s
        mock_cursor.__exit__ = MagicMock(return_value=False)

        def recording_execute(sql: str) -> None:
            execute_log.append(sql)

        mock_cursor.execute.side_effect = recording_execute

        class FakeConn:
            read_only = False

            def cursor(self) -> Any:
                return mock_cursor

            def __enter__(self) -> "FakeConn":
                return self

            def __exit__(self, *_: Any) -> None:
                pass

        return FakeConn, execute_log

    def test_sql_sampler_passes_connect_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(envs, "LUMILAKE_DATA_PROFILE_CONNECT_TIMEOUT_S", 5.0)

        connect_kwargs: list[dict[str, Any]] = []

        FakeConn, _ = self._make_tracking_conn()

        def fake_connect(conn_str: str, **kwargs: Any) -> Any:
            connect_kwargs.append(kwargs)
            return FakeConn()

        with patch(
            "lumilake_server.runtime.runtime_graph.psycopg.connect", fake_connect
        ):
            _default_sql_sampler("postgresql://test", "SELECT id FROM t LIMIT 1")

        assert len(connect_kwargs) == 1
        assert connect_kwargs[0].get("connect_timeout") == 5

    def test_sql_sampler_sets_statement_timeout_before_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(envs, "LUMILAKE_DATA_PROFILE_STATEMENT_TIMEOUT_S", 10.0)

        FakeConn, execute_log = self._make_tracking_conn()

        def fake_connect(conn_str: str, **kwargs: Any) -> Any:
            return FakeConn()

        with patch(
            "lumilake_server.runtime.runtime_graph.psycopg.connect", fake_connect
        ):
            _default_sql_sampler("postgresql://test", "SELECT id FROM t LIMIT 1")

        assert len(execute_log) == 2
        assert execute_log[0] == "SET LOCAL statement_timeout = 10000"
        assert "SELECT id FROM t" in execute_log[1]

    def test_sql_sampler_translates_query_cancelled_into_actionable_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(envs, "LUMILAKE_DATA_PROFILE_ENABLE_LIVE_SAMPLING", True)
        monkeypatch.setattr(envs, "LUMILAKE_DATA_PROFILE_STATEMENT_TIMEOUT_S", 10.0)

        execute_log: list[str] = []
        mock_cursor = MagicMock()
        mock_cursor.description = [MagicMock(name="id")]
        mock_cursor.__enter__ = lambda s: s
        mock_cursor.__exit__ = MagicMock(return_value=False)

        def recording_execute(sql: str) -> None:
            execute_log.append(sql)
            if "SELECT" in sql.upper() and "statement_timeout" not in sql:
                raise psycopg.errors.QueryCanceled(
                    "canceling statement due to statement timeout"
                )

        mock_cursor.execute.side_effect = recording_execute

        class FakeConn:
            read_only = False

            def cursor(self) -> Any:
                return mock_cursor

            def __enter__(self) -> "FakeConn":
                return self

            def __exit__(self, *_: Any) -> None:
                pass

        def fake_connect(conn_str: str, **kwargs: Any) -> Any:
            return FakeConn()

        stock = input_placeholder("Stock")
        upstream = DataRetrievalOp(
            data_spec={
                "type": "sql",
                "connection_string": "postgresql://upstream",
                "template": "SELECT key FROM lookup WHERE s='{s}'",
                "params": [{"label": "s", "node": stock.id}],
            },
            inputs=[stock],
        )
        downstream = DataRetrievalOp(
            data_spec={
                "type": "sql",
                "connection_string": "postgresql://downstream",
                "template": "SELECT * FROM facts WHERE key='{key}'",
                "params": [
                    {
                        "label": "key",
                        "node": upstream.id,
                        "path": "items.table.key",
                    }
                ],
            },
            inputs=[upstream],
        )
        llm = LLMChatOp(
            [OpMessage(role="user", content=downstream)],
            config=GenerationConfig(model="x"),
        )
        compiled = Graph.from_ops([as_output("r", llm)]).compile(Stock=["NVDA"])

        with patch(
            "lumilake_server.runtime.runtime_graph.psycopg.connect", fake_connect
        ):
            with pytest.raises(ValueError) as excinfo:
                _build_profile_node(compiled, downstream.id)

        message = str(excinfo.value)
        assert upstream.id in message
        assert "statement_timeout" in message.lower() or "STATEMENT_TIMEOUT" in message
        assert "LUMILAKE_DATA_PROFILE_STATEMENT_TIMEOUT_S" in message
        assert "sample_value" in message


class TestCollectorTypeDefaultFallback:
    def test_string_constraint_without_variants_yields_empty_string(self) -> None:
        result = _collect_data_profile_param_candidates(
            params=[],
            constraints=[{"name": "x", "type": "string"}],
        )
        assert result == {"x": [""]}

    def test_int_constraint_defaults_to_zero(self) -> None:
        result = _collect_data_profile_param_candidates(
            params=[],
            constraints=[{"name": "n", "type": "int"}],
        )
        assert result == {"n": [0]}

    def test_datetime_constraint_defaults_to_epoch_string(self) -> None:
        result = _collect_data_profile_param_candidates(
            params=[],
            constraints=[{"name": "d", "type": "datetime"}],
        )
        assert result == {"d": ["1970-01-01"]}


class TestTimeoutEnvValidation:
    def test_envs_rejects_zero_connect_timeout(self) -> None:
        result = _positive_float("LUMILAKE_DATA_PROFILE_CONNECT_TIMEOUT_S", "0", 5.0)
        assert result == 5.0

    def test_envs_rejects_negative_statement_timeout(self) -> None:
        result = _positive_float(
            "LUMILAKE_DATA_PROFILE_STATEMENT_TIMEOUT_S", "-3.5", 10.0
        )
        assert result == 10.0

    def test_envs_rejects_non_finite_values(self) -> None:
        for raw in ("nan", "inf", "-inf"):
            result = _positive_float(
                "LUMILAKE_DATA_PROFILE_CONNECT_TIMEOUT_S", raw, 5.0
            )
            assert result == 5.0, f"{raw!r} should fall back to default"

    def test_sub_second_connect_timeout_rounds_up_to_int(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(envs, "LUMILAKE_DATA_PROFILE_CONNECT_TIMEOUT_S", 0.5)

        connect_kwargs: list[dict[str, Any]] = []
        mock_cursor = MagicMock()
        mock_cursor.description = [MagicMock(name="id")]
        mock_cursor.fetchall.return_value = [(1,)]
        mock_cursor.__enter__ = lambda s: s
        mock_cursor.__exit__ = MagicMock(return_value=False)

        class FakeConn:
            read_only = False

            def cursor(self) -> Any:
                return mock_cursor

            def __enter__(self) -> "FakeConn":
                return self

            def __exit__(self, *_: Any) -> None:
                pass

        def fake_connect(conn_str: str, **kwargs: Any) -> Any:
            connect_kwargs.append(kwargs)
            return FakeConn()

        with patch(
            "lumilake_server.runtime.runtime_graph.psycopg.connect", fake_connect
        ):
            _default_sql_sampler("postgresql://test", "SELECT id FROM t LIMIT 1")

        assert len(connect_kwargs) == 1
        assert connect_kwargs[0].get("connect_timeout") == 1

    def test_sub_second_statement_timeout_rounds_to_ms(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(envs, "LUMILAKE_DATA_PROFILE_STATEMENT_TIMEOUT_S", 0.5)

        execute_log: list[str] = []
        mock_cursor = MagicMock()
        mock_cursor.description = [MagicMock(name="id")]
        mock_cursor.fetchall.return_value = [(1,)]
        mock_cursor.__enter__ = lambda s: s
        mock_cursor.__exit__ = MagicMock(return_value=False)

        def recording_execute(sql: str) -> None:
            execute_log.append(sql)

        mock_cursor.execute.side_effect = recording_execute

        class FakeConn:
            read_only = False

            def cursor(self) -> Any:
                return mock_cursor

            def __enter__(self) -> "FakeConn":
                return self

            def __exit__(self, *_: Any) -> None:
                pass

        def fake_connect(conn_str: str, **kwargs: Any) -> Any:
            return FakeConn()

        with patch(
            "lumilake_server.runtime.runtime_graph.psycopg.connect", fake_connect
        ):
            _default_sql_sampler("postgresql://test", "SELECT id FROM t LIMIT 1")

        assert len(execute_log) == 2
        assert execute_log[0] == "SET LOCAL statement_timeout = 500"
