"""Regression tests for the S3 data-profile cost model handling retrieval
nodes whose template placeholders are driven by a non-SQL source
(``InputOp``, other non-runtime DSL ops).

When a placeholder's upstream is a non-SQL op there's no
``estimated_rows`` to use as the file-count projection, so the fallback
here treats input-driven placeholders as ``file_count = 1`` — the
minimum the op is guaranteed to fetch per invocation.
"""

from lumilake_server.runtime.data_profile_utils import _derive_s3_profile_for_graph
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp


def _s3_node(
    node_id: str,
    *,
    template: str,
    params: list[dict[str, str]],
) -> RuntimeOp:
    return RuntimeOp(
        node_id=node_id,
        task_type="data_retrieval",
        backend="data_retrieval",
        model="data_retrieval",
        data_spec={
            "type": "s3",
            "template": template,
            "params": params,
        },
        model_spec={},
        inference_spec={},
    )


def _sql_node(node_id: str) -> RuntimeOp:
    return RuntimeOp(
        node_id=node_id,
        task_type="data_retrieval",
        backend="data_retrieval",
        model="data_retrieval",
        data_spec={
            "type": "sql",
            "template": "SELECT 1",
            "params": [],
        },
        model_spec={},
        inference_spec={},
    )


def _snapshot_with_folder(
    folder: str, *, file_size: int = 200
) -> tuple[dict[str, int | None], list[str]]:
    """Return (listing_sizes, listing_folders) for a single folder.

    One sampled file so ``_average_size_for_folders`` has a byte sample.
    """
    path = f"{folder}sample.json"
    return ({path: file_size}, [folder])


class TestInputDrivenProfile:
    def test_input_driven_placeholder_produces_single_file_count(self) -> None:
        """An InputOp-referenced placeholder never materializes a runtime
        node. The profile must tolerate that and assume one file per
        invocation rather than crashing with ``KeyError``."""
        # ``node_refs[field]`` points at an op id that isn't in
        # ``graph.nodes`` — an upstream ``InputOp`` consumed only as a
        # retrieval template parameter.
        input_op_id = "InputOp-external-query"
        s3_node = _s3_node(
            "retrieve_by_tweet_id",
            template="eventx/impressions/by_tweet_id/{key}.json",
            params=[
                {"label": "key", "node": input_op_id, "path": "data"},
            ],
        )
        graph = RuntimeGraph(
            nodes={s3_node.node_id: s3_node},
            node_order=[s3_node.node_id],
            output_node_map={},
            dsl_to_runtime={},
        )
        listing_sizes, listing_folders = _snapshot_with_folder(
            "eventx/impressions/by_tweet_id/"
        )
        derived = _derive_s3_profile_for_graph(
            graph=graph,
            graph_key="gk",
            org_id="org",
            projected_sql={},
            listing_sizes=listing_sizes,
            listing_folders=listing_folders,
        )
        key = next(iter(derived))
        (estimate,) = derived[key]
        # Exactly one row emitted, with file_count==1 (input-driven
        # fallback).
        assert estimate["cost_estimates"][0]["estimated_files"] == 1

    def test_sql_driven_placeholder_still_uses_sql_estimated_rows(self) -> None:
        """Regression guard: SQL placeholders still honor the SQL
        projection path (previous behavior must not be broken)."""
        from lumilake_server.data_profile_models import (
            DataProfileCostEstimate,
            DataProfileResultRow,
        )

        sql = _sql_node("sql_source")
        s3 = _s3_node(
            "retrieve_by_sql_key",
            template="eventx/impressions/by_tweet_id/{key}.json",
            params=[{"label": "key", "node": sql.node_id, "path": "data"}],
        )
        graph = RuntimeGraph(
            nodes={sql.node_id: sql, s3.node_id: s3},
            node_order=[sql.node_id, s3.node_id],
            output_node_map={},
            dsl_to_runtime={},
        )
        listing_sizes, listing_folders = _snapshot_with_folder(
            "eventx/impressions/by_tweet_id/"
        )
        projected_sql = {
            f"data_profile::{sql.node_id}::{sql.node_id}_query": [
                DataProfileResultRow(
                    node_id=sql.node_id,
                    raw_node_id=sql.node_id,
                    query_name=f"{sql.node_id}_query",
                    table="public.t",
                    cost_estimates=[
                        DataProfileCostEstimate(
                            plan_id="pg_estimate",
                            description="pg projection",
                            raw_cost=1.0,
                            estimated_files=1,
                            total_size_bytes=100,
                            avg_file_size_bytes=100,
                            estimated_rows=37,
                            footprints={},
                        )
                    ],
                )
            ],
        }
        derived = _derive_s3_profile_for_graph(
            graph=graph,
            graph_key="gk",
            org_id="org",
            projected_sql=projected_sql,
            listing_sizes=listing_sizes,
            listing_folders=listing_folders,
        )
        s3_key = next(k for k in derived if s3.node_id in k)
        (s3_estimate,) = derived[s3_key]
        # SQL path: file_count == SQL estimated_rows (37).
        assert s3_estimate["cost_estimates"][0]["estimated_files"] == 37

    def test_non_sql_runtime_ref_is_rejected(self) -> None:
        """A placeholder that points at a concrete non-SQL runtime node
        is a modeling error — file count is undefined in that case. Reject
        explicitly rather than silently fall back to ``file_count = 1`` as
        if it were an input-driven DSL op."""
        import pytest

        # Upstream S3 retrieval node — valid runtime node, but its output
        # isn't SQL estimated_rows, so it can't source a file-count
        # placeholder. Its own template is fully static (no placeholders)
        # so it passes earlier profile validation; we're specifically
        # exercising the downstream's placeholder-source check.
        upstream_s3 = _s3_node(
            "upstream_retrieval",
            template="eventx/other/static.json",
            params=[],
        )
        downstream = _s3_node(
            "downstream_retrieval",
            template="eventx/impressions/by_tweet_id/{key}.json",
            params=[{"label": "key", "node": upstream_s3.node_id, "path": "data"}],
        )
        graph = RuntimeGraph(
            nodes={upstream_s3.node_id: upstream_s3, downstream.node_id: downstream},
            node_order=[upstream_s3.node_id, downstream.node_id],
            output_node_map={},
            dsl_to_runtime={},
        )
        upstream_snap = _snapshot_with_folder("eventx/other/")
        downstream_snap = _snapshot_with_folder("eventx/impressions/by_tweet_id/")
        listing_sizes = {**upstream_snap[0], **downstream_snap[0]}
        listing_folders = [*upstream_snap[1], *downstream_snap[1]]
        with pytest.raises(RuntimeError, match="points at a non-SQL runtime node"):
            _derive_s3_profile_for_graph(
                graph=graph,
                graph_key="gk",
                org_id="org",
                projected_sql={},
                listing_sizes=listing_sizes,
                listing_folders=listing_folders,
            )

    def test_mixed_sql_and_input_refs_is_rejected(self) -> None:
        """Mixing SQL-driven and input-driven placeholders on one node
        leaves the file count ambiguous — reject explicitly."""
        import pytest

        sql = _sql_node("sql_source")
        input_op_id = "InputOp-external"
        s3 = _s3_node(
            "mixed_retrieval",
            # Both placeholders in the filename portion — static folder
            # prefix, so we exercise the ref-source-mixing check rather
            # than the folder-must-be-static check above it.
            template="eventx/impressions/by_tweet_id/{key}-{shard}.json",
            params=[
                {"label": "key", "node": sql.node_id, "path": "data"},
                {"label": "shard", "node": input_op_id, "path": "data"},
            ],
        )
        graph = RuntimeGraph(
            nodes={sql.node_id: sql, s3.node_id: s3},
            node_order=[sql.node_id, s3.node_id],
            output_node_map={},
            dsl_to_runtime={},
        )
        listing_sizes, listing_folders = _snapshot_with_folder(
            "eventx/impressions/by_tweet_id/"
        )
        with pytest.raises(
            RuntimeError, match="cannot mix SQL-driven and input-driven"
        ):
            _derive_s3_profile_for_graph(
                graph=graph,
                graph_key="gk",
                org_id="org",
                projected_sql={},
                listing_sizes=listing_sizes,
                listing_folders=listing_folders,
            )
