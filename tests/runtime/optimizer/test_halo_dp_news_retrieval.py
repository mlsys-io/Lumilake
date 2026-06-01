"""Regression for the HALO DP scheduler.

The ``news-retrieval-text`` family of n8n workflows produces runtime graphs
where one or more ``data_retrieval`` CPU nodes have *no* downstream GPU
consumer (for example, an SQL/S3 result that is written back to storage but
not re-fed into an LLM). The DP path's CPU auto-fill (``_auto_cpu_batch``)
only considered CPU **ancestors** of the chosen GPU subset, so such CPU
leaves were never scheduled and the solver raised
``"DP failed to find a valid schedule; check DAG dependencies."`` even
though the DAG itself is well-formed.

This fixture mirrors the parser → runtime_graph → HaloOptimizer pipeline
for that workflow class: ``backend=vllm`` for inference, ``backend=data_retrieval``
with ``data_spec.type in {'sql','s3'}`` for retrieval, chained dependencies,
and a mixed GPU/CPU worker pool.
"""

from lumilake_server.runtime.optimizer.halo import HaloOptimizer
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp


def _news_retrieval_runtime_graph() -> RuntimeGraph:
    nodes = {
        "sql_param_proposer": RuntimeOp(
            node_id="sql_param_proposer",
            task_type="inference",
            backend="vllm",
            model="Qwen/Qwen2.5-1.5B-Instruct",
            data_spec={
                "type": "graph_template",
                "template": {"name": "format"},
            },
            model_spec={},
            inference_spec={"max_tokens": 256},
        ),
        "news_sql": RuntimeOp(
            node_id="news_sql",
            task_type="data_retrieval",
            backend="data_retrieval",
            model="data_retrieval",
            data_spec={
                "type": "sql",
                "template": "SELECT id, title FROM demo.fact_news_metadata",
            },
            model_spec={},
            inference_spec={},
            dependencies=("sql_param_proposer",),
        ),
        "news_s3": RuntimeOp(
            node_id="news_s3",
            task_type="data_retrieval",
            backend="data_retrieval",
            model="data_retrieval",
            data_spec={
                "type": "s3",
                "template": "unstructured/news-html/{id}.html",
            },
            model_spec={},
            inference_spec={},
            dependencies=("news_sql",),
        ),
        "news_summary": RuntimeOp(
            node_id="news_summary",
            task_type="inference",
            backend="vllm",
            model="meta-llama/Llama-3.1-8B-Instruct",
            data_spec={"type": "dataframe", "template": {"name": "summary"}},
            model_spec={},
            inference_spec={"max_tokens": 512},
            dependencies=("news_sql", "news_s3"),
        ),
        "news_summary_archive": RuntimeOp(
            node_id="news_summary_archive",
            task_type="data_retrieval",
            backend="data_retrieval",
            model="data_retrieval",
            data_spec={
                "type": "s3",
                "template": "archive/summaries/{id}.json",
            },
            model_spec={},
            inference_spec={},
            dependencies=("news_summary",),
        ),
        "analyze_news": RuntimeOp(
            node_id="analyze_news",
            task_type="inference",
            backend="vllm",
            model="meta-llama/Llama-3.1-8B-Instruct",
            data_spec={"type": "graph_template", "template": {"name": "aggregate"}},
            model_spec={},
            inference_spec={"max_tokens": 512},
            dependencies=("news_sql", "news_summary"),
        ),
    }
    return RuntimeGraph(
        nodes=nodes,
        node_order=list(nodes),
        output_node_map={},
        dsl_to_runtime={},
    )


def test_halo_dp_schedules_cpu_leaf_after_gpu_node() -> None:
    optimizer = HaloOptimizer()
    graph = _news_retrieval_runtime_graph()

    schedule = optimizer.generate_schedule(
        graph=graph,
        worker_names=["gpu-0", "cpu-0"],
        worker_profiles={
            "gpu-0": {"has_gpu": True},
            "cpu-0": {"has_gpu": False},
        },
    )

    scheduled = {
        node_id
        for worker_nodes in schedule.worker_assignment.values()
        for node_id in worker_nodes
    }
    assert scheduled == set(graph.nodes), (
        "HALO DP must schedule every node, including CPU leaves with no GPU"
        f" descendant; got {sorted(scheduled)}"
    )
