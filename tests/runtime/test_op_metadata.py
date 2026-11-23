"""The metadata an op declares matches the nodes its lowering emits."""

import pytest

from lumilake import envs
from lumilake_server.common import GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
    DataRetrievalOp,
    LambdaOp,
    LLMChatOp,
    OpMessage,
    RowShape,
    as_output,
    input_placeholder,
)
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder
from tests.support.data_specs import column_table, wire

_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")


def _build(op, task_type_override=None):
    graph = Graph.from_ops([as_output("result", op)])
    inputs = {"Stock": ["NVDA", "AAPL"]} if graph.input_ops else {}
    return RuntimeGraphBuilder().build(
        graph.compile(**inputs), task_type_override=task_type_override
    )


def _whole_list(items):
    return list(items)


@pytest.mark.parametrize(
    ("rowwise_template", "shape", "flatten"),
    [(None, RowShape.PER_ROW, False), ("{symbol}", RowShape.GROUPED, True)],
)
def test_chat_shape_decides_how_its_consumer_reads_it(
    rowwise_template: str | None, shape: RowShape, flatten: bool
) -> None:
    """A row template groups a table into one dataframe item, so a consumer
    flattens it; a plain chat emits one value per row, so it does not."""
    chat = LLMChatOp(
        [OpMessage(role="user", content=input_placeholder("Stock"))],
        config=GenerationConfig(model=_MODEL),
        rowwise_template=rowwise_template,
        rowwise_columns=(
            column_table(
                (
                    [{"label": "symbol", "data": {"type": "list", "items": ["NVDA"]}}]
                    if rowwise_template
                    else None
                ),
                "rowwise_columns",
            )
        ),
    )
    consumer = LambdaOp([chat], _whole_list, mode="list")

    assert chat.row_shape is shape
    stage = wire(_build(consumer).nodes[consumer.id])["plan"][0]
    assert stage.get("flatten", False) is flatten


@pytest.mark.parametrize(
    ("data_spec", "profilable"),
    [
        ({"type": "lumid", "mode": "sql", "template": "select a from t"}, True),
        ({"type": "lumid", "mode": "s3", "template": "s3://b/{k}"}, True),
        ({"type": "lumid", "mode": "agent", "description": "Find a price."}, False),
    ],
)
def test_profilable_matches_whether_profile_mode_profiles_the_op(
    data_spec: dict, profilable: bool
) -> None:
    op = DataRetrievalOp(data_spec=data_spec)

    assert op.profilable is profilable
    profiled = [
        node
        for node in _build(op, task_type_override="data_profile").nodes.values()
        if node.task_type == "data_profiling"
    ]
    assert bool(profiled) is profilable
