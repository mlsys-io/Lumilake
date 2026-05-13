import pytest

from tests.runtime.runtime_manager.flowmesh.flowmesh_mocker import (
    FlowmeshJobMocker,
    FlowmeshMockShapeError,
)


def _task_spec(nodes: list[dict[str, object]]) -> dict[str, object]:
    return {"spec": {"graph": {"nodes": nodes}}}


def test_flowmesh_mocker_executes_in_topological_order_with_reversed_nodes() -> None:
    spec = _task_spec(
        [
            {
                "name": "n2",
                "dependsOn": ["n0", "n1"],
                "spec": {
                    "data": {
                        "type": "graph_template",
                        "template": {
                            "name": "format",
                            "columns": [
                                {
                                    "label": "a",
                                    "node": "n0",
                                    "path": "items.output",
                                },
                                {
                                    "label": "b",
                                    "node": "n1",
                                    "path": "items.output",
                                },
                            ],
                            "options": {
                                "format": {
                                    "steps": [
                                        {
                                            "label": "merged",
                                            "template": "{left}-{right}",
                                            "arguments": [
                                                {"label": "left", "value": "a"},
                                                {"label": "right", "value": "b"},
                                            ],
                                        }
                                    ],
                                    "messages": [{"role": "user", "content": "merged"}],
                                }
                            },
                        },
                    }
                },
            },
            {
                "name": "n1",
                "dependsOn": [],
                "spec": {"data": {"type": "list", "items": ["u", "v"]}},
            },
            {
                "name": "n0",
                "dependsOn": [],
                "spec": {"data": {"type": "list", "items": ["x", "y"]}},
            },
        ]
    )
    context = FlowmeshJobMocker().run(spec)
    assert "n2" in context
    assert len(context["n2"]["items"]) == 2


def test_flowmesh_mocker_fails_on_mismatched_list_template_arguments() -> None:
    spec = _task_spec(
        [
            {
                "name": "n0",
                "dependsOn": [],
                "spec": {"data": {"type": "list", "items": ["a", "b"]}},
            },
            {
                "name": "n1",
                "dependsOn": [],
                "spec": {"data": {"type": "list", "items": ["x", "y", "z"]}},
            },
            {
                "name": "n2",
                "dependsOn": ["n0", "n1"],
                "spec": {
                    "data": {
                        "type": "graph_template",
                        "template": {
                            "name": "format",
                            "columns": [
                                {
                                    "label": "a",
                                    "node": "n0",
                                    "path": "items.output",
                                },
                                {
                                    "label": "b",
                                    "node": "n1",
                                    "path": "items.output",
                                },
                            ],
                            "options": {
                                "format": {
                                    "steps": [
                                        {
                                            "label": "merged",
                                            "template": "{left}-{right}",
                                            "arguments": [
                                                {"label": "left", "value": "a"},
                                                {"label": "right", "value": "b"},
                                            ],
                                        }
                                    ],
                                    "messages": [{"role": "user", "content": "merged"}],
                                }
                            },
                        },
                    }
                },
            },
        ]
    )
    with pytest.raises(
        FlowmeshMockShapeError,
        match="All list-type format arguments must have the same length",
    ):
        FlowmeshJobMocker().run(spec)


def test_flowmesh_mocker_accepts_inlined_indexed_dependency_slices() -> None:
    spec = _task_spec(
        [
            {
                "name": "n0",
                "dependsOn": [],
                "spec": {"data": {"type": "list", "items": ["a", "b", "c", "d"]}},
            },
            {
                "name": "n1",
                "dependsOn": ["n0"],
                "spec": {
                    "data": {
                        "type": "graph_template",
                        "template": {
                            "name": "format",
                            "columns": [
                                {
                                    "label": "x",
                                    "data": {
                                        "type": "list",
                                        "items": [
                                            {
                                                "node": "n0",
                                                "path": "items[0].output",
                                            },
                                            {
                                                "node": "n0",
                                                "path": "items[1].output",
                                            },
                                        ],
                                    },
                                },
                                {
                                    "label": "seed",
                                    "data": {"type": "list", "items": ["p", "q"]},
                                },
                            ],
                            "options": {
                                "format": {
                                    "steps": [
                                        {
                                            "label": "merged",
                                            "template": "{left}:{right}",
                                            "arguments": [
                                                {"label": "left", "value": "x"},
                                                {"label": "right", "value": "seed"},
                                            ],
                                        }
                                    ],
                                    "messages": [{"role": "user", "content": "merged"}],
                                }
                            },
                        },
                    }
                },
            },
        ]
    )
    context = FlowmeshJobMocker().run(spec)
    assert len(context["n1"]["items"]) == 2


def test_flowmesh_mocker_supports_dataframe_column_type() -> None:
    spec = _task_spec(
        [
            {
                "name": "n0",
                "dependsOn": [],
                "spec": {
                    "data": {
                        "type": "list",
                        "items": [
                            {"table": {"symbol": "NVDA", "price": 120.0}},
                            {"table": {"symbol": "MSFT", "price": 300.0}},
                        ],
                    }
                },
            },
            {
                "name": "n1",
                "dependsOn": ["n0"],
                "spec": {
                    "data": {
                        "type": "graph_template",
                        "template": {
                            "name": "format",
                            "columns": [
                                {
                                    "label": "df",
                                    "data": {
                                        "type": "dataframe",
                                        "columns": [
                                            {
                                                "label": "symbol",
                                                "node": "n0",
                                                "path": "items.table.symbol",
                                            },
                                            {
                                                "label": "price",
                                                "node": "n0",
                                                "path": "items.table.price",
                                            },
                                        ],
                                    },
                                }
                            ],
                            "options": {
                                "format": {
                                    "steps": [
                                        {
                                            "label": "prompt",
                                            "template": "{content}",
                                            "arguments": [
                                                {"label": "content", "value": "df"}
                                            ],
                                        }
                                    ],
                                    "messages": [{"role": "user", "content": "prompt"}],
                                }
                            },
                        },
                    }
                },
            },
        ]
    )
    context = FlowmeshJobMocker().run(spec)
    assert len(context["n1"]["items"]) == 1


def test_flowmesh_mocker_supports_function_steps() -> None:
    spec = _task_spec(
        [
            {
                "name": "n0",
                "dependsOn": [],
                "spec": {"data": {"type": "list", "items": ["x", "y"]}},
            },
            {
                "name": "n1",
                "dependsOn": ["n0"],
                "spec": {
                    "data": {
                        "type": "graph_template",
                        "template": {
                            "name": "format",
                            "columns": [
                                {
                                    "label": "base",
                                    "node": "n0",
                                    "path": "items.output",
                                }
                            ],
                            "options": {
                                "format": {
                                    "steps": [
                                        {
                                            "label": "fn_out",
                                            "function": "def f(args): return args",
                                            "arguments": ["base", "const"],
                                        }
                                    ],
                                    "messages": [{"role": "user", "content": "fn_out"}],
                                }
                            },
                        },
                    }
                },
            },
        ]
    )
    context = FlowmeshJobMocker().run(spec)
    assert len(context["n1"]["items"]) == 2


def test_flowmesh_mocker_supports_sql_table_column_paths() -> None:
    spec = _task_spec(
        [
            {
                "name": "q0",
                "dependsOn": [],
                "spec": {"data": {"type": "sql"}},
            },
            {
                "name": "n1",
                "dependsOn": ["q0"],
                "spec": {
                    "data": {
                        "type": "graph_template",
                        "template": {
                            "name": "format",
                            "columns": [
                                {
                                    "label": "sym",
                                    "node": "q0",
                                    "path": "items.table.symbol",
                                }
                            ],
                            "options": {
                                "format": {
                                    "messages": [{"role": "user", "content": "{sym}"}]
                                }
                            },
                        },
                    }
                },
            },
        ]
    )
    context = FlowmeshJobMocker().run(spec)
    assert len(context["n1"]["items"]) >= 1


def test_flowmesh_mocker_fails_on_missing_expression_path() -> None:
    spec = _task_spec(
        [
            {
                "name": "n0",
                "dependsOn": [],
                "spec": {"data": {"type": "list", "items": ["x", "y"]}},
            },
            {
                "name": "n1",
                "dependsOn": ["n0"],
                "spec": {
                    "data": {
                        "type": "graph_template",
                        "template": {
                            "name": "format",
                            "columns": [
                                {
                                    "label": "bad",
                                    "node": "n0",
                                    "path": "items.table.symbol",
                                }
                            ],
                            "options": {
                                "format": {
                                    "messages": [{"role": "user", "content": "{bad}"}]
                                }
                            },
                        },
                    }
                },
            },
        ]
    )
    with pytest.raises(FlowmeshMockShapeError, match="Expression attribute"):
        FlowmeshJobMocker().run(spec)
