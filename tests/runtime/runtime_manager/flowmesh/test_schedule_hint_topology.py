from lumilake.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager


def test_build_flat_schedule_hint_from_worker_assignment() -> None:
    order, selected = FlowmeshRuntimeManager._build_flat_schedule_hint(
        worker_assignment={
            "gpu-0": ["n1"],
            "gpu-1": ["n2", "n1"],
        },
        node_names=["n1", "n2"],
        node_dependencies={
            "n1": [],
            "n2": ["n1"],
        },
    )
    assert order == ["n1", "n2"]
    assert selected == {
        "n1": ["gpu-0", "gpu-1"],
        "n2": ["gpu-1"],
    }


def test_build_flat_schedule_hint_is_topological() -> None:
    order, selected = FlowmeshRuntimeManager._build_flat_schedule_hint(
        worker_assignment={
            "gpu-0": ["n2", "n1"],
        },
        node_names=["n1", "n2"],
        node_dependencies={
            "n1": [],
            "n2": ["n1"],
        },
    )
    assert order == ["n1", "n2"]
    assert selected == {
        "n1": ["gpu-0"],
        "n2": ["gpu-0"],
    }


def test_build_index_partitions_supports_non_dividable_worker_count() -> None:
    assert FlowmeshRuntimeManager._build_index_partitions(4, 3) == [
        (0, 2),
        (2, 3),
        (3, 4),
    ]
