from collections.abc import Callable
from typing import Any


def topological_sort[T](
    graph: dict[T, set[T]], secondary_key: Callable[[T], Any] | None = None
) -> list[T]:
    in_degree = {node: 0 for node in graph}
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1

    ready_nodes = [node for node, deg in in_degree.items() if deg == 0]
    result = []
    while ready_nodes:
        if secondary_key is not None:
            ready_nodes.sort(key=secondary_key)
        current_layer = ready_nodes
        ready_nodes = []

        result.extend(current_layer)

        # Decrement in-degree of neighbors and collect next layer
        for node in current_layer:
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    ready_nodes.append(neighbor)

    if len(result) != len(in_degree):
        raise ValueError("Graph has at least one cycle.")

    return result
