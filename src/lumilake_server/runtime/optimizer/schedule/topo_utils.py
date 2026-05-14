import heapq
from collections.abc import Mapping, Sequence


def topological_order(
    dependencies: Mapping[str, Sequence[str]],
    nodes: Sequence[str] | None = None,
) -> list[str]:
    """Deterministic Kahn topo sort over the provided dependency mapping."""
    node_set = set(nodes) if nodes is not None else set(dependencies)
    for node_id, parents in dependencies.items():
        if nodes is None or node_id in node_set:
            for parent in parents:
                if nodes is None or parent in node_set:
                    node_set.add(parent)

    indegree: dict[str, int] = {node_id: 0 for node_id in node_set}
    children: dict[str, list[str]] = {node_id: [] for node_id in node_set}
    for node_id in node_set:
        for parent in dependencies.get(node_id, ()):
            if parent not in node_set:
                continue
            indegree[node_id] += 1
            children.setdefault(parent, []).append(node_id)

    ready = [node_id for node_id, deg in indegree.items() if deg == 0]
    heapq.heapify(ready)

    order: list[str] = []
    while ready:
        node_id = heapq.heappop(ready)
        order.append(node_id)
        for child in children.get(node_id, ()):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)

    if len(order) != len(node_set):
        missing = node_set.difference(order)
        raise ValueError(
            "Cycle detected or missing nodes when building topo order:"
            f" {sorted(missing)}"
        )
    return order
