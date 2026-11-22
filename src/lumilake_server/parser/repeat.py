"""The ``Repeat`` YAML construct: a fixed-count repeat of an op body.

``Repeat`` is a YAML construct, not a runtime op: :func:`unroll_repeats`
rewrites each repeat entry into ordinary op entries before any id is
registered, so the graph, the optimizer and the record store see one declared
op per iteration. Iteration ``i`` copies each body entry to ``R.i.<bodyId>``;
``$in`` reads ``input`` on the first iteration and ``R.(i-1).<output>``
afterwards, and another body id resolves within the same iteration. Elsewhere
``R`` refers to ``R.<times>.<output>``.
"""

from collections.abc import Callable, Iterator, Mapping
from typing import Any, NamedTuple

from lumilake_server.ops.router_ops import ROUTER_OP

REPEAT_OP = "Repeat"
INPUT_REF = "$in"
_FIELDS = frozenset({"id", "op", "times", "input", "output", "body"})


class _Repeat(NamedTuple):
    id: str
    times: int
    input_ref: str
    output_ref: str
    body: list[dict[str, Any]]

    @property
    def body_ids(self) -> list[str]:
        return [entry["id"] for entry in self.body]

    def at(self, index: int, body_id: str) -> str:
        return f"{self.id}.{index}.{body_id}"

    def names(self, index: int) -> dict[str, str]:
        """The names iteration ``index`` writes, ``$in`` included."""
        read = self.input_ref if index == 1 else self.at(index - 1, self.output_ref)
        return {INPUT_REF: read, **{bid: self.at(index, bid) for bid in self.body_ids}}


class _Scoped(Mapping[str, str]):
    """A parser id map with one op's scoped names in front of it.

    Every parser and emitter lookup goes through this, so a name that is out
    of place at the referring op is refused by the lookup itself.
    """

    def __init__(
        self,
        ids: Mapping[str, str],
        where: str,
        body_owner: dict[str, str],
        names: dict[str, str],
    ) -> None:
        self._ids = ids
        self._where = where
        self._body_owner = body_owner
        self._names = names

    def target(self, ref: Any) -> Any:
        """The id ``ref`` names in this op's scope, before id derivation."""
        if not isinstance(ref, str) or ref in self._names:
            return self._names.get(ref, ref)
        if ref == INPUT_REF:
            raise ValueError(f"{self._where} may not read '{INPUT_REF}'")
        owner = self._body_owner.get(ref.partition(".")[0])
        if owner is not None:
            raise ValueError(f"{self._where} reads '{ref}', a body id of '{owner}'")
        return ref

    def __getitem__(self, ref: str) -> str:
        return self._ids[self.target(ref)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._ids)

    def __len__(self) -> int:
        return len(self._ids)


References = Callable[[Mapping[str, str], str, str], _Scoped]


def unroll_repeats(ops: list[Any], known_ids: set[str]) -> tuple[list[Any], References]:
    """Replace every ``Repeat`` entry with its unrolled iterations.

    ``known_ids`` holds the workflow input names, so a repeat's ``input``
    reference is checked at submit time. The returned callable binds a parser
    id map to one referring op, so its lookups resolve that op's names first.
    """
    plain = {
        _entry_id(entry)
        for entry in ops
        if isinstance(entry, dict) and entry.get("op") != REPEAT_OP
    }
    repeats: dict[str, _Repeat] = {}
    for idx, entry in enumerate(ops):
        if not isinstance(entry, dict) or entry.get("op") != REPEAT_OP:
            continue
        spec = _parse_repeat(entry, idx)
        if spec.id in repeats or spec.id in known_ids or spec.id in plain:
            raise ValueError(f"duplicate id '{spec.id}'")
        repeats[spec.id] = spec

    owner: dict[str, str] = {}
    for spec in repeats.values():
        for body_id in spec.body_ids:
            if body_id in owner or body_id in plain:
                raise ValueError(
                    f"Repeat '{spec.id}' body id '{body_id}' collides with an id "
                    "outside the repeat"
                )
            owner[body_id] = spec.id
        if spec.output_ref not in spec.body_ids:
            raise ValueError(
                f"Repeat '{spec.id}' output '{spec.output_ref}' is not one of its "
                f"body ids {sorted(spec.body_ids)}"
            )
        if spec.input_ref not in known_ids | plain:
            raise ValueError(f"Repeat '{spec.id}' input '{spec.input_ref}' is unknown")

    outside = {
        rid: spec.at(spec.times, spec.output_ref) for rid, spec in repeats.items()
    }
    frames: dict[str, dict[str, str]] = {}
    unrolled: list[Any] = []
    for entry in ops:
        if not isinstance(entry, dict) or entry.get("op") != REPEAT_OP:
            unrolled.append(entry)
            continue
        spec = repeats[entry["id"]]
        for index in range(1, spec.times + 1):
            iteration = spec.names(index)
            frames.update((spec.at(index, bid), iteration) for bid in spec.body_ids)
            unrolled.extend({**copy, "id": iteration[copy["id"]]} for copy in spec.body)

    def bind(ids: Mapping[str, str], where: str, referrer: str) -> _Scoped:
        return _Scoped(ids, where, owner, frames.get(referrer, outside))

    return unrolled, bind


def _parse_repeat(entry: dict[str, Any], idx: int) -> _Repeat:
    repeat_id = _entry_id(entry)
    if not repeat_id:
        raise ValueError(f"Repeat at index {idx} missing 'id'")
    if unknown := sorted(set(entry) - _FIELDS):
        raise ValueError(f"Repeat '{repeat_id}' has unknown fields: {unknown}")

    times = entry.get("times")
    if isinstance(times, bool) or not isinstance(times, int) or times < 1:
        raise ValueError(f"Repeat '{repeat_id}' 'times' must be a positive integer")

    raw_body = entry.get("body")
    if not isinstance(raw_body, list) or not raw_body:
        raise ValueError(f"Repeat '{repeat_id}' requires a non-empty 'body' list")
    body: list[dict[str, Any]] = []
    for body_idx, candidate in enumerate(raw_body):
        if not isinstance(candidate, dict):
            raise ValueError(
                f"Repeat '{repeat_id}' body entry {body_idx} must be a mapping"
            )
        body_id, op_kind = _entry_id(candidate), candidate.get("op")
        if op_kind in (REPEAT_OP, ROUTER_OP):
            raise ValueError(f"Repeat '{repeat_id}' body may not hold a '{op_kind}'")
        if not body_id or "." in body_id:
            raise ValueError(
                f"Repeat '{repeat_id}' body id '{body_id}' may not contain '.'"
            )
        body.append({**candidate, "id": body_id})

    source, carried = (_ref(entry, repeat_id, key) for key in ("input", "output"))
    return _Repeat(repeat_id, times, source, carried, body)


def _ref(entry: dict[str, Any], repeat_id: str, key: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Repeat '{repeat_id}' requires a non-empty '{key}'")
    if INPUT_REF in value:
        raise ValueError(f"Repeat '{repeat_id}' may not use '{INPUT_REF}' in '{key}'")
    return value.strip()


def _entry_id(entry: Any) -> str:
    value = entry.get("id") if isinstance(entry, dict) else None
    return value.strip() if isinstance(value, str) else ""
