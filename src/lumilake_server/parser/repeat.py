"""The ``RepeatOp`` YAML construct: a fixed-count repeat of an op body.

:func:`unroll_repeats` rewrites each repeat entry into ordinary op entries
before any id is registered, so one declared op exists per iteration.
"""

from collections.abc import Callable, Iterator, Mapping
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict, StrictInt

from lumilake_server.ops.router_ops import ROUTER_OP, RouterSpec
from lumilake_server.ops.spec_validation import validate_fields
from lumilake_server.parser.keyed_seed import keyed_seed

REPEAT_OP = "RepeatOp"
INPUT_REF = "$in"
_MAX_UNROLLED = 1000
_FIELDS = frozenset({"id", "op", "times", "input", "output", "body"})


class _SeedConfig(BaseModel):
    """The part of a body entry's ``config`` the unroll reads."""

    model_config = ConfigDict(extra="allow")

    seed: StrictInt | None = None


class _BodyEntry(BaseModel):
    """A body entry as the unroll reads it: only its seed."""

    model_config = ConfigDict(extra="allow")

    config: _SeedConfig | None = None


class _Repeat(NamedTuple):
    id: str
    times: int
    input_ref: str
    output_ref: str
    body: list[dict[str, Any]]
    branches: dict[str, tuple[str, ...]]

    @property
    def body_ids(self) -> list[str]:
        return [entry["id"] for entry in self.body]

    @property
    def written_ids(self) -> list[str]:
        """Every name one iteration registers: each body id and its router branches."""
        names: list[str] = []
        for body_id in self.body_ids:
            names.append(body_id)
            names.extend(
                f"{body_id}.{branch}" for branch in self.branches.get(body_id, ())
            )
        return names

    def at(self, index: int, body_id: str) -> str:
        return f"{self.id}.{index}.{body_id}"

    def names(self, index: int, source: str) -> dict[str, str]:
        """The names iteration ``index`` writes, ``$in`` included."""
        read = source if index == 1 else self.at(index - 1, self.output_ref)
        return {
            INPUT_REF: read,
            **{name: self.at(index, name) for name in self.written_ids},
        }


class _Scoped(Mapping[str, str]):
    """A parser id map with one op's scoped names in front of it: a lookup
    that is out of place at the referring op raises."""

    def __init__(
        self,
        ids: Mapping[str, str],
        where: str,
        body_owner: dict[str, str],
        names: dict[str, str],
        own: str | None,
    ) -> None:
        self._ids = ids
        self._where = where
        self._body_owner = body_owner
        self._names = names
        self._own = own

    def target(self, ref: Any) -> Any:
        """The id ``ref`` names in this op's scope, before id derivation."""
        if not isinstance(ref, str):
            return ref
        if ref == self._own:
            raise ValueError(f"RepeatOp '{self._own}' body reads its own id")
        if ref in self._names:
            return self._names[ref]
        if ref == INPUT_REF:
            raise ValueError(f"{self._where} may not read '{INPUT_REF}'")
        owner = self._body_owner.get(ref)
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


class Unrolled(NamedTuple):
    """The unrolled ops, their reference binder and each copy's seed identity."""

    ops: list[Any]
    bind: References
    template_ids: dict[str, str]
    rounds: dict[str, int]
    copied_routers: frozenset[str]


def unroll_repeats(ops: list[Any], known_ids: set[str]) -> Unrolled:
    """Replace every ``RepeatOp`` entry with its unrolled iterations.

    ``known_ids`` holds the workflow input names, which a repeat's ``input``
    may name. Repeats are unrolled in dependency order. The returned binder
    takes a parser id map to one referring op. Each copy keeps the seed
    identity of the body entry and one-based round it runs as.
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
    if repeats:
        plain |= _router_branch_ids(ops)

    owner: dict[str, str] = {}
    for spec in repeats.values():
        for body_id in spec.written_ids:
            if body_id in owner or body_id in plain:
                raise ValueError(
                    f"RepeatOp '{spec.id}' body id '{body_id}' collides with an id "
                    "outside the repeat"
                )
            owner[body_id] = spec.id
        if spec.output_ref in spec.branches:
            branches = ", ".join(
                f"'{spec.output_ref}.{branch}'"
                for branch in spec.branches[spec.output_ref]
            )
            raise ValueError(
                f"RepeatOp '{spec.id}' output '{spec.output_ref}' is a body router; "
                f"name its branch, one of {branches}"
            )
        if spec.output_ref not in spec.body_ids:
            raise ValueError(
                f"RepeatOp '{spec.id}' output '{spec.output_ref}' is not one of its "
                f"body ids {sorted(spec.body_ids)}"
            )
        if spec.input_ref not in known_ids | plain | (set(repeats) - {spec.id}):
            raise ValueError(
                f"RepeatOp '{spec.id}' input '{spec.input_ref}' is unknown"
            )

    ordered = _in_dependency_order(repeats)
    outside: dict[str, str] = {}
    sources: dict[str, str] = {}
    total = 0
    for spec in ordered:
        sources[spec.id] = outside.get(spec.input_ref, spec.input_ref)
        outside[spec.id] = spec.at(spec.times, spec.output_ref)
        total += spec.times * len(spec.body)
        if total > _MAX_UNROLLED:
            raise ValueError(
                f"RepeatOp '{spec.id}' would unroll {total} entries, "
                f"over the limit of {_MAX_UNROLLED}"
            )

    frames: dict[str, dict[str, str]] = {}
    frame_owner: dict[str, str] = {}
    iterations: dict[str, list[Any]] = {}
    template_ids: dict[str, str] = {}
    rounds: dict[str, int] = {}
    copied_routers: set[str] = set()
    for spec in ordered:
        iterations[spec.id] = []
        for index in range(1, spec.times + 1):
            iteration = spec.names(index, sources[spec.id])
            frames.update(
                (spec.at(index, name), {**outside, **iteration})
                for name in spec.written_ids
            )
            frame_owner.update(
                (spec.at(index, name), spec.id) for name in spec.written_ids
            )
            copies = [
                _iteration_copy(copy, iteration[copy["id"]], spec.id, index)
                for copy in spec.body
            ]
            for body, copy in zip(spec.body, copies):
                template_ids[copy["id"]] = body["id"]
                rounds[copy["id"]] = index
                if body["id"] in spec.branches:
                    copied_routers.add(copy["id"])
            iterations[spec.id].extend(copies)

    unrolled: list[Any] = []
    for entry in ops:
        if isinstance(entry, dict) and entry.get("op") == REPEAT_OP:
            unrolled.extend(iterations[entry["id"]])
        else:
            unrolled.append(entry)

    def bind(ids: Mapping[str, str], where: str, referrer: str) -> _Scoped:
        return _Scoped(
            ids,
            where,
            owner,
            frames.get(referrer, outside),
            frame_owner.get(referrer),
        )

    return Unrolled(unrolled, bind, template_ids, rounds, frozenset(copied_routers))


def _iteration_copy(
    entry: dict[str, Any], new_id: str, repeat_id: str, index: int
) -> dict[str, Any]:
    """``entry`` as iteration ``index`` of ``repeat_id`` runs it."""
    copy = {**entry, "id": new_id}
    seed = _literal_seed(entry, repeat_id)
    if seed is None:
        return copy
    redrawn = _iteration_seed(seed, entry["id"], index)
    return {**copy, "config": {**entry["config"], "seed": redrawn}}


def _literal_seed(entry: dict[str, Any], repeat_id: str) -> int | None:
    """The literal seed ``entry`` carries, or None when its config has none."""
    where = f"RepeatOp '{repeat_id}' body entry '{entry['id']}'"
    config = validate_fields(_BodyEntry, entry, where).config
    return None if config is None else config.seed


def _iteration_seed(seed: int, body_id: str, index: int) -> int:
    """The seed iteration ``index`` of body entry ``body_id`` draws."""
    return keyed_seed(seed, body_id, index, "")


def _router_branch_ids(ops: list[Any]) -> set[str]:
    """The branch ids the routers outside any repeat register, ``<id>.<branch>``."""
    branches: set[str] = set()
    for entry in ops:
        if not isinstance(entry, dict) or entry.get("op") != ROUTER_OP:
            continue
        router = validate_fields(RouterSpec, entry, f"RouterOp '{_entry_id(entry)}'")
        branches.update(f"{router.id}.{branch}" for branch in router.branches)
    return branches


def _in_dependency_order(repeats: dict[str, _Repeat]) -> list[_Repeat]:
    """``repeats`` in unroll order: a repeat's source repeat comes first."""
    ordered: list[_Repeat] = []
    pending = repeats.copy()
    while pending:
        ready = [spec for spec in pending.values() if spec.input_ref not in pending]
        if not ready:
            named = ", ".join(f"'{rid}'" for rid in sorted(pending))
            raise ValueError(f"cycle detected between repeats {named}")
        ordered.extend(ready)
        pending = {rid: spec for rid, spec in pending.items() if spec not in ready}
    return ordered


def _parse_repeat(entry: dict[str, Any], idx: int) -> _Repeat:
    repeat_id = _entry_id(entry)
    if not repeat_id:
        raise ValueError(f"RepeatOp at index {idx} missing 'id'")
    if unknown := sorted(set(entry) - _FIELDS):
        raise ValueError(f"RepeatOp '{repeat_id}' has unknown fields: {unknown}")

    times = entry.get("times")
    if isinstance(times, bool) or not isinstance(times, int) or times < 1:
        raise ValueError(f"RepeatOp '{repeat_id}' 'times' must be a positive integer")

    raw_body = entry.get("body")
    if not isinstance(raw_body, list) or not raw_body:
        raise ValueError(f"RepeatOp '{repeat_id}' requires a non-empty 'body' list")
    body: list[dict[str, Any]] = []
    branches: dict[str, tuple[str, ...]] = {}
    for body_idx, candidate in enumerate(raw_body):
        if not isinstance(candidate, dict):
            raise ValueError(
                f"RepeatOp '{repeat_id}' body entry {body_idx} must be a mapping"
            )
        body_id, op_kind = _entry_id(candidate), candidate.get("op")
        if op_kind == REPEAT_OP:
            raise ValueError(f"RepeatOp '{repeat_id}' body may not hold a '{op_kind}'")
        if not body_id or "." in body_id:
            raise ValueError(
                f"RepeatOp '{repeat_id}' body id '{body_id}' may not contain '.'"
            )
        body.append({**candidate, "id": body_id})
        if op_kind == ROUTER_OP:
            router = validate_fields(
                RouterSpec, candidate, f"RepeatOp '{repeat_id}' body entry '{body_id}'"
            )
            branches[body_id] = router.branches

    source, carried = (_ref(entry, repeat_id, key) for key in ("input", "output"))
    return _Repeat(repeat_id, times, source, carried, body, branches)


def _ref(entry: dict[str, Any], repeat_id: str, key: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"RepeatOp '{repeat_id}' requires a non-empty '{key}'")
    if INPUT_REF in value:
        raise ValueError(f"RepeatOp '{repeat_id}' may not use '{INPUT_REF}' in '{key}'")
    return value.strip()


def _entry_id(entry: Any) -> str:
    value = entry.get("id") if isinstance(entry, dict) else None
    return value.strip() if isinstance(value, str) else ""
