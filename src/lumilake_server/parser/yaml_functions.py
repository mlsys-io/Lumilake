"""Workflow-level ``functions:`` sources and the ``uses:`` that includes them.

A workflow may keep helper code in one top-level ``functions:`` mapping and
have each ``LambdaOp`` name the blocks it needs with ``uses:``, so a helper is
written once instead of copied into every op that calls it. The sources are
resolved while the entry is still raw: the named blocks are appended after the
op's ``code`` in the listed order, so the dispatched source, the
inline-namespace checks and every digest see the full program exactly as if it
had been written in the op.

Appending keeps the op's own function the first name the source binds, which
is what an inline Lambda needs. That namespace has no imports and no
``try``/``raise``-family builtins, so an inline Lambda can only use blocks
that need neither.
"""

import ast
import symtable
from dataclasses import dataclass
from typing import Any, Self

_SEPARATOR = "\n\n"


def _import_targets(tree: ast.Module) -> dict[str, str]:
    """The module each top-level import binds a name to."""
    targets: dict[str, str] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                targets[alias.asname or alias.name.split(".", 1)[0]] = alias.name
        elif isinstance(stmt, ast.ImportFrom) and stmt.level == 0 and stmt.module:
            for alias in stmt.names:
                if alias.name != "*":
                    targets[alias.asname or alias.name] = f"{stmt.module}.{alias.name}"
    return targets


@dataclass(frozen=True)
class _Bindings:
    """What a source binds at module level."""

    names: frozenset[str]
    imports: dict[str, str]

    @classmethod
    def parse(cls, source: str, where: str) -> Self:
        try:
            tree = ast.parse(source)
            table = symtable.symtable(source, "<workflow function>", "exec")
        except SyntaxError as exc:
            raise ValueError(f"{where} does not parse as Python: {exc.msg}") from exc
        names = frozenset(
            symbol.get_name()
            for symbol in table.get_symbols()
            if symbol.is_assigned() or symbol.is_imported()
        )
        return cls(names=names, imports=_import_targets(tree))


def _collides(first: _Bindings, second: _Bindings) -> set[str]:
    """The names both sources bind, ignoring the same import written twice."""
    collisions: set[str] = set()
    for name in first.names & second.names:
        if name in first.imports and name in second.imports:
            if first.imports[name] == second.imports[name]:
                continue
        collisions.add(name)
    return collisions


@dataclass(frozen=True)
class WorkflowFunctions:
    """The workflow's ``functions:`` block, keyed by the name ``uses:`` names."""

    sources: dict[str, str]

    @classmethod
    def from_block(cls, raw: Any) -> Self:
        """Validate a top-level ``functions:`` mapping."""
        if raw is None:
            return cls(sources={})
        if not isinstance(raw, dict):
            raise ValueError("'functions' must be a mapping of name -> Python source")
        sources: dict[str, str] = {}
        for name, source in raw.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("'functions' names must be non-empty strings")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(
                    f"function {name!r} must be a non-empty Python source string"
                )
            try:
                ast.parse(source)
            except SyntaxError as exc:
                raise ValueError(
                    f"function {name!r} does not parse as Python: {exc.msg}"
                    f" (line {exc.lineno})"
                ) from exc
            sources[name] = source
        return cls(sources=sources)

    def include(self, uses: Any, op_id: str, code: str | None) -> str:
        """The op's source with the blocks it ``uses`` appended, in order."""
        names = self._checked_uses(uses, op_id)
        own = None
        if code is not None:
            own = _Bindings.parse(code, f"LambdaOp '{op_id}' code")
        blocks = [code.strip()] if code is not None else []
        included: list[tuple[str, _Bindings]] = []
        for name in names:
            source = self.sources[name]
            current = _Bindings.parse(source, f"function {name!r}")
            for other_name, other in included:
                shared = sorted(_collides(other, current))
                if shared:
                    raise ValueError(
                        f"LambdaOp '{op_id}' uses functions {other_name!r} and"
                        f" {name!r}, which both define {shared[0]!r}"
                    )
            if own is not None:
                shared = sorted(_collides(own, current))
                if shared:
                    raise ValueError(
                        f"LambdaOp '{op_id}' code defines {shared[0]!r}, which"
                        f" the function {name!r} it uses also defines"
                    )
            included.append((name, current))
            blocks.append(source.strip())
        return _SEPARATOR.join(blocks)

    def _checked_uses(self, uses: Any, op_id: str) -> list[str]:
        if not isinstance(uses, list) or not all(
            isinstance(name, str) for name in uses
        ):
            raise ValueError(
                f"LambdaOp '{op_id}' 'uses' must be a list of function names"
            )
        seen: set[str] = set()
        for name in uses:
            if name in seen:
                raise ValueError(
                    f"LambdaOp '{op_id}' 'uses' names the function {name!r}"
                    " more than once"
                )
            seen.add(name)
            if name not in self.sources:
                raise ValueError(
                    f"LambdaOp '{op_id}' uses unknown function {name!r};"
                    f" 'functions' defines {sorted(self.sources)}"
                )
        return uses
