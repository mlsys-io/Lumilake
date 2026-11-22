"""Workflow-level ``functions:`` sources and the ``uses:`` that includes them.

A workflow may keep helper code in one top-level ``functions:`` mapping and
have each ``LambdaOp`` name the blocks it needs with ``uses:``, so a helper is
written once instead of copied into every op that calls it. The sources are
resolved while the entry is still raw: the named blocks are prepended to the
op's ``code`` in the listed order, so the dispatched source, the
inline-namespace checks and every digest see the full program exactly as if it
had been written in the op.
"""

import ast
import symtable
from dataclasses import dataclass
from typing import Any, Self

_SEPARATOR = "\n\n"


def _bound_names(source: str) -> set[str]:
    """The names ``source`` binds at module level.

    That is every top-level ``def``, ``class``, import and assignment, plus
    whatever a top-level compound statement (``if``, ``try``, ...) binds.
    """
    table = symtable.symtable(source, "<workflow function>", "exec")
    return {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_assigned() or symbol.is_imported()
    }


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

    def include(
        self, uses: Any, op_uses_ok: bool, op_id: str, code: str | None
    ) -> str | None:
        """The op's source with the blocks it ``uses`` prepended, in order."""
        if uses is None:
            return code
        if not op_uses_ok:
            raise ValueError(
                f"op '{op_id}' declares 'uses', which only a LambdaOp accepts"
            )
        names = self._checked_uses(uses, op_id)
        blocks: list[str] = []
        bound: dict[str, str] = {}
        for name in names:
            source = self.sources[name]
            for bound_name in sorted(_bound_names(source)):
                previous = bound.setdefault(bound_name, name)
                if previous != name:
                    raise ValueError(
                        f"LambdaOp '{op_id}' uses functions {previous!r} and"
                        f" {name!r}, which both define {bound_name!r}"
                    )
            blocks.append(source.strip())
        if code is not None:
            for bound_name in sorted(_bound_names(code)):
                owner = bound.get(bound_name)
                if owner is not None:
                    raise ValueError(
                        f"LambdaOp '{op_id}' code defines {bound_name!r}, which"
                        f" the function {owner!r} it uses also defines"
                    )
            blocks.append(code.strip())
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
