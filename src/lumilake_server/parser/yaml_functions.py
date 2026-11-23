"""Workflow-level ``functions:`` sources and the ``uses:`` that include them."""

import ast
import symtable
from dataclasses import dataclass
from typing import Annotated, Any, Self

from pydantic import (
    BaseModel,
    RootModel,
    StringConstraints,
    ValidationError,
    field_validator,
)

_SEPARATOR = "\n\n"
_Ident = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _validated[M: BaseModel](model: type[M], payload: Any, where: str) -> M:
    """``payload`` as ``model``; its first error names ``where``."""
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        error = exc.errors()[0]
        if error["type"] == "value_error":
            raise ValueError(f"{where}: {error['ctx']['error']}") from exc
        raise ValueError(f"{where}: {error['msg']}") from exc


def _import_targets(tree: ast.Module) -> dict[str, str]:
    """The module each top-level import binds a name to."""
    targets: dict[str, str] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.asname:
                    targets[alias.asname] = alias.name
                else:
                    bound = alias.name.split(".", 1)[0]
                    targets[bound] = bound
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
    """The names both sources bind, ignoring an import written twice."""
    collisions: set[str] = set()
    for name in first.names & second.names:
        if name in first.imports and name in second.imports:
            if first.imports[name] == second.imports[name]:
                continue
        collisions.add(name)
    return collisions


class WorkflowFunctions(RootModel[dict[_Ident, _Ident]]):
    """A workflow's ``functions:`` block, keyed by its ``uses:`` name."""

    @field_validator("root")
    @classmethod
    def _each_parses(cls, sources: dict[str, str]) -> dict[str, str]:
        for name, source in sources.items():
            try:
                ast.parse(source)
            except SyntaxError as exc:
                raise ValueError(
                    f"function {name!r} does not parse as Python: {exc.msg} (line "
                    f"{exc.lineno})"
                ) from exc
        return sources

    def include(self, uses: list[str], op_id: str, code: str | None) -> str:
        """The op's source with the blocks it ``uses`` appended."""
        unknown = [name for name in uses if name not in self.root]
        if unknown:
            raise ValueError(
                f"LambdaOp '{op_id}' uses unknown function {unknown[0]!r}; 'functions' "
                f"defines {sorted(self.root)}"
            )
        own = None
        if code is not None:
            own = _Bindings.parse(code, f"LambdaOp '{op_id}' code")
        blocks = [code.strip()] if code is not None else []
        included: list[tuple[str, _Bindings]] = []
        for name in uses:
            source = self.root[name]
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
                        f"LambdaOp '{op_id}' code defines {shared[0]!r}, which the "
                        f"function {name!r} it uses also defines"
                    )
            included.append((name, current))
            blocks.append(source.strip())
        return _SEPARATOR.join(blocks)


def parse_functions(block: Any) -> WorkflowFunctions:
    """A workflow's ``functions:`` block."""
    return _validated(
        WorkflowFunctions,
        block if block is not None else {},
        "'functions'",
    )
