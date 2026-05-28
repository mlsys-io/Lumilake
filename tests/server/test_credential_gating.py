"""Static checks that the scheduler FlowMesh credential never reaches a
user-driven code path. Failure here is a privilege-escalation regression."""

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src" / "lumilake_server"
_PACKAGES = _REPO_ROOT / "packages"


def _all_py_files(*roots: Path) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        files.extend(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    return files


def test_runtime_token_env_is_read_only_in_flowmesh_client() -> None:
    """``envs.RUNTIME_TOKEN`` may only be read from ``flowmesh_client.py``."""
    reference = re.compile(r"\benvs\.RUNTIME_TOKEN\b")
    offenders: list[str] = []
    for py in _all_py_files(_SRC, _PACKAGES):
        if py.name == "envs.py":
            continue
        text = py.read_text()
        if reference.search(text):
            rel = py.relative_to(_REPO_ROOT)
            if rel != Path("src/lumilake_server/runtime/flowmesh_client.py"):
                offenders.append(str(rel))
    assert offenders == [], (
        "envs.RUNTIME_TOKEN must only be read from "
        "src/lumilake_server/runtime/flowmesh_client.py; "
        f"unexpected readers: {offenders}"
    )


def test_flowmesh_for_server_only_referenced_from_runtime_manager() -> None:
    """``flowmesh_for_server`` may only be referenced from the runtime
    manager. Scans for bare references and ``import ... as`` aliases."""
    import ast

    direct_reference = re.compile(r"\bflowmesh_for_server\b")
    allowed = {
        Path("src/lumilake_server/runtime/flowmesh_client.py"),  # definition
        Path("src/lumilake_server/runtime/runtime_manager/flowmesh.py"),  # caller
    }
    offenders: list[str] = []
    for py in _all_py_files(_SRC, _PACKAGES):
        rel = py.relative_to(_REPO_ROOT)
        if rel in allowed:
            continue
        text = py.read_text()
        if direct_reference.search(text):
            offenders.append(str(rel))
            continue
        # Catch `from lumilake_server.runtime.flowmesh_client import (
        #     flowmesh_for_server as _alias)`.
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "flowmesh_for_server":
                        offenders.append(str(rel))
                        break
    assert offenders == [], (
        "flowmesh_for_server must only be referenced by flowmesh_client.py "
        f"(definition) and runtime_manager/flowmesh.py (callers); "
        f"unexpected references: {offenders}"
    )


def test_runtime_manager_worker_apis_have_no_route_callers() -> None:
    """``get_workers``/``get_worker_profile`` carry the scheduler credential
    and must not be reachable from any route handler."""
    forbidden = re.compile(r"\.get_workers\b|\.get_worker_profile\b")
    routes_dir = _SRC / "routes"
    offenders: list[str] = []
    for py in _all_py_files(routes_dir):
        rel = py.relative_to(_REPO_ROOT)
        if forbidden.search(py.read_text()):
            offenders.append(str(rel))
    assert offenders == [], (
        "Route handlers must not call runtime_manager.get_workers / "
        "get_worker_profile (they carry the scheduler credential). "
        f"Offending files: {offenders}"
    )


@pytest.mark.parametrize(
    "symbol", ["flowmesh_for_server", "get_workers", "get_worker_profile"]
)
def test_gating_targets_still_exist(symbol: str) -> None:
    """Each gated symbol must still appear under ``src/`` so the scans
    above don't silently no-op after a rename."""
    pattern = re.compile(rf"\b{symbol}\b")
    found = any(pattern.search(p.read_text()) for p in _all_py_files(_SRC))
    assert found, f"{symbol!r} not found anywhere in src/; gating test would no-op"
