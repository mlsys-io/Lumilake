"""Version helpers for the FlowMesh SDK package."""

from importlib.metadata import PackageNotFoundError, version

_PACKAGE_NAME = "flowmesh-sdk"
_STATIC_VERSION = "0.1.10rc1"


def _resolve_version() -> str:
    try:
        return version(_PACKAGE_NAME)
    except PackageNotFoundError:
        return _STATIC_VERSION


__version__ = _resolve_version()
USER_AGENT = f"flowmesh-sdk-python/{__version__}"
