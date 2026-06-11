import lumid_hooks
from lumid_hooks import (
    IdentityProvider,
    PermissionChecker,
    PrincipalContext,
    ResourceRef,
    ResourceRegistrar,
    SubmissionGuard,
)
from lumilake_hook import (
    BaseBindings,
    HookBindings,
    LumilakeUsageSink,
    OptimizerProvider,
    ResourceAction,
    ResourceKind,
    UsageRow,
)

from .optimizer_providers import OPTIMIZER_PROVIDERS

IDENTITY_PROVIDERS: list[IdentityProvider] = []
SUBMISSION_GUARDS: list[SubmissionGuard] = []
USAGE_SINKS: list[LumilakeUsageSink] = []
PERMISSION_CHECKERS: list[PermissionChecker] = []
RESOURCE_REGISTRARS: list[ResourceRegistrar] = []


def register(bindings: lumid_hooks.HookBindings) -> None:
    IDENTITY_PROVIDERS.extend(bindings.identity_providers)
    SUBMISSION_GUARDS.extend(bindings.submission_guards)
    USAGE_SINKS.extend(bindings.usage_sinks)
    PERMISSION_CHECKERS.extend(bindings.permission_checkers)
    RESOURCE_REGISTRARS.extend(bindings.resource_registrars)
    if isinstance(bindings, HookBindings):
        OPTIMIZER_PROVIDERS.extend(bindings.optimizer_providers)


__all__ = [
    "BaseBindings",
    "HookBindings",
    "IDENTITY_PROVIDERS",
    "IdentityProvider",
    "LumilakeUsageSink",
    "OptimizerProvider",
    "PERMISSION_CHECKERS",
    "PermissionChecker",
    "PrincipalContext",
    "RESOURCE_REGISTRARS",
    "ResourceAction",
    "ResourceKind",
    "ResourceRef",
    "ResourceRegistrar",
    "SUBMISSION_GUARDS",
    "SubmissionGuard",
    "USAGE_SINKS",
    "UsageRow",
    "register",
]
