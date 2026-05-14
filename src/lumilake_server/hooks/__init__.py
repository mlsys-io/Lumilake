from lumid_hooks import (
    HookBindings,
    IdentityProvider,
    PermissionChecker,
    PrincipalContext,
    ResourceRef,
    ResourceRegistrar,
    SubmissionGuard,
)
from lumilake_hook import (
    BaseBindings,
    LumilakeUsageSink,
    ResourceAction,
    ResourceKind,
    UsageRow,
)

IDENTITY_PROVIDERS: list[IdentityProvider] = []
SUBMISSION_GUARDS: list[SubmissionGuard] = []
USAGE_SINKS: list[LumilakeUsageSink] = []
PERMISSION_CHECKERS: list[PermissionChecker] = []
RESOURCE_REGISTRARS: list[ResourceRegistrar] = []


def register(bindings: HookBindings) -> None:
    IDENTITY_PROVIDERS.extend(bindings.identity_providers)
    SUBMISSION_GUARDS.extend(bindings.submission_guards)
    USAGE_SINKS.extend(bindings.usage_sinks)
    PERMISSION_CHECKERS.extend(bindings.permission_checkers)
    RESOURCE_REGISTRARS.extend(bindings.resource_registrars)


__all__ = [
    "BaseBindings",
    "HookBindings",
    "IDENTITY_PROVIDERS",
    "IdentityProvider",
    "LumilakeUsageSink",
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
