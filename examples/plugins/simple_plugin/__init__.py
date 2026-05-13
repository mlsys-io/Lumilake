from lumilake_hook import BaseBindings

from .simple_identity import SimpleIdentityProvider
from .simple_optimizer import install_optimizer
from .simple_permissions import SimplePermissionChecker
from .simple_registrar import SimpleResourceRegistrar
from .simple_submission import SimpleSubmissionGuard
from .simple_usage import SimpleUsageSink


def install() -> BaseBindings:
    install_optimizer()
    return BaseBindings(
        identity_providers=(SimpleIdentityProvider(),),
        submission_guards=(SimpleSubmissionGuard(),),
        usage_sinks=(SimpleUsageSink(),),
        permission_checkers=(SimplePermissionChecker(),),
        resource_registrars=(SimpleResourceRegistrar(),),
    )


__all__ = ["install"]
