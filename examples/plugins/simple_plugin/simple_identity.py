import logging

from lumid_hooks import PrincipalContext

from . import state


class SimpleIdentityProvider:
    name = "simple_plugin.identity"

    async def resolve(
        self,
        raw_token: str,
        logger: logging.Logger,
    ) -> PrincipalContext | None:
        principal = state.TOKENS.get(raw_token)
        if principal is None:
            logger.info("%s: unknown token, deferring", self.name)
            return None
        logger.info("%s: resolved principal_id=%s", self.name, principal.principal_id)
        return principal
