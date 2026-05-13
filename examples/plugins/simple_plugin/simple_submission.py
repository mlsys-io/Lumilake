import logging

from fastapi import HTTPException
from lumid_hooks import PrincipalContext

from . import state


class SimpleSubmissionGuard:
    name = "simple_plugin.submission"

    async def check(self, principal: PrincipalContext, logger: logging.Logger) -> None:
        if principal.principal_id in state.BLOCKED_PRINCIPALS:
            logger.warning(
                "%s: blocking principal_id=%s", self.name, principal.principal_id
            )
            raise HTTPException(status_code=403, detail="principal is blocked")
        logger.info("%s: allowing principal_id=%s", self.name, principal.principal_id)
