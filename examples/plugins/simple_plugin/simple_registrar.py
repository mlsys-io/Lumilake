import logging

from lumid_hooks import PrincipalContext, ResourceRef

from . import state


class SimpleResourceRegistrar:
    name = "simple_plugin.registrar"

    async def register(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        logger: logging.Logger,
    ) -> None:
        if resource.id is None:
            return
        state.OWNERSHIP[(resource.kind, resource.id)] = principal.principal_id
        logger.info(
            "%s: registered %s/%s for principal_id=%s",
            self.name,
            resource.kind,
            resource.id,
            principal.principal_id,
        )

    async def deregister(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        logger: logging.Logger,
    ) -> None:
        if resource.id is None:
            return
        state.OWNERSHIP.pop((resource.kind, resource.id), None)
        logger.info("%s: deregistered %s/%s", self.name, resource.kind, resource.id)
