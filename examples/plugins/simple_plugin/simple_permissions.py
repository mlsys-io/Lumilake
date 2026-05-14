import logging

from fastapi import HTTPException
from lumid_hooks import PrincipalContext, ResourceRef
from lumilake_hook import ResourceKind

from . import state


def _is_admin(principal: PrincipalContext) -> bool:
    return "admin" in principal.scopes


def _has_data_access(principal: PrincipalContext) -> bool:
    return _is_admin(principal) or "data" in principal.scopes


class SimplePermissionChecker:
    name = "simple_plugin.permissions"

    async def accessible_ids(
        self,
        principal: PrincipalContext,
        kind: str,
        action: str,
        logger: logging.Logger,
    ) -> frozenset[str] | None:
        if _is_admin(principal):
            return None
        ids = frozenset(
            resource_id
            for (resource_kind, resource_id), owner in state.OWNERSHIP.items()
            if resource_kind == kind and owner == principal.principal_id
        )
        logger.info(
            "%s: principal_id=%s sees %d %s",
            self.name,
            principal.principal_id,
            len(ids),
            kind,
        )
        return ids

    async def require(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        action: str,
        logger: logging.Logger,
    ) -> None:
        if _is_admin(principal):
            return
        if resource.kind in {
            ResourceKind.TABLE.value,
            ResourceKind.OBJECT_PREFIX.value,
        }:
            if _has_data_access(principal):
                return
            raise HTTPException(
                status_code=403, detail="principal may not access data resources"
            )
        if resource.id is None:
            if principal.scopes:
                return
            raise HTTPException(status_code=403, detail="principal has no scopes")
        owner = state.OWNERSHIP.get((resource.kind, resource.id))
        if owner == principal.principal_id:
            return
        logger.warning(
            "%s: deny %s on %s/%s for principal_id=%s",
            self.name,
            action,
            resource.kind,
            resource.id,
            principal.principal_id,
        )
        raise HTTPException(
            status_code=403,
            detail=f"principal may not {action} {resource.kind}/{resource.id}",
        )
