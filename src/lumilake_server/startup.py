"""Server startup: reconcile helper, invoked from main.lifespan."""

import logging
from typing import Protocol, runtime_checkable

from lumid_hooks import ResourceRef

from lumilake_server import hooks
from lumilake_server.hooks import ResourceKind
from lumilake_server.utils.job_storage import get_job_storage


@runtime_checkable
class ReconcilingRegistrar(Protocol):
    async def reconcile(
        self,
        refs: list[ResourceRef],
        logger: logging.Logger,
    ) -> None: ...


async def reconcile_registrars(logger: logging.Logger) -> None:
    """Pass all known job refs to every registrar that has a reconcile method.

    A failure in one registrar is logged and the sweep continues.
    """
    storage = get_job_storage()
    job_refs = [
        ResourceRef(kind=ResourceKind.JOB.value, id=summary.job_id)
        for summary in storage.iter_summaries()
    ]
    for registrar in hooks.RESOURCE_REGISTRARS:
        if not isinstance(registrar, ReconcilingRegistrar):
            continue
        try:
            await registrar.reconcile(job_refs, logger)
            logger.info(
                "ResourceRegistrar %s reconciled: kind=%s, ids=%d",
                registrar.name,
                ResourceKind.JOB.value,
                len(job_refs),
            )
        except Exception:
            logger.exception(
                "ResourceRegistrar %s reconcile failed; store left untouched.",
                registrar.name,
            )
