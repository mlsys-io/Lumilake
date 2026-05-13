import logging
from collections.abc import Sequence

from lumilake_hook import UsageRow

from . import state


class SimpleUsageSink:
    name = "simple_plugin.usage"

    async def emit(self, rows: Sequence[UsageRow], logger: logging.Logger) -> None:
        state.USAGE_LEDGER.extend(rows)
        logger.info(
            "%s: appended %d row(s), ledger_size=%d",
            self.name,
            len(rows),
            len(state.USAGE_LEDGER),
        )
