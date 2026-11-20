"""Join lowering: one standalone python step merging keyed rows.

The first join input fixes the output rows. Each further input contributes the
fields of the row with the same key, and each attach value lands on the row it
was computed from. The step is built by :mod:`join_build`, which a consumer
that lowers before this phase also calls, so a join's output feeds any op that
reads it.
"""

from lumilake_server.ops.join_ops import JoinOp
from lumilake_server.runtime.runtime_graph.base import _RuntimeGraphBuilderBase
from lumilake_server.runtime.runtime_graph.join_build import build_join_step
from lumilake_server.runtime.runtime_graph.lowering import (
    LoweringContext,
    LoweringResult,
    Phase,
    register,
)


class _JoinBuilder(_RuntimeGraphBuilderBase):
    @register(JoinOp, Phase.JOIN)
    def _lower_join(self, op: JoinOp, ctx: LoweringContext) -> LoweringResult | None:
        """Lower a join to its own python step, unless a consumer that lowers
        earlier already built it."""
        if op.id in ctx.dsl_to_runtime:
            return None
        return LoweringResult(nodes=build_join_step(self, op, ctx))
