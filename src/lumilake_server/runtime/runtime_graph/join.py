"""Join lowering: one standalone python step merging keyed rows.

The join phase runs after every other phase, so the ops a join reads have their runtime
nodes before its step is built.
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
    def _lower_join(self, op: JoinOp, ctx: LoweringContext) -> LoweringResult:
        """Lower a join to its own python step."""
        return LoweringResult(nodes=[build_join_step(self, op, ctx)])
