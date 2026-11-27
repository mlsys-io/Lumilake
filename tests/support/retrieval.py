from typing import Any

from lumilake_server.ops import DataRetrievalOp, Op
from lumilake_server.utils.retrieval_spec import RetrievalSpec


def retrieval_op(
    data_spec: dict[str, Any], inputs: list[Op] | None = None
) -> DataRetrievalOp:
    return DataRetrievalOp(
        data_spec=RetrievalSpec.model_validate(data_spec), inputs=inputs
    )
