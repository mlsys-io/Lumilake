import pytest

from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager


def _manager() -> FlowmeshRuntimeManager:
    return FlowmeshRuntimeManager()


def test_api_reasoning_present_still_raises_clear_error() -> None:
    """A reasoning model returns content=None with a reasoning fragment when
    its budget is exhausted; that must fail with guidance, not surface the
    truncated fragment as the node output."""
    results = {
        "text": None,
        "response_json": {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning": (
                            "The user wants me to calculate 17 x 23 and show my"
                        ),
                    },
                }
            ]
        },
    }
    with pytest.raises(RuntimeError, match="raise max_tokens"):
        _manager()._resolve_output_items(results, "out-1", task_type="api")
