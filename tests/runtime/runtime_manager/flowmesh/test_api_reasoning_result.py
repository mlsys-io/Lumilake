import pytest

from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager


def _manager() -> FlowmeshRuntimeManager:
    return FlowmeshRuntimeManager()


def test_api_reasoning_content_surfaced_when_content_none() -> None:
    """A reasoning model returns content=None with a 'reasoning' payload when
    its budget is exhausted; that reasoning must be surfaced, not treated as
    'no response text'."""
    results = {
        "text": None,
        "response_json": {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning": "chain of thought",
                    },
                }
            ],
            "usage": {
                "completion_tokens": 16,
                "completion_tokens_details": {"reasoning_tokens": 16},
            },
        },
    }
    items = _manager()._resolve_output_items(results, "out-1", task_type="api")
    assert items == [{"output": "chain of thought"}]


def test_api_reasoning_content_none_raises_clear_error() -> None:
    """A response with neither content nor reasoning fails with an error that
    names the real cause (reasoning budget exhausted), not a generic message."""
    results = {
        "text": None,
        "response_json": {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": None},
                }
            ]
        },
    }
    with pytest.raises(RuntimeError, match="reasoning model"):
        _manager()._resolve_output_items(results, "out-1", task_type="api")
