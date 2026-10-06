"""The package root re-exports ``WorkflowUsage`` so callers can import the
usage model without reaching into an implementation module."""

from lumilake import WorkflowUsage
from lumilake.resources._log_models import WorkflowUsage as _ImplWorkflowUsage


def test_workflow_usage_reexported_from_package_root() -> None:
    assert WorkflowUsage is _ImplWorkflowUsage


def test_workflow_usage_all_zero() -> None:
    usage = WorkflowUsage(
        prompt_tokens=0,
        completion_tokens=0,
        reasoning_tokens=0,
        calls=0,
        retries=0,
        truncated_calls=0,
        wall_sec=0.0,
    )
    assert usage.model_dump() == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "calls": 0,
        "retries": 0,
        "truncated_calls": 0,
        "wall_sec": 0.0,
    }
