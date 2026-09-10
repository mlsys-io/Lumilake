"""Verify the live/non-live progress rendering surfaces dynamic execution
(rounds) progress as well as static batch progress."""

from lumilake_cli.commands.job import (
    _execution_summary,
    _format_progress_line,
)


def _dynamic_progress(succeeded: int, pending: int, completed: bool) -> dict:
    return {
        "execution": {
            "completed": completed,
            "details": {
                "succeeded": succeeded,
                "failed": 0,
                "pending": pending,
                "dispatched": 0,
            },
        },
        "batch_progress": {
            "total": 0,
            "completed": 0,
            "running": 0,
            "pending": 0,
            "failed": 0,
            "batches": [],
            "overall_progress": {
                "total_nodes": 0,
                "completed_nodes": 0,
                "percentage": 0.0,
                "total_inputs": 0,
                "completed_inputs": 0,
                "raw_nodes": 0,
                "flowmesh_nodes": 0,
                "total_nodes_runtime": 0,
                "pending_runtime_nodes_raw": 0,
                "processing_runtime_nodes_raw": 0,
                "processing_runtime_nodes_optimized": 0,
                "processed_runtime_nodes_raw": 0,
                "processed_runtime_nodes_optimized": 0,
            },
            "eta_seconds": None,
        },
    }


def test_execution_summary_renders_rounds() -> None:
    text = _execution_summary(
        _dynamic_progress(succeeded=2, pending=1, completed=False)
    )
    assert text is not None
    rendered = text.plain
    assert "Execution (running)" in rendered
    assert "2/3 rounds done" in rendered
    assert "1 pending" in rendered


def test_execution_summary_marks_completed() -> None:
    text = _execution_summary(_dynamic_progress(succeeded=3, pending=0, completed=True))
    assert text is not None
    assert "Execution (done)" in text.plain
    assert "3/3 rounds done" in text.plain


def test_format_progress_line_includes_rounds() -> None:
    line = _format_progress_line(
        "req-abc",
        "running",
        _dynamic_progress(succeeded=1, pending=2, completed=False),
        5.0,
    )
    assert "1/3 rounds" in line
    assert "req-abc" in line
