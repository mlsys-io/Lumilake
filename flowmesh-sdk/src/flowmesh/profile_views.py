"""Stringification / dataframe helpers for `ProfileSummary`.

The SDK leaves rendering to the caller, but exposes a few convenience
adapters so downstream Python tools (notebooks, lumilake, ad-hoc scripts)
don't have to reimplement them.
"""

import re
from typing import Any

import pandas as pd

from .models.traces import EventSummary, ProfileSummary

_MERMAID_SAFE = re.compile(r"[^A-Za-z0-9_]+")


def _mermaid_node_id(value: str) -> str:
    cleaned = _MERMAID_SAFE.sub("_", value).strip("_")
    return cleaned or "node"


def _event_summary_dataframe(summary: EventSummary) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "event_type": summary.event_type,
            "count": summary.count,
            "total_seconds": summary.total_seconds,
            "avg_seconds": summary.avg_seconds,
            "min_seconds": summary.min_seconds,
            "max_seconds": summary.max_seconds,
        }
    )
    return df.sort_values("total_seconds", ascending=False).reset_index(drop=True)


def hardware_dataframe(
    summary: ProfileSummary, *, on_critical_path: bool = False
) -> pd.DataFrame:
    """Compute-time breakdown as a DataFrame, restricted to the CP if requested."""
    hw = (
        summary.critical_path.hardware_summary
        if on_critical_path and summary.critical_path is not None
        else summary.e2e_breakdown.hardware_summary
    )
    return _event_summary_dataframe(hw)


def network_dataframe(
    summary: ProfileSummary, *, on_critical_path: bool = False
) -> pd.DataFrame:
    """Network-active-time breakdown as a DataFrame; restrict to CP if requested."""
    net = (
        summary.critical_path.network_summary
        if on_critical_path and summary.critical_path is not None
        else summary.e2e_breakdown.network_summary
    )
    return _event_summary_dataframe(net)


def critical_path_dataframe(summary: ProfileSummary) -> pd.DataFrame:
    """Per-node active vs wait on the critical path."""
    if summary.critical_path is None:
        return pd.DataFrame(columns=["data_id", "active_seconds", "wait_seconds"])
    awb = summary.critical_path.active_wait_breakdown
    return pd.DataFrame(
        {
            "data_id": awb.data_id,
            "active_seconds": awb.active_seconds,
            "wait_seconds": awb.wait_seconds,
        }
    )


def to_mermaid(summary: ProfileSummary | dict[str, Any]) -> str:
    """Lineage DAG as Mermaid ``graph TD`` source."""
    if isinstance(summary, dict):
        summary = ProfileSummary.model_validate(summary)
    lines = ["graph TD"]
    seen: set[str] = set()
    for edge in summary.lineage:
        src = _mermaid_node_id(edge.source_data_id)
        dst = _mermaid_node_id(edge.data_id)
        if src not in seen:
            lines.append(f'    {src}["{edge.source_data_id}"]')
            seen.add(src)
        if dst not in seen:
            lines.append(f'    {dst}["{edge.data_id}"]')
            seen.add(dst)
        lines.append(f"    {src} --> {dst}")
    for data_id in summary.data_ids:
        node = _mermaid_node_id(data_id)
        if node not in seen:
            lines.append(f'    {node}["{data_id}"]')
            seen.add(node)
    return "\n".join(lines)
