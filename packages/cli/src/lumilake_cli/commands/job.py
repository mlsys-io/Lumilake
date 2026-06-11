"""Job submission, monitoring, and result retrieval commands."""

import json
import tarfile
import time
from pathlib import Path
from typing import Any

import requests
import typer
from rich.console import Console
from rich.table import Table

from ..core import logging
from ..core.http import HttpError, client_from_config
from ..core.query import extend_params
from ..core.typer import get_typer

app = get_typer(help="Submit, monitor, and manage optimization jobs.")


def _unwrap(response_json: dict[str, Any]) -> dict[str, Any]:
    """Unwrap the ``{"ok": ..., "data": {...}}`` envelope used by all API responses.

    Returns the inner ``data`` payload when present, or the raw response otherwise.
    """
    return response_json.get("data", response_json)


def _build_inputs(
    input_values: list[str] | None,
    input_files: list[str] | None,
    input_json: Path | None,
) -> dict[str, Any]:
    """Build the inputs dict from CLI flags.

    Raises typer.Exit on validation errors.
    """
    inputs: dict[str, Any] = {}

    # --input-json takes a full JSON object
    if input_json is not None:
        if not input_json.is_file():
            logging.error(f"Input JSON file not found: {input_json}")
            raise typer.Exit(code=1)
        try:
            parsed = json.loads(input_json.read_text())
        except json.JSONDecodeError as exc:
            logging.error(f"Invalid JSON in {input_json}: {exc}")
            raise typer.Exit(code=1)
        if not isinstance(parsed, dict):
            logging.error(f"{input_json} must contain a JSON object")
            raise typer.Exit(code=1)
        inputs.update(parsed)

    # --input-file Name=path.txt  (one value per non-empty line)
    if input_files:
        for raw in input_files:
            if "=" not in raw:
                logging.error(f"Invalid --input-file {raw!r}. Expected Name=path.txt")
                raise typer.Exit(code=1)
            name, file_path_str = raw.split("=", 1)
            name = name.strip()
            if not name:
                logging.error(f"Empty input name in --input-file {raw!r}")
                raise typer.Exit(code=1)
            file_path = Path(file_path_str.strip())
            if not file_path.is_file():
                logging.error(f"Input file not found: {file_path}")
                raise typer.Exit(code=1)
            lines = [
                line.strip()
                for line in file_path.read_text().splitlines()
                if line.strip()
            ]
            if not lines:
                logging.error(f"Input file is empty: {file_path}")
                raise typer.Exit(code=1)
            if name in inputs:
                logging.error(f"Duplicate input name: {name!r}")
                raise typer.Exit(code=1)
            inputs[name] = lines

    # --input Name=val1,val2,val3 (comma-separated; repeatable for different names)
    if input_values:
        for raw in input_values:
            if "=" not in raw:
                logging.error(f"Invalid --input {raw!r}. Expected Name=val1,val2,...")
                raise typer.Exit(code=1)
            name, values_str = raw.split("=", 1)
            name = name.strip()
            if not name:
                logging.error(f"Empty input name in --input {raw!r}")
                raise typer.Exit(code=1)
            if name in inputs:
                logging.error(f"Duplicate input name: {name!r}")
                raise typer.Exit(code=1)
            values = [v.strip() for v in values_str.split(",") if v.strip()]
            if not values:
                logging.error(f"Empty value list in --input {raw!r}")
                raise typer.Exit(code=1)
            inputs[name] = values

    return inputs


@app.command()
def submit(
    workflow: Path = typer.Argument(..., help="Path to workflow file"),
    workflow_format: str = typer.Option(
        "n8n", "--format", "-f", help="Workflow format (native|n8n|yaml)"
    ),
    priority: str = typer.Option(
        "medium", "--priority", "-p", help="Job priority (low|medium|high)"
    ),
    name: str | None = typer.Option(None, "--name", "-n", help="Job name"),
    output_type: str = typer.Option(
        "s3", "--output-type", help="Output location type (s3|db)"
    ),
    output_prefix: str | None = typer.Option(
        None,
        "--output-prefix",
        help="S3 output prefix (required when --output-type s3)",
    ),
    output_connection_string: str | None = typer.Option(
        None,
        "--output-connection-string",
        help="S3 connection string (s3://user:pass@endpoint/bucket)",
    ),
    output_table: str | None = typer.Option(
        None, "--output-table", help="DB output table (required when --output-type db)"
    ),
    output_column: str | None = typer.Option(
        None,
        "--output-column",
        help="DB output column (required when --output-type db)",
    ),
    input_batch_size: int | None = typer.Option(
        None, "--batch-size", help="Input batch size"
    ),
    input_values: list[str] | None = typer.Option(
        None,
        "--input",
        help=(
            "Input as Name=val1,val2,val3 (comma-separated; whitespace around "
            "values is trimmed). Repeat for different names: "
            "`--input query=NVDA,TSLA --input year=2024`. A single value "
            "still uses the same syntax: `--input query=NVDA`."
        ),
    ),
    input_files: list[str] | None = typer.Option(
        None,
        "--input-file",
        help=(
            "Input from file as Name=path.txt; one value per non-empty line. "
            "Repeatable: `--input-file tickers=tickers.txt`."
        ),
    ),
    input_json: Path | None = typer.Option(
        None,
        "--input-json",
        help=(
            "JSON file with the full inputs object, e.g. "
            '`{"query": ["NVDA", "TSLA"], "year": ["2024"]}`.'
        ),
    ),
    optimizer: str | None = typer.Option(
        None,
        "--optimizer",
        help="Override the server default optimizer (must be in /optimizer).",
    ),
) -> None:
    """Submit a workflow for optimization and execution."""
    if not workflow.exists():
        logging.error(f"Workflow file not found: {workflow}")
        raise typer.Exit(code=1)
    if workflow_format not in {"native", "n8n", "yaml"}:
        logging.error(f"Unsupported workflow format: {workflow_format}")
        raise typer.Exit(code=1)

    inputs = _build_inputs(input_values, input_files, input_json)
    if not inputs:
        logging.error(
            "At least one input is required. "
            "Use --input, --input-file, or --input-json."
        )
        raise typer.Exit(code=1)

    workflow_text = workflow.read_text()

    if output_type == "s3":
        if not output_prefix:
            logging.error("--output-prefix is required when --output-type is s3")
            raise typer.Exit(code=1)
        output_location: dict[str, Any] = {"type": "s3", "prefix": output_prefix}
        if output_connection_string:
            output_location["connection_string"] = output_connection_string
    elif output_type == "db":
        if not output_table or not output_column:
            logging.error("--output-table and --output-column required for db output")
            raise typer.Exit(code=1)
        output_location = {
            "type": "db",
            "table": output_table,
            "column": output_column,
        }
    else:
        logging.error(f"Unsupported output type: {output_type}")
        raise typer.Exit(code=1)

    item: dict[str, Any] = {
        "workflow": workflow_text,
        "inputs": inputs,
        "output_location": output_location,
        "name": name or workflow.stem,
    }
    if input_batch_size is not None:
        item["input_batch_size"] = input_batch_size

    payload: dict[str, Any] = {
        "data": [item],
        "priority": priority,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer

    client = client_from_config()
    try:
        response = client.post(
            "/jobs",
            version_prefix=True,
            json=payload,
            headers={"Workflow-Format": workflow_format},
        )
    except HttpError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(json.dumps(response.json(), indent=2))


@app.command("list")
def list_jobs(
    page: int = typer.Option(1, "--page", help="Page number"),
    page_size: int = typer.Option(10, "--page-size", help="Results per page"),
    status: list[str] | None = typer.Option(
        None, "--status", "-s", help="Filter by status (repeatable)"
    ),
    include_all: bool = typer.Option(
        False, "--all", help="Include all users' jobs (requires admin scope)"
    ),
) -> None:
    """List submitted jobs."""
    client = client_from_config()
    params: list[tuple[str, str]] = [
        ("page", str(page)),
        ("page_size", str(page_size)),
    ]
    extend_params(params, "status", status)
    if include_all:
        params.append(("include_all", "true"))
    try:
        response = client.get("/jobs", version_prefix=True, params=params)
    except HttpError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(json.dumps(response.json(), indent=2))


@app.command("info")
def job_info(
    job_id: str = typer.Argument(..., help="Job identifier"),
) -> None:
    """Retrieve the current status and metadata for a job."""
    client = client_from_config()
    try:
        response = client.get(f"/jobs/{job_id}", version_prefix=True)
    except HttpError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(json.dumps(response.json(), indent=2))


@app.command()
def progress(
    job_id: str = typer.Argument(..., help="Job identifier"),
) -> None:
    """Get detailed progress information for a job."""
    client = client_from_config()
    try:
        response = client.get(f"/jobs/{job_id}/progress", version_prefix=True)
    except HttpError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(json.dumps(response.json(), indent=2))


@app.command()
def result(
    job_id: str = typer.Argument(..., help="Job identifier"),
) -> None:
    """Retrieve the result of a completed job."""
    client = client_from_config()
    try:
        response = client.get(f"/jobs/{job_id}/result", version_prefix=True)
    except HttpError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(json.dumps(response.json(), indent=2))


@app.command()
def inputs(
    job_id: str = typer.Argument(..., help="Job identifier"),
) -> None:
    """Retrieve the inputs of a submitted job."""
    client = client_from_config()
    try:
        response = client.get(f"/jobs/{job_id}/inputs", version_prefix=True)
    except HttpError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(json.dumps(response.json(), indent=2))


@app.command()
def cancel(
    job_id: str = typer.Argument(..., help="Job identifier"),
) -> None:
    """Cancel a running job."""
    client = client_from_config()
    try:
        response = client.post(f"/jobs/{job_id}/cancel", version_prefix=True)
    except HttpError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(json.dumps(response.json(), indent=2))


def _poll_once(client: Any, job_id: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Fetch job status and progress. Returns (job, status, progress)."""
    job_resp = client.get(f"/jobs/{job_id}", version_prefix=True)
    job = _unwrap(job_resp.json())
    status = job.get("status", "")

    prog: dict[str, Any] = {}
    try:
        prog_resp = client.get(f"/jobs/{job_id}/progress", version_prefix=True)
        prog = _unwrap(prog_resp.json()).get("progress", {})
    except HttpError:
        pass
    return job, status, prog


def _format_progress_line(
    job_id: str, status: str, prog: dict[str, Any], elapsed: float
) -> str:
    """Build a single-line progress summary for non-live mode.

    Uses only ASCII characters — no special fonts needed.
    """
    parts = [f"[{job_id}] {status}"]
    batch_progress = prog.get("batch_progress") if isinstance(prog, dict) else None
    if isinstance(batch_progress, dict):
        overall = batch_progress.get("overall_progress")
        if isinstance(overall, dict):
            processed = overall.get("processed_runtime_nodes_raw")
            total_raw = overall.get("raw_nodes")
            eta = overall.get("eta_seconds")
            if processed is not None and total_raw is not None:
                pct = processed / total_raw * 100 if total_raw > 0 else 0
                width = 20
                filled = int(width * pct / 100)
                bar = "#" * filled + "-" * (width - filled)
                parts.append(f"[{bar}] {processed}/{total_raw} nodes ({pct:.0f}%)")
            if eta is not None:
                parts.append(f"ETA {eta:.0f}s")
        completed = batch_progress.get("completed", 0)
        total = batch_progress.get("total", 0)
        running = batch_progress.get("running", 0)
        if total > 0:
            parts.append(f"batches {completed}/{total} done, {running} running")
    parts.append(f"{elapsed:.0f}s")
    return "  ".join(parts)


def _build_progress_panel(job_id: str, progress: dict[str, Any]) -> Any:
    """Build a Rich Panel with a batch-level progress table for ``progress``.

    Renders one ``Overall`` row (raw-node progress + outcome counters) and
    one ``Running batches`` row (active batch progress + average elapsed
    time), wrapped in a header line that summarizes raw/FlowMesh node
    totals, ETA, and the done/running batch counts.
    """
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    def _int(value: Any) -> int:
        if isinstance(value, (int, float)):
            return max(0, int(value))
        return 0

    def _bar(completed: int, total: int) -> str:
        pct = min(100.0, max(0.0, completed / total * 100)) if total > 0 else 0.0
        width = 20
        filled = int(width * pct / 100)
        return f"{'#' * filled}{'-' * (width - filled)} {pct:>5.1f}%"

    if not isinstance(progress, dict) or "batch_progress" not in progress:
        return Panel(
            Text("Waiting for batch progress...", style="dim"),
            title=f"[bold cyan]Job {job_id}[/bold cyan]",
            border_style="blue",
        )

    bp = progress["batch_progress"]
    if not isinstance(bp, dict):
        return Panel(
            Text("Invalid progress data", style="red"),
            title=f"[bold cyan]Job {job_id}[/bold cyan]",
            border_style="red",
        )

    overall = bp.get("overall_progress", {})
    batches = bp.get("batches", [])

    pending_raw = _int(overall.get("pending_runtime_nodes_raw"))
    processing_raw = _int(overall.get("processing_runtime_nodes_raw"))
    processed_raw = _int(overall.get("processed_runtime_nodes_raw"))
    total_raw = pending_raw + processing_raw + processed_raw
    raw_nodes = _int(overall.get("raw_nodes"))
    fm_nodes = _int(overall.get("flowmesh_nodes"))

    o_succ = o_fail = o_disp = o_pend = 0
    r_succ = r_fail = r_disp = r_pend = r_total = r_count = 0
    r_elapsed_sum = 0.0
    r_elapsed_n = 0
    for b in batches:
        if not isinstance(b, dict):
            continue
        n = b.get("nodes", {})
        if not isinstance(n, dict):
            continue
        o_succ += _int(n.get("succeeded"))
        o_fail += _int(n.get("failed"))
        o_disp += _int(n.get("dispatched"))
        o_pend += _int(n.get("pending"))
        if b.get("status") == "RUNNING":
            r_count += 1
            r_succ += _int(n.get("succeeded"))
            r_fail += _int(n.get("failed"))
            r_disp += _int(n.get("dispatched"))
            r_pend += _int(n.get("pending"))
            r_total += _int(n.get("total"))
            elapsed = b.get("elapsed_time")
            if isinstance(elapsed, (int, float)):
                r_elapsed_sum += float(elapsed)
                r_elapsed_n += 1

    r_done = r_succ + r_fail
    r_time = f"{r_elapsed_sum / r_elapsed_n:.1f}s(avg)" if r_elapsed_n > 0 else "---"

    table = Table(
        title=f"Job {job_id} - Progress",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Scope", style="cyan", width=24)
    table.add_column("Progress", width=28)
    table.add_column("Raw Nodes", justify="right", width=14)
    table.add_column("S/F/R/P (succeed/fail/run/pend)", width=28)

    table.add_row(
        "Overall",
        _bar(processed_raw, total_raw),
        f"{processed_raw}/{total_raw}",
        f"S:{o_succ}  F:{o_fail}  R:{o_disp}  P:{o_pend}",
    )
    table.add_row(
        f"Running ({r_count} batches)",
        _bar(r_done, r_total),
        str(processing_raw),
        f"S:{r_succ}  F:{r_fail}  R:{r_disp}  P:{r_pend}; t={r_time}",
    )

    summary_parts = [f"Raw/FlowMesh nodes: {raw_nodes}/{fm_nodes}"]
    eta = bp.get("eta_seconds")
    if isinstance(eta, (int, float)):
        summary_parts.append(f"ETA: {eta:.1f}s")
    c_batches = int(bp.get("completed", 0))
    r_batches = int(bp.get("running", 0))
    summary_parts.append(f"Batches done/running: {c_batches}/{r_batches}")
    header = Text("; ".join(summary_parts), style="bold cyan")

    return Panel(
        Group(header, table),
        title="Batch Progress",
        border_style="blue",
        padding=(1, 2),
    )


@app.command()
def watch(
    job_id: str = typer.Argument(..., help="Job identifier"),
    interval: float = typer.Option(
        3.0, "--interval", "-i", help="Polling interval in seconds"
    ),
    timeout: float = typer.Option(
        180.0, "--timeout", "-t", help="Max seconds to wait before giving up"
    ),
    live: bool = typer.Option(
        False, "--live", "-l", help="Rich live progress table (for interactive use)"
    ),
) -> None:
    """Monitor a job by polling until completion.

    Default mode appends one line per status change (suitable for CI/logs).
    Use --live for a rich interactive progress table with batch details.
    """
    client = client_from_config()
    last_line: str | None = None
    start = time.time()

    from contextlib import AbstractContextManager, nullcontext

    live_ctx: AbstractContextManager[Any]
    if live:
        from rich.console import Console
        from rich.live import Live

        console = Console(stderr=True)
        live_ctx = Live(
            _build_progress_panel(job_id, {}),
            console=console,
            refresh_per_second=4,
        )
    else:
        live_ctx = nullcontext()

    try:
        with live_ctx as live_display:
            while True:
                try:
                    job, status, prog = _poll_once(client, job_id)
                except HttpError as exc:
                    logging.warning(f"Error fetching job: {exc}")
                    time.sleep(interval)
                    continue

                elapsed = time.time() - start

                if status == "cancelled":
                    logging.error(f"Job {job_id} was cancelled.")
                    logging.log(json.dumps(job, indent=2))
                    raise typer.Exit(code=1)

                if live_display is not None:
                    live_display.update(_build_progress_panel(job_id, prog))
                else:
                    line = _format_progress_line(job_id, status, prog, elapsed)
                    # Compare without the elapsed suffix so we print on
                    # status/progress changes, not just because time passed.
                    line_key = line.rsplit("  ", 1)[0]
                    if line_key != last_line:
                        logging.info(line)
                        last_line = line_key

                if status in {"completed", "failed"}:
                    if live_display is not None:
                        live_display.update(_build_progress_panel(job_id, prog))
                    logging.log(json.dumps(job, indent=2))
                    try:
                        result_resp = client.get(
                            f"/jobs/{job_id}/result", version_prefix=True
                        )
                        logging.log(json.dumps(_unwrap(result_resp.json()), indent=2))
                    except HttpError:
                        logging.warning("Could not fetch job result.")
                    if status == "failed":
                        raise typer.Exit(code=1)
                    return

                if elapsed > timeout:
                    logging.error(
                        f"Job {job_id} did not finish within {timeout}s. "
                        "It is still running — use a larger --timeout or check later."
                    )
                    raise typer.Exit(code=1)

                time.sleep(interval)
    except KeyboardInterrupt:
        logging.warning("Cancelled by user.")
        raise typer.Exit(code=1)


@app.command()
def artifact(
    job_id: str = typer.Argument(..., help="Job identifier"),
    path: str = typer.Option(..., "--path", help="Artifact path to download"),
    output: Path = typer.Option(..., "--output", "-o", help="Output file path"),
) -> None:
    """Download a job artifact."""
    client = client_from_config()
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        client.download(
            f"/jobs/{job_id}/artifact",
            output,
            version_prefix=True,
            params={"path": path},
        )
    except HttpError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    except OSError as exc:
        logging.error(f"Failed to write {output}: {exc}")
        raise typer.Exit(code=1)
    logging.success(f"Artifact saved to {output}")


@app.command()
def preview(
    workflow: Path = typer.Argument(..., help="Path to workflow file"),
    workflow_format: str = typer.Option(
        "n8n", "--format", "-f", help="Workflow format (native|n8n|yaml)"
    ),
    input_values: list[str] | None = typer.Option(
        None, "--input", help="Input as Name=v1,v2,v3 (repeatable for different names)"
    ),
    input_files: list[str] | None = typer.Option(
        None, "--input-file", help="Input from file as Name=path.txt (repeatable)"
    ),
    input_json: Path | None = typer.Option(
        None, "--input-json", help="JSON file with full inputs object"
    ),
    optimizer: str | None = typer.Option(
        None,
        "--optimizer",
        help="Override the server default optimizer (must be in /optimizer).",
    ),
) -> None:
    """Preview the optimization schedule for a workflow without executing it.

    Accepts the same payload shape as submit. The preview endpoint ignores
    output_location and priority.
    """
    if not workflow.exists():
        logging.error(f"Workflow file not found: {workflow}")
        raise typer.Exit(code=1)

    inputs = _build_inputs(input_values, input_files, input_json)
    if not inputs:
        logging.error(
            "At least one input is required. "
            "Use --input, --input-file, or --input-json."
        )
        raise typer.Exit(code=1)

    workflow_text = workflow.read_text()
    payload: dict[str, Any] = {
        "data": [{"workflow": workflow_text, "inputs": inputs}],
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer

    client = client_from_config()
    try:
        response = client.post(
            "/jobs/preview",
            version_prefix=True,
            json=payload,
            headers={"Workflow-Format": workflow_format},
        )
    except HttpError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(json.dumps(response.json(), indent=2))


def _render_workflows_table(job_id: str, workflows: list[dict[str, Any]]) -> Any:
    table = Table(
        title=f"Job {job_id} - FlowMesh Workflows",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Workflow ID", style="cyan", overflow="fold")
    table.add_column("Status", style="magenta")
    table.add_column("Submitted At", overflow="fold")
    table.add_column("Tasks", justify="right")
    table.add_column("Succeeded", justify="right")
    table.add_column("Failed", justify="right")
    for wf in workflows:
        table.add_row(
            str(wf.get("workflow_id", "")),
            str(wf.get("status", "")),
            str(wf.get("submitted_at") or ""),
            str(wf.get("task_count") if wf.get("task_count") is not None else ""),
            str(
                wf.get("succeeded_count")
                if wf.get("succeeded_count") is not None
                else ""
            ),
            str(wf.get("failed_count") if wf.get("failed_count") is not None else ""),
        )
    return table


@app.command("workflows")
def workflows(
    job_id: str = typer.Argument(..., help="Job identifier"),
    as_json: bool = typer.Option(
        False, "--json", help="Emit raw JSON instead of a table"
    ),
) -> None:
    """List FlowMesh workflows associated with a job."""
    client = client_from_config()
    try:
        response = client.get(f"/jobs/{job_id}/workflows", version_prefix=True)
    except HttpError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    payload = _unwrap(response.json())
    workflow_list = payload.get("workflows", []) if isinstance(payload, dict) else []
    if as_json:
        logging.log(json.dumps(payload, indent=2))
        return
    Console().print(_render_workflows_table(job_id, workflow_list))


def _format_log_line(entry: dict[str, Any]) -> str:
    event = entry.get("event", {}) if isinstance(entry, dict) else {}
    if not isinstance(event, dict):
        event = {}
    ts = event.get("ts") or ""
    level = event.get("level") or ""
    stream = event.get("stream") or ""
    message = event.get("message") or ""
    head = f"{ts} {level:<5}".strip()
    if stream:
        head = f"{head} [{stream}]"
    return f"{head} {message}".rstrip()


def _print_log_entries(entries: list[dict[str, Any]], as_json: bool) -> None:
    for entry in entries:
        if as_json:
            logging.log(json.dumps(entry))
        else:
            logging.log(_format_log_line(entry))


def _parse_sse_entries(raw: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for block in raw.split("\n\n"):
        current_event: str | None = None
        for line in block.splitlines():
            if line.startswith("event:"):
                current_event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data = line[len("data:") :].strip()
                if data:
                    try:
                        parsed = json.loads(data)
                        if current_event == "error" or (
                            isinstance(parsed, dict)
                            and parsed.get("kind") == "stream_error"
                        ):
                            parsed["_sse_error"] = True
                        entries.append(parsed)
                    except json.JSONDecodeError:
                        pass
    return entries


logs_app = get_typer(help="Fetch, stream, or download logs for a FlowMesh workflow.")


@logs_app.command("show")
def logs_show(
    job_id: str = typer.Argument(..., help="Job identifier"),
    workflow_id: str = typer.Argument(..., help="FlowMesh workflow identifier"),
    limit: int = typer.Option(200, "--limit", help="Maximum entries per page (1-1000)"),
    before: str | None = typer.Option(
        None, "--before", help="Cursor to fetch entries older than this point"
    ),
    after: str | None = typer.Option(
        None, "--after", help="Cursor to fetch entries newer than this point"
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit raw JSON for each entry"),
) -> None:
    """Fetch one page of paginated logs for a FlowMesh workflow."""
    if limit < 1 or limit > 1000:
        logging.error("--limit must be between 1 and 1000")
        raise typer.Exit(code=1)
    client = client_from_config()
    url = f"/jobs/{job_id}/workflows/{workflow_id}/logs"
    params: list[tuple[str, str]] = [("limit", str(limit))]
    if before is not None:
        params.append(("before", before))
    if after is not None:
        params.append(("after", after))
    try:
        resp = client.get(url, version_prefix=True, params=params)
    except HttpError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    payload = _unwrap(resp.json())
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    _print_log_entries(entries, as_json)


@logs_app.command("stream")
def logs_stream(
    job_id: str = typer.Argument(..., help="Job identifier"),
    workflow_id: str = typer.Argument(..., help="FlowMesh workflow identifier"),
    cursor: str | None = typer.Option(
        None, "--cursor", help="Resume streaming from this cursor position"
    ),
) -> None:
    """Stream live log entries for a FlowMesh workflow via SSE."""
    client = client_from_config()
    stream_url = f"/jobs/{job_id}/workflows/{workflow_id}/logs/stream"
    stream_params: list[tuple[str, str]] = []
    if cursor is not None:
        stream_params.append(("cursor", cursor))
    try:
        base = client.base_url.rstrip("/") + "/api/v1"
        headers = {"Accept": "text/event-stream"}
        if client.api_key:
            headers["Authorization"] = f"Bearer {client.api_key}"
        with requests.get(
            base + stream_url,
            headers=headers,
            params=stream_params or None,
            stream=True,
            timeout=client.timeout,
        ) as resp:
            if resp.status_code >= 400:
                logging.error(f"Stream error: {resp.status_code} {resp.text}")
                raise typer.Exit(code=1)
            buffer = ""
            stream_error: dict[str, Any] | None = None
            for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    buffer += chunk
                    while "\n\n" in buffer:
                        block, buffer = buffer.split("\n\n", 1)
                        for entry in _parse_sse_entries(block + "\n\n"):
                            if entry.get("_sse_error"):
                                stream_error = entry
                                break
                            _print_log_entries([entry], False)
                        if stream_error is not None:
                            break
                if stream_error is not None:
                    break
            if stream_error is not None:
                code = stream_error.get("code", "APIError")
                message = stream_error.get("message", "Upstream stream error.")
                logging.error(f"log stream error [{code}]: {message}")
                raise typer.Exit(code=1)
    except KeyboardInterrupt:
        raise typer.Exit(code=0)


@logs_app.command("download")
def logs_download(
    job_id: str = typer.Argument(..., help="Job identifier"),
    workflow_id: str = typer.Argument(..., help="FlowMesh workflow identifier"),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Directory to extract logs into"
    ),
) -> None:
    """Download per-task archived logs for a FlowMesh workflow."""
    output.mkdir(parents=True, exist_ok=True)
    client = client_from_config()
    try:
        client.download(
            f"/jobs/{job_id}/workflows/{workflow_id}/logs/download",
            output / f"{workflow_id}-logs.tar",
            version_prefix=True,
        )
    except HttpError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    except OSError as exc:
        logging.error(f"Failed to write archive: {exc}")
        raise typer.Exit(code=1)

    archive_path = output / f"{workflow_id}-logs.tar"
    extracted: list[Path] = []
    try:
        with tarfile.open(archive_path, "r") as tf:
            members = tf.getmembers()
            if not members:
                logging.log(f"No logs downloaded for workflow {workflow_id}.")
                archive_path.unlink(missing_ok=True)
                return
            tf.extractall(output, filter="data")
            for member in members:
                extracted.append(output / member.name)
    except tarfile.TarError as exc:
        logging.error(f"Failed to extract log archive: {exc}")
        archive_path.unlink(missing_ok=True)
        raise typer.Exit(code=1)

    archive_path.unlink(missing_ok=True)
    for path in extracted:
        logging.log(str(path))


app.add_typer(logs_app, name="logs")
