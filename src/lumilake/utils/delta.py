"""Delta Lake I/O helpers for use inside LambdaOps.

Delta Lake (backed by Parquet + a transaction log) gives us:
- Atomic commits — no half-written table state on worker failure.
- Version history — every write produces a new version; readers can
  time-travel to any historic version via the ``version`` argument.
- Schema evolution — callers can add columns over time without rewriting
  existing table history.

Importing this module requires ``deltalake`` and ``pyarrow``; both ship
in the ``delta`` and ``test`` dependency groups. Callers are expected to
install those before pulling in ``lumilake.utils.delta``.

All functions accept either local paths (``./foo``, ``file:///tmp/foo``)
or S3-style URIs (``s3://bucket/key``). S3 credentials/endpoints are
passed via keyword args; when omitted, ``deltalake`` falls back to the
default AWS credential chain.
"""

from typing import Any, Literal

import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

WriteMode = Literal["append", "overwrite", "error", "ignore"]


def _build_storage_options(
    *,
    endpoint: str | None,
    access_key: str | None,
    secret_key: str | None,
    region: str,
    use_ssl: bool,
    allow_unsafe_rename: bool,
) -> dict[str, str] | None:
    """Assemble ``object_store``-style storage options for deltalake.

    Returns ``None`` when no overrides are supplied so the default AWS
    credential chain is used unchanged.

    ``allow_unsafe_rename`` is opt-in (default ``False``). Deltalake
    normally refuses concurrent S3 writes without a lock provider; the
    flag disables that safety check and is safe **only** for setups
    with serialized writers (for example, local MinIO development).
    Enabling it against real AWS with concurrent writers
    can corrupt the table.
    """
    opts: dict[str, str] = {}
    if endpoint:
        opts["AWS_ENDPOINT_URL"] = endpoint
        # MinIO-style path-style addressing; always safe to set for S3-compat.
        opts["AWS_ALLOW_HTTP"] = "true" if not use_ssl else "false"
        opts["AWS_S3_ADDRESSING_STYLE"] = "path"
    if access_key is not None:
        opts["AWS_ACCESS_KEY_ID"] = access_key
    if secret_key is not None:
        opts["AWS_SECRET_ACCESS_KEY"] = secret_key
    if region:
        opts["AWS_REGION"] = region
    if allow_unsafe_rename and (endpoint or access_key is not None):
        opts["AWS_S3_ALLOW_UNSAFE_RENAME"] = "true"
    return opts or None


def _coerce_to_arrow(data: pa.Table | Any) -> pa.Table:
    """Convert pandas/records into pyarrow; passthrough for Table."""
    if isinstance(data, pa.Table):
        return data
    # Avoid hard pandas dep at import time; only require if caller passes one.
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return pa.Table.from_pandas(data, preserve_index=False)
    except ImportError:  # pragma: no cover
        pass
    if isinstance(data, list):
        return pa.Table.from_pylist(data)
    raise TypeError(
        f"Unsupported data type for write_delta: {type(data).__name__}. "
        "Expected pyarrow.Table, pandas.DataFrame, or list[dict]."
    )


def read_delta(
    path: str,
    *,
    version: int | None = None,
    columns: list[str] | None = None,
    endpoint: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    region: str = "us-east-1",
    use_ssl: bool = False,
) -> pa.Table:
    """Read a Delta Lake table into a pyarrow Table.

    Args:
        path: Local path or ``s3://`` URI to the Delta table root.
        version: Time-travel to a specific commit version. ``None`` reads
            the latest.
        columns: Column projection; ``None`` reads all columns.
        endpoint / access_key / secret_key / region / use_ssl: S3-only.
            Ignored when ``path`` is local.
    """
    storage_options = _build_storage_options(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        region=region,
        use_ssl=use_ssl,
        allow_unsafe_rename=False,
    )
    table = DeltaTable(path, version=version, storage_options=storage_options)
    return table.to_pyarrow_table(columns=columns)


def write_delta(
    data: pa.Table | Any,
    path: str,
    *,
    mode: WriteMode = "append",
    partition_by: list[str] | None = None,
    endpoint: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    region: str = "us-east-1",
    use_ssl: bool = False,
    allow_unsafe_rename: bool = False,
) -> None:
    """Write data to a Delta Lake table.

    Args:
        data: ``pyarrow.Table``, ``pandas.DataFrame``, or ``list[dict]``.
        path: Local path or ``s3://`` URI for the Delta table root.
        mode: ``"append"`` (default), ``"overwrite"``, ``"error"``, or
            ``"ignore"``. Each non-error call produces a new table version.
        partition_by: Optional list of column names to partition by.
        endpoint / access_key / secret_key / region / use_ssl: S3-only.
        allow_unsafe_rename: opt-in for setups with serialized writers
            (for example, local MinIO development). Sets
            ``AWS_S3_ALLOW_UNSAFE_RENAME`` which disables deltalake's
            concurrency-safety fence; only safe when you can guarantee
            no concurrent writers to the same table. Default ``False``.
    """
    arrow_table = _coerce_to_arrow(data)
    storage_options = _build_storage_options(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        region=region,
        use_ssl=use_ssl,
        allow_unsafe_rename=allow_unsafe_rename,
    )
    write_deltalake(
        path,
        arrow_table,
        mode=mode,
        partition_by=partition_by,
        storage_options=storage_options,
    )


def list_delta_versions(
    path: str,
    *,
    limit: int | None = None,
    endpoint: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    region: str = "us-east-1",
    use_ssl: bool = False,
) -> list[dict[str, Any]]:
    """Return the commit history for a Delta table, newest first.

    Each entry includes ``version``, ``timestamp``, ``operation``, and
    the standard deltalake history metadata. Useful for comparing
    writes against historical table snapshots.
    """
    storage_options = _build_storage_options(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        region=region,
        use_ssl=use_ssl,
        allow_unsafe_rename=False,
    )
    table = DeltaTable(path, storage_options=storage_options)
    return table.history(limit=limit)
