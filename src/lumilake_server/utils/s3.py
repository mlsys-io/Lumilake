import certifi
import urllib3
from minio import Minio


def build_minio_http_client(
    *,
    cert_file: str | None,
    timeout_seconds: float | None = None,
) -> urllib3.PoolManager:
    timeout = (
        urllib3.Timeout(connect=timeout_seconds, read=timeout_seconds)
        if timeout_seconds is not None
        else None
    )
    if cert_file:
        return urllib3.PoolManager(
            cert_reqs="CERT_REQUIRED",
            ca_certs=cert_file if cert_file else certifi.where(),
            timeout=timeout,
            retries=False,
        )
    return urllib3.PoolManager(timeout=timeout, retries=False)


def create_minio_client(
    *,
    endpoint: str,
    access_key: str,
    secret_key: str,
    cert_file: str | None = None,
    secure: bool | None = None,
    timeout_seconds: float | None = None,
) -> Minio:
    resolved_secure = bool(cert_file) if secure is None else secure
    http_client = build_minio_http_client(
        cert_file=cert_file,
        timeout_seconds=timeout_seconds,
    )
    return Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=resolved_secure,
        http_client=http_client,
    )
