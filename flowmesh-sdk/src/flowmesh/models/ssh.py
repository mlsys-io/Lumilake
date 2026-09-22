"""SSH-related models."""

from pydantic import BaseModel


class SSHConnectionInfo(BaseModel):
    connection_id: str
    access_mode: str
    task_id: str
    workflow_id: str | None = None
    worker_id: str | None = None
    node_id: str | None = None
    session_id: str | None = None
    username: str | None = None
    source_ip: str | None = None
    source_port: int | None = None
    connected_at: str
