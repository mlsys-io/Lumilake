"""SSH resource operations."""

import builtins
from typing import Any

from ..models.ssh import SSHConnectionInfo
from ..ssh import (
    build_ssh_task_yaml,
    detect_public_key,
    ssh_connection_commands,
    ssh_proxy_url,
)
from ._base import AsyncResource, SyncResource


class SSH(SyncResource):
    """Synchronous SSH operations."""

    def list(
        self,
        query_params: list[tuple[str, str]] | None = None,
    ) -> builtins.list[SSHConnectionInfo]:
        """List active SSH connections."""
        data = self._client._request(
            "GET", "/ssh/connections", params=query_params or None
        )
        return [SSHConnectionInfo.model_validate(c) for c in data]

    def build_task_yaml(
        self,
        name: str,
        public_key: str | None,
        user: str,
        mode: str,
        ttl: int,
        idle_timeout: int,
        gpu: int | None,
        gpu_memory: str | None,
        cpu: int | None,
        memory: str | None,
        image: str | None,
        worker: str | None,
        env_pairs: builtins.list[str] | None,
        interactive: bool | None = None,
        command: builtins.list[str] | None = None,
        entrypoint: builtins.list[str] | None = None,
    ) -> str:
        """Build an SSH task definition."""
        return build_ssh_task_yaml(
            name=name,
            public_key=public_key,
            user=user,
            mode=mode,
            ttl=ttl,
            idle_timeout=idle_timeout,
            gpu=gpu,
            gpu_memory=gpu_memory,
            cpu=cpu,
            memory=memory,
            image=image,
            worker=worker,
            env_pairs=env_pairs,
            interactive=interactive,
            command=command,
            entrypoint=entrypoint,
        )

    def detect_public_key(self) -> str:
        """Detect a local SSH public key from ``~/.ssh``."""
        return detect_public_key()

    def proxy_url(self, task_id: str) -> str:
        """Build the websocket proxy URL for a task."""
        return ssh_proxy_url(self._client.base_url, task_id)

    def connection_commands(
        self,
        task_id: str,
        ssh_info: dict[str, Any],
    ) -> builtins.list[tuple[str, str]]:
        """Build suggested connection commands for an SSH task."""
        return ssh_connection_commands(
            task_id, ssh_info, base_url=self._client.base_url
        )


class AsyncSSH(AsyncResource):
    """Asynchronous SSH operations."""

    async def list(
        self,
        query_params: list[tuple[str, str]] | None = None,
    ) -> builtins.list[SSHConnectionInfo]:
        """List active SSH connections."""
        data = await self._client._request(
            "GET", "/ssh/connections", params=query_params or None
        )
        return [SSHConnectionInfo.model_validate(c) for c in data]

    async def build_task_yaml(
        self,
        name: str,
        public_key: str | None,
        user: str,
        mode: str,
        ttl: int,
        idle_timeout: int,
        gpu: int | None,
        gpu_memory: str | None,
        cpu: int | None,
        memory: str | None,
        image: str | None,
        worker: str | None,
        env_pairs: builtins.list[str] | None,
        interactive: bool | None = None,
        command: builtins.list[str] | None = None,
        entrypoint: builtins.list[str] | None = None,
    ) -> str:
        """Build an SSH task definition."""
        return build_ssh_task_yaml(
            name=name,
            public_key=public_key,
            user=user,
            mode=mode,
            ttl=ttl,
            idle_timeout=idle_timeout,
            gpu=gpu,
            gpu_memory=gpu_memory,
            cpu=cpu,
            memory=memory,
            image=image,
            worker=worker,
            env_pairs=env_pairs,
            interactive=interactive,
            command=command,
            entrypoint=entrypoint,
        )

    async def detect_public_key(self) -> str:
        """Detect a local SSH public key from ``~/.ssh``."""
        return detect_public_key()

    async def proxy_url(self, task_id: str) -> str:
        """Build the websocket proxy URL for a task."""
        return ssh_proxy_url(self._client.base_url, task_id)

    async def connection_commands(
        self,
        task_id: str,
        ssh_info: dict[str, Any],
    ) -> builtins.list[tuple[str, str]]:
        """Build suggested connection commands for an SSH task."""
        return ssh_connection_commands(
            task_id, ssh_info, base_url=self._client.base_url
        )
