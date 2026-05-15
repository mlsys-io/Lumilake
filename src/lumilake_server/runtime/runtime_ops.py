from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RuntimeOpSchema(BaseModel):
    """Pydantic schema for a serialized ``RuntimeOp``.

    The dataclass below stays the in-process representation. This model
    is used at boundaries (HTTP, IPC) to enforce field shape, types,
    required fields, and reject unknown keys.
    """

    model_config = ConfigDict(extra="forbid")

    task_type: str
    backend: str
    model: str
    data_spec: dict[str, Any] = Field(default_factory=dict)
    model_spec: dict[str, Any] = Field(default_factory=dict)
    inference_spec: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    output_spec: dict[str, Any] | None = None
    condition: dict[str, str] | None = None


@dataclass(frozen=True)
class RuntimeOp:
    node_id: str
    task_type: str
    backend: str
    model: str
    data_spec: dict[str, Any]
    model_spec: dict[str, Any]
    inference_spec: dict[str, Any]
    dependencies: tuple[str, ...] = ()
    output_spec: dict[str, Any] | None = None
    condition: dict[str, str] | None = None

    @property
    def model_key(self) -> tuple[str, str, str]:
        return (self.task_type, self.backend, self.model)

    def serialize(self) -> dict[str, Any]:
        return RuntimeOpSchema(
            task_type=self.task_type,
            backend=self.backend,
            model=self.model,
            data_spec=self.data_spec,
            model_spec=self.model_spec,
            inference_spec=self.inference_spec,
            dependencies=list(self.dependencies),
            output_spec=self.output_spec,
            condition=self.condition,
        ).model_dump(exclude_none=True)

    @classmethod
    def from_schema(cls, node_id: str, schema: RuntimeOpSchema) -> "RuntimeOp":
        return cls(
            node_id=node_id,
            task_type=schema.task_type,
            backend=schema.backend,
            model=schema.model,
            data_spec=schema.data_spec,
            model_spec=schema.model_spec,
            inference_spec=schema.inference_spec,
            dependencies=tuple(schema.dependencies),
            output_spec=schema.output_spec,
            condition=schema.condition,
        )

    @classmethod
    def deserialize(cls, node_id: str, payload: Mapping[str, Any]) -> "RuntimeOp":
        return cls.from_schema(node_id, RuntimeOpSchema.model_validate(payload))

    def to_flowmesh_node(self) -> dict[str, Any]:
        spec_payload: dict[str, Any] = {
            "taskType": self.task_type,
            "data": self.data_spec,
        }
        if self.task_type in {
            "inference",
            "diffusion",
            "embedding",
            "omni_text2image",
        }:
            spec_payload["model"] = self.model_spec
        if self.task_type in {"inference", "diffusion", "omni_text2image"}:
            spec_payload["inference"] = self.inference_spec
        spec: dict[str, Any] = {
            "name": self.node_id,
            "spec": spec_payload,
        }
        # FlowMesh requires condition.node in dependsOn — it gates on the
        # task's result.
        deps: list[str] = [*self.dependencies]
        if self.condition is not None and self.condition["node"] not in deps:
            deps.append(self.condition["node"])
        if deps:
            spec["dependsOn"] = deps
        if self.output_spec:
            spec["spec"]["output"] = self.output_spec
        if self.condition:
            spec["spec"]["condition"] = self.condition
        return spec
