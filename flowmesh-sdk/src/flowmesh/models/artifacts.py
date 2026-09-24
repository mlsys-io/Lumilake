from pydantic import BaseModel, Field


class ArtifactRef(BaseModel):
    path: str


class ArtifactContext(BaseModel):
    base_dir: str
    base_url: str | None = Field(default=None, exclude_if=lambda v: v is None)
