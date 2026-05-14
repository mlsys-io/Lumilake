from typing import Literal

from pydantic import BaseModel, Field


class DBLocation(BaseModel):
    type: Literal["db"] = Field(
        description="Location type. Always `db` for database table/column targets."
    )
    table: str = Field(description="Table name (optionally schema-qualified).")
    column: str = Field(description="Column name.")


class S3Location(BaseModel):
    type: Literal["s3"] = Field(
        description="Location type. Always `s3` for object storage prefixes."
    )
    prefix: str = Field(
        description=(
            "S3 object key prefix. Resolved against the configured S3_URL. "
            "Use a trailing slash to denote a folder prefix."
        )
    )
    connection_string: str | None = Field(
        default=None,
        description=(
            "Optional full S3 connection string (s3://user:pass@endpoint:port/bucket). "
            "If provided, credentials are used when Flowmesh fetches files."
        ),
    )


IOLocation = DBLocation | S3Location
