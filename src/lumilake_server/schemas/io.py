from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DBLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["db"] = Field(
        description="Location type. Always `db` for database table/column targets."
    )
    table: str = Field(description="Table name (optionally schema-qualified).")
    column: str = Field(description="Column name.")


class S3Location(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["s3"] = Field(
        description="Location type. Always `s3` for object storage prefixes."
    )
    prefix: str = Field(
        description=(
            "S3 object key prefix used as a logical key prefix in lumid-data-app's "
            "store. Use a trailing slash to denote a folder prefix."
        )
    )


IOLocation = DBLocation | S3Location
