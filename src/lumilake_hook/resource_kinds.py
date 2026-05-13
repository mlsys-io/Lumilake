from enum import StrEnum


class ResourceKind(StrEnum):
    JOB = "job"
    ARTIFACT = "artifact"
    TRACE = "trace"
    TABLE = "table"
    OBJECT_PREFIX = "object-prefix"
    WORKER = "worker"
    SYSTEM = "system"


class ResourceAction(StrEnum):
    READ = "read"
    WRITE = "write"
    CANCEL = "cancel"
    ADMIN = "admin"
