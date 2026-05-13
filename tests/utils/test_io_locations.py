from dataclasses import dataclass
from typing import Any

from lumilake.schemas.io import S3Location
from lumilake.utils import io_locations


class _FakeS3Error(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self.closed = False
        self.released = False

    def read(self) -> bytes:
        return self._data

    def close(self) -> None:
        self.closed = True

    def release_conn(self) -> None:
        self.released = True


@dataclass
class _FakeObject:
    object_name: str


class _FakeMinio:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.put_calls: list[dict[str, Any]] = []

    def put_object(
        self,
        *,
        bucket_name: str,
        object_name: str,
        data: Any,
        length: int,
        content_type: str,
    ) -> None:
        body = data.read()
        assert len(body) == length
        self.objects[(bucket_name, object_name)] = body
        self.put_calls.append(
            {
                "bucket": bucket_name,
                "object_name": object_name,
                "content_type": content_type,
            }
        )

    def get_object(self, bucket_name: str, object_name: str) -> _FakeResponse:
        try:
            data = self.objects[(bucket_name, object_name)]
        except KeyError as exc:
            raise _FakeS3Error("NoSuchKey") from exc
        return _FakeResponse(data)

    def list_objects(
        self, bucket_name: str, *, prefix: str, recursive: bool
    ) -> list[_FakeObject]:
        assert recursive is True
        return [
            _FakeObject(object_name)
            for (bucket, object_name), _body in sorted(self.objects.items())
            if bucket == bucket_name and object_name.startswith(prefix)
        ]


def test_sharded_index_helpers_roundtrip_and_preserve_relative_paths(
    monkeypatch,
) -> None:
    fake = _FakeMinio()

    def fake_s3_client_for_uri(uri: str) -> tuple[_FakeMinio, str, str]:
        bucket, obj = io_locations._split_bucket_object(uri)
        return fake, bucket, obj

    monkeypatch.setattr(io_locations, "_s3_client_for_uri", fake_s3_client_for_uri)
    monkeypatch.setattr(io_locations, "S3Error", _FakeS3Error)

    location = S3Location(
        type="s3",
        prefix="graphs/run-1",
        connection_string="s3://user:pass@minio:9000/test-bucket/base-prefix",
    )
    io_locations.write_sharded_index(
        location,
        {
            "manifest.json": '{"version": 1}',
            "chunks/part-000.parquet": b"PAR1",
            "meta/custom.bin": b"\x00\x01",
        },
        content_types={"meta/custom.bin": "application/x-custom"},
    )

    assert fake.put_calls == [
        {
            "bucket": "test-bucket",
            "object_name": "base-prefix/graphs/run-1/manifest.json",
            "content_type": "application/json",
        },
        {
            "bucket": "test-bucket",
            "object_name": "base-prefix/graphs/run-1/chunks/part-000.parquet",
            "content_type": "application/vnd.apache.parquet",
        },
        {
            "bucket": "test-bucket",
            "object_name": "base-prefix/graphs/run-1/meta/custom.bin",
            "content_type": "application/x-custom",
        },
    ]
    assert io_locations.read_s3_json(location, "manifest.json") == {"version": 1}
    assert io_locations.read_s3_bytes(location, "chunks/part-000.parquet") == b"PAR1"
    assert io_locations.read_s3_bytes(location, "missing.json") is None
    assert io_locations.read_s3_json(location, "missing.json") is None
    assert io_locations.list_sharded_index(location) == [
        "chunks/part-000.parquet",
        "manifest.json",
        "meta/custom.bin",
    ]
    assert io_locations.list_sharded_index(location, "chunks") == [
        "chunks/part-000.parquet"
    ]


def test_write_sharded_index_requires_non_empty_prefix() -> None:
    location = S3Location(
        type="s3",
        prefix="",
        connection_string="s3://user:pass@minio:9000/test-bucket/base-prefix",
    )

    try:
        io_locations.write_sharded_index(location, {"manifest.json": "{}"})
    except ValueError as exc:
        assert str(exc) == "write_sharded_index requires a non-empty S3 prefix"
    else:  # pragma: no cover
        raise AssertionError("expected write_sharded_index to reject empty prefixes")
