#!/usr/bin/env python3
"""Export the e2e demo dataset to a release-uploadable bundle."""

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from lumilake_deploy._demo_data import (
    compose_key_prefix,
    find_default_env_file,
    human_bytes,
    info,
    make_minio_client,
    parse_s3_url,
    require_env,
    resolve_env,
)

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2].parent / "demo-data-bundle"
DEFAULT_SCHEMA = "lumilake_demo"
DEFAULT_S3_PREFIX = "example-data"
NEWS_SUBDIRS = ("html", "images")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-file", type=Path)
    p.add_argument("--database-url")
    p.add_argument("--s3-url")
    p.add_argument("--s3-data-prefix")
    p.add_argument("--s3-cert-file")
    p.add_argument("--s3-prefix", default=DEFAULT_S3_PREFIX)
    p.add_argument("--schema", default=DEFAULT_SCHEMA)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def pg_dump(database_url: str, schema: str, out_path: Path) -> None:
    if shutil.which("pg_dump") is None:
        raise SystemExit("pg_dump not on PATH. Install postgresql-client.")
    cmd = [
        "pg_dump",
        "-Fc",
        "--no-owner",
        "--no-acl",
        f"--schema={schema}",
        "-f",
        str(out_path),
        database_url,
    ]
    info(f"[1/2] pg_dump --schema={schema} -> {out_path}")
    subprocess.run(cmd, check=True)


def download_news_tree(client: Any, bucket: str, key_prefix: str, dest: Path) -> int:
    base = key_prefix.rstrip("/")
    scan_prefix = f"{base}/news/" if base else "news/"
    count = 0
    for sub in NEWS_SUBDIRS:
        (dest / "news" / sub).mkdir(parents=True, exist_ok=True)
    for obj in client.list_objects(bucket, prefix=scan_prefix, recursive=True):
        if obj.is_dir or not obj.object_name:
            continue
        rel = obj.object_name[len(scan_prefix.rstrip("/")) + 1 :]
        target = dest / "news" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        client.fget_object(bucket, obj.object_name, str(target))
        count += 1
        if count % 200 == 0:
            info(f"       downloaded {count} objects...")
    return count


def make_tarball(src_root: Path, out_path: Path) -> None:
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(src_root / "news", arcname="news")


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    env_file = args.env_file or find_default_env_file()
    env = resolve_env(
        env_file,
        overrides={
            "DATABASE_URL": args.database_url,
            "S3_URL": args.s3_url,
            "S3_DATA_PREFIX": args.s3_data_prefix,
            "S3_CERT_FILE": args.s3_cert_file,
        },
    )
    require_env(env, ["DATABASE_URL", "S3_URL", "S3_DATA_PREFIX"])

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pg_path = out_dir / "lumilake-demo-pg.dump"
    s3_path = out_dir / "lumilake-demo-s3.tar.gz"

    if (pg_path.exists() or s3_path.exists()) and not args.force:
        raise SystemExit(
            f"Artifacts already exist in {out_dir}. Pass --force to overwrite."
        )
    pg_path.unlink(missing_ok=True)
    s3_path.unlink(missing_ok=True)

    pg_dump(env["DATABASE_URL"], args.schema, pg_path)
    info(f"       wrote {pg_path} ({human_bytes(pg_path.stat().st_size)})")

    cfg = parse_s3_url(
        env["S3_URL"], env["S3_DATA_PREFIX"], env.get("S3_CERT_FILE") or None
    )
    client = make_minio_client(cfg)

    full_key_prefix = compose_key_prefix(cfg.base_prefix, args.s3_prefix)
    info(f"[2/2] bundling s3://{cfg.bucket}/{full_key_prefix}/news -> {s3_path}")
    with tempfile.TemporaryDirectory(prefix="lumilake-demo-news-") as stage:
        stage_path = Path(stage)
        count = download_news_tree(client, cfg.bucket, full_key_prefix, stage_path)
        info(f"       downloaded {count} objects")
        make_tarball(stage_path, s3_path)
    info(f"       wrote {s3_path} ({human_bytes(s3_path.stat().st_size)})")

    info("")
    info("Done. Next:")
    info("  gh release create demo-data-v1 \\")
    info(f"    {pg_path} \\")
    info(f"    {s3_path} \\")
    info("    --notes 'Demo dataset for Lumilake e2e workflows.'")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
