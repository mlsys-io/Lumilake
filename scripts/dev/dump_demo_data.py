#!/usr/bin/env python3
"""Export the e2e demo dataset to a release-uploadable bundle."""

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from lumilake_deploy._demo_data import (
    LumidBlobClient,
    compose_key_prefix,
    find_default_env_file,
    human_bytes,
    info,
    lumid_config_from_env,
    require_env,
    resolve_env,
)

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2].parent / "demo-data-bundle"
DEFAULT_SCHEMA = "lumilake_demo"
DEFAULT_S3_PREFIX = "example-data"


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-file", type=Path)
    p.add_argument("--database-url")
    p.add_argument("--lumid-data-url", help="lumid-data-app base URL")
    p.add_argument("--lumid-data-token", help="lumid-data-app bearer token")
    p.add_argument(
        "--blob-prefix",
        help="logical prefix inside lumid-data-app's blob store "
        "(matches the Lumilake server's S3_DATA_PREFIX)",
    )
    p.add_argument("--news-key-prefix", default=DEFAULT_S3_PREFIX)
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


def download_news_tree(client: LumidBlobClient, key_prefix: str, dest: Path) -> int:
    base = key_prefix.strip("/")
    scan_prefix = f"{base}/news" if base else "news"
    strip_len = len(scan_prefix) + 1  # include trailing '/'
    count = 0
    for key in client.iter_blob_keys(f"{scan_prefix}/"):
        if not key.startswith(f"{scan_prefix}/"):
            continue
        # Tarball uses ``news/<rel>`` paths — strip the absolute prefix.
        rel = key[strip_len:]
        target = dest / "news" / rel
        client.download_blob(key, target)
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
            "LUMID_DATA_URL": args.lumid_data_url,
            "LUMID_DATA_TOKEN": args.lumid_data_token,
            "S3_DATA_PREFIX": args.blob_prefix,
        },
    )
    require_env(env, ["DATABASE_URL", "S3_DATA_PREFIX"])
    lumid_cfg = lumid_config_from_env(env)

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

    client = LumidBlobClient(lumid_cfg)
    full_key_prefix = compose_key_prefix(env["S3_DATA_PREFIX"], args.news_key_prefix)
    info(
        f"[2/2] bundling {lumid_cfg.base_url}/blobs/{full_key_prefix}/news -> "
        f"{s3_path}"
    )
    with tempfile.TemporaryDirectory(prefix="lumilake-demo-news-") as stage:
        stage_path = Path(stage)
        count = download_news_tree(client, full_key_prefix, stage_path)
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
