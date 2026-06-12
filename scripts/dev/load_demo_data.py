#!/usr/bin/env python3
"""Fetch the Lumilake e2e demo dataset and load it into the local stack."""

import argparse
import mimetypes
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import urllib3
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

REPO = "mlsys-io/lumilake_OSS"
DEFAULT_TAG = "demo-data-v1"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "lumilake-demo"
EMBEDDED_SCHEMA = "lumilake_demo"
DEFAULT_S3_PREFIX = "example-data"
PG_ASSET = "lumilake-demo-pg.dump"
S3_ASSET = "lumilake-demo-s3.tar.gz"


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
    p.add_argument("--tag", default=DEFAULT_TAG)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--news-key-prefix", default=DEFAULT_S3_PREFIX)
    p.add_argument("--drop-schema", action="store_true")
    p.add_argument("--pg-restore-jobs", type=int, default=4)
    return p.parse_args(argv)


def gh_download(tag: str, asset: str, dest: Path) -> None:
    if dest.is_file():
        info(f"       cached: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("gh") is not None:
        subprocess.run(
            [
                "gh",
                "release",
                "download",
                tag,
                "-R",
                REPO,
                "--pattern",
                asset,
                "--output",
                str(dest),
            ],
            check=True,
        )
        return
    url = f"https://github.com/{REPO}/releases/download/{tag}/{asset}"
    info(f"       gh not found; downloading via https from {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    http = urllib3.PoolManager()
    resp = http.request("GET", url, preload_content=False, redirect=True)
    try:
        if resp.status >= 400:
            raise SystemExit(f"download failed: HTTP {resp.status} for {url}")
        with open(tmp, "wb") as f:
            for chunk in resp.stream(64 * 1024):
                f.write(chunk)
    finally:
        resp.release_conn()
    tmp.replace(dest)


def pg_restore(database_url: str, dump_path: Path, drop: bool, jobs: int) -> None:
    if shutil.which("pg_restore") is None:
        raise SystemExit("pg_restore not on PATH. Install postgresql-client.")
    if drop:
        if shutil.which("psql") is None:
            raise SystemExit("psql not on PATH; cannot --drop-schema.")
        info(f"       dropping schema {EMBEDDED_SCHEMA} (cascade)")
        subprocess.run(
            [
                "psql",
                database_url,
                "-c",
                f"DROP SCHEMA IF EXISTS {EMBEDDED_SCHEMA} CASCADE;",
            ],
            check=True,
        )
    subprocess.run(
        [
            "pg_restore",
            "-j",
            str(jobs),
            "--no-owner",
            "--no-acl",
            "--dbname",
            database_url,
            str(dump_path),
        ],
        check=True,
    )


def _guess_content_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def upload_news_tree(client: LumidBlobClient, key_prefix: str, src_root: Path) -> int:
    base = key_prefix.strip("/")
    upload_base = f"{base}/news" if base else "news"
    count = 0
    news_root = src_root / "news"
    for path in news_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(news_root).as_posix()
        key = f"{upload_base}/{rel}"
        client.put_blob(key, path.read_bytes(), _guess_content_type(path))
        count += 1
        if count % 200 == 0:
            info(f"       uploaded {count} objects...")
    return count


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

    cache = args.cache_dir.expanduser().resolve()
    pg_path = cache / PG_ASSET
    s3_path = cache / S3_ASSET

    info(f"[1/3] fetching {args.tag} from {REPO} -> {cache}")
    gh_download(args.tag, PG_ASSET, pg_path)
    gh_download(args.tag, S3_ASSET, s3_path)
    info(f"       pg dump  {human_bytes(pg_path.stat().st_size)}")
    info(f"       blob tar {human_bytes(s3_path.stat().st_size)}")

    info(f"[2/3] restoring schema {EMBEDDED_SCHEMA} into DATABASE_URL")
    pg_restore(env["DATABASE_URL"], pg_path, args.drop_schema, args.pg_restore_jobs)

    client = LumidBlobClient(lumid_cfg)
    full_key_prefix = compose_key_prefix(env["S3_DATA_PREFIX"], args.news_key_prefix)
    info(f"[3/3] uploading news/ -> {lumid_cfg.base_url}/blobs/{full_key_prefix}/news")
    with tempfile.TemporaryDirectory(prefix="lumilake-demo-news-") as stage:
        stage_path = Path(stage)
        with tarfile.open(s3_path, "r:gz") as tar:
            tar.extractall(stage_path, filter="data")
        count = upload_news_tree(client, full_key_prefix, stage_path)
    info(f"       uploaded {count} objects")

    info("")
    info("Demo dataset loaded. Try:")
    info("  lumilake job submit examples/templates/yaml/trading-agent.yaml \\")
    info("    --format yaml --input 'Stock=NVDA,AAPL' \\")
    info("    --output-prefix demo/trading-agent")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
