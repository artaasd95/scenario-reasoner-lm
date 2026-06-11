"""S3-compatible bucket sync for RunPod network volumes and remote object stores."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def bucket_config() -> dict[str, str]:
    return {
        "endpoint": _env("BUCKET_ENDPOINT_URL", _env("S3_ENDPOINT_URL")),
        "bucket": _env("BUCKET_NAME", _env("S3_BUCKET")),
        "access_key": _env("BUCKET_ACCESS_KEY", _env("AWS_ACCESS_KEY_ID")),
        "secret_key": _env("BUCKET_SECRET_KEY", _env("AWS_SECRET_ACCESS_KEY")),
        "region": _env("BUCKET_REGION", _env("AWS_DEFAULT_REGION", "auto")),
        "prefix": _env("BUCKET_PREFIX", ""),
    }


def remote_key(local_path: Path, local_root: Path, prefix: str) -> str:
    rel = local_path.relative_to(local_root).as_posix()
    if prefix:
        return f"{prefix.rstrip('/')}/{rel}"
    return rel


def upload_tree(local_root: Path, *, dry_run: bool = True) -> list[str]:
    """Upload directory tree to configured bucket (dry-run by default)."""
    cfg = bucket_config()
    if not cfg["bucket"]:
        raise ValueError("Set BUCKET_NAME or S3_BUCKET for bucket sync")

    uploaded: list[str] = []
    manifest = {
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "local_root": str(local_root.resolve()),
        "bucket": cfg["bucket"],
        "prefix": cfg["prefix"],
        "objects": [],
    }

    client = None
    if not dry_run:
        import boto3

        client = boto3.client(
            "s3",
            endpoint_url=cfg["endpoint"] or None,
            aws_access_key_id=cfg["access_key"] or None,
            aws_secret_access_key=cfg["secret_key"] or None,
            region_name=cfg["region"] or None,
        )

    for path in sorted(local_root.rglob("*")):
        if not path.is_file():
            continue
        key = remote_key(path, local_root, cfg["prefix"])
        manifest["objects"].append(key)
        uploaded.append(key)
        if client is not None:
            client.upload_file(str(path), cfg["bucket"], key)

    manifest_path = local_root / ".bucket_sync_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return uploaded


def download_tree(local_root: Path, *, keys: list[str] | None = None, dry_run: bool = True) -> list[str]:
    """Download objects from bucket into local_root."""
    cfg = bucket_config()
    if not cfg["bucket"]:
        raise ValueError("Set BUCKET_NAME or S3_BUCKET for bucket sync")

    downloaded: list[str] = []
    if dry_run:
        return keys or []

    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=cfg["endpoint"] or None,
        aws_access_key_id=cfg["access_key"] or None,
        aws_secret_access_key=cfg["secret_key"] or None,
        region_name=cfg["region"] or None,
    )
    prefix = cfg["prefix"].rstrip("/") + "/" if cfg["prefix"] else ""
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=cfg["bucket"], Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if keys and key not in keys:
                continue
            rel = key[len(prefix) :] if prefix and key.startswith(prefix) else key
            dest = local_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(cfg["bucket"], key, str(dest))
            downloaded.append(key)
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Bucket-native artifact sync")
    parser.add_argument("direction", choices=["upload", "download"])
    parser.add_argument("path", type=Path, default=Path("experiments/results"), nargs="?")
    parser.add_argument("--execute", action="store_true", help="Perform real transfer (default: dry-run)")
    args = parser.parse_args()
    dry_run = not args.execute
    if args.direction == "upload":
        keys = upload_tree(args.path, dry_run=dry_run)
    else:
        keys = download_tree(args.path, dry_run=dry_run)
    mode = "dry-run" if dry_run else "executed"
    print(f"{mode}: {len(keys)} object(s)")


if __name__ == "__main__":
    main()
