#!/usr/bin/env python3
"""List GCS raw-data objects and generate per-task instance-id files."""

from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_PATTERN = (
    r"(?:^|/)task-?(?P<task_id>\d+)[^/]*/" r"(?:episode_(?P<episode_id>\d{8})|task\d+_(?P<instance_id>\d+))\.hdf5$"
)


def parse_bucket_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    bucket_and_prefix = uri[len("gs://") :]
    bucket, _, prefix = bucket_and_prefix.partition("/")
    return bucket, prefix


class PythonGCSLister:
    def __init__(self, credentials_path: str = "") -> None:
        import google.auth
        from google.auth.transport.requests import Request

        scopes = ["https://www.googleapis.com/auth/devstorage.read_only"]
        if credentials_path:
            from google.auth import load_credentials_from_file

            self.credentials, _ = load_credentials_from_file(credentials_path, scopes=scopes)
        else:
            self.credentials, _ = google.auth.default(scopes=scopes)
        self.request = Request()

    def _headers(self) -> dict[str, str]:
        if not self.credentials.valid:
            self.credentials.refresh(self.request)
        return {"Authorization": f"Bearer {self.credentials.token}"}

    def list_objects(self, bucket: str, prefix: str) -> list[dict[str, str]]:
        objects = []
        page_token = ""
        while True:
            query = {
                "prefix": prefix,
                "fields": "items(name,size),nextPageToken",
                "maxResults": "1000",
            }
            if page_token:
                query["pageToken"] = page_token
            url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?{urllib.parse.urlencode(query)}"
            req = urllib.request.Request(url, headers=self._headers())
            with urllib.request.urlopen(req, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
            objects.extend(payload.get("items", []))
            page_token = payload.get("nextPageToken", "")
            if not page_token:
                return objects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket_uri", default="gs://behavior-demos")
    parser.add_argument("--prefix", default="", help="Optional prefix inside the bucket.")
    parser.add_argument("--credentials", default="", help="Google credentials JSON path. Defaults to ADC.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest_csv", default="")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    args = parser.parse_args()

    bucket, uri_prefix = parse_bucket_uri(args.bucket_uri.rstrip("/"))
    prefix = args.prefix or uri_prefix
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    matcher = re.compile(args.pattern)
    client = PythonGCSLister(args.credentials)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_to_instances: dict[str, set[str]] = {}
    rows = []
    total_size = 0
    for obj in client.list_objects(bucket, prefix):
        name = obj["name"]
        match = matcher.search(name)
        if not match:
            continue
        size = int(obj.get("size", 0))
        total_size += size
        task_id = f"{int(match.group('task_id')):04d}"
        instance_id_match = match.groupdict().get("instance_id")
        if instance_id_match is None:
            episode_id = match.group("episode_id")
            instance_id = episode_id[4:]
        else:
            instance_id = f"{int(instance_id_match):03d}"
            episode_id = f"{task_id}{instance_id}0"
        task_to_instances.setdefault(task_id, set()).add(instance_id)
        rows.append(
            {
                "task_id": task_id,
                "instance_id": instance_id,
                "episode_id": episode_id,
                "object": name,
                "size": str(size),
            }
        )

    for task_id, instance_ids in sorted(task_to_instances.items()):
        path = output_dir / f"task-{task_id}.txt"
        path.write_text("\n".join(sorted(instance_ids)) + "\n")
        print(f"{path}: {len(instance_ids)} instances")

    if args.manifest_csv:
        manifest_path = Path(args.manifest_csv)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["task_id", "instance_id", "episode_id", "object", "size"])
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: (row["task_id"], row["instance_id"], row["object"])))
        print(f"{manifest_path}: {len(rows)} objects")

    print(f"summary tasks={len(task_to_instances)} objects={len(rows)} bytes={total_size}")


if __name__ == "__main__":
    main()
