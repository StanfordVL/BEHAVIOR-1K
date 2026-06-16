#!/usr/bin/env python3
"""Download BEHAVIOR raw HDF5 demos from GCS into the challenge raw-data layout."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import signal
import subprocess
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_BUCKET_URI = "gs://behavior-demos"
DEFAULT_TARGET_ROOT = "/vision/group/behavior/2026-challenge-rawdata"
DEFAULT_SOURCE_TEMPLATE = "{bucket_uri}/task-{task_id:04d}/episode_{episode_id}.hdf5"


def format_episode_id(task_id: int, instance_id: int) -> str:
    if task_id >= 50:
        return f"{task_id:04d}{instance_id:03d}0"
    return f"{task_id:04d}{instance_id:04d}"


def parse_ids(value: str) -> list[int]:
    ids = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        ids.append(int(item))
    return ids


def load_instance_ids(args: argparse.Namespace) -> list[int]:
    if args.instance_ids:
        return parse_ids(args.instance_ids)
    if args.instance_ids_file:
        ids = []
        for line in Path(args.instance_ids_file).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(int(line.split(",")[0]))
        return ids
    return [args.first_instance_id + i * args.instance_stride for i in range(args.num_instances)]


def load_manifest_jobs(args: argparse.Namespace) -> list[tuple[int, str, str, Path]]:
    jobs = []
    bucket_uri = args.bucket_uri.rstrip("/")
    task_dir = Path(args.target_root) / f"task-{args.task_id:04d}"
    with Path(args.gcs_manifest_csv).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = int(row["task_id"])
            if task_id != args.task_id:
                continue
            instance_id = int(row["instance_id"])
            episode_id = row.get("episode_id") or format_episode_id(task_id, instance_id)
            object_name = row.get("object") or row.get("source")
            if object_name is None:
                raise ValueError(f"Manifest row lacks object/source column: {row}")
            source = object_name if object_name.startswith("gs://") else f"{bucket_uri}/{object_name}"
            dest = task_dir / f"episode_{episode_id}.hdf5"
            jobs.append((instance_id, episode_id, source, dest))
    return jobs


def resolve_tool(tool: str) -> str:
    if tool in {"python", "python-gcs", "gcs-python"}:
        return "python-gcs"
    if tool != "auto":
        return tool
    for candidate in ("gcloud", "gsutil"):
        path = shutil.which(candidate)
        if path:
            return path
    return "python-gcs"


def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    bucket_and_object = uri[len("gs://") :]
    bucket, _, object_name = bucket_and_object.partition("/")
    if not bucket or not object_name:
        raise ValueError(f"Expected gs://bucket/object URI, got: {uri}")
    return bucket, object_name


class PythonGCSClient:
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

    def download(self, source: str, dest: Path) -> None:
        bucket, object_name = parse_gs_uri(source)
        encoded_object = urllib.parse.quote(object_name, safe="")
        url = f"https://storage.googleapis.com/download/storage/v1/b/{bucket}/o/{encoded_object}?alt=media"
        req = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(req, timeout=120) as response, dest.open("wb") as f:
            shutil.copyfileobj(response, f, length=1024 * 1024)


def build_copy_command(tool: str, source: str, dest: Path) -> list[str]:
    name = Path(tool).name
    if name == "gsutil":
        return [tool, "-q", "cp", source, str(dest)]
    return [tool, "storage", "cp", "--quiet", source, str(dest)]


def chmod_group_writable(path: Path) -> None:
    try:
        path.chmod(0o664)
    except PermissionError:
        pass


def download_one(
    tool: str,
    client: PythonGCSClient | None,
    source: str,
    dest: Path,
    dry_run: bool,
) -> tuple[str, str, str]:
    if dest.exists() and dest.stat().st_size > 0:
        return ("exists", source, str(dest))

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.part-{os.getpid()}-{uuid.uuid4().hex}")

    if dry_run:
        return ("dry-run", source, str(dest))

    if tool == "python-gcs":
        try:
            assert client is not None
            client.download(source, tmp)
        except Exception as e:
            if tmp.exists():
                tmp.unlink()
            return ("failed", source, f"{type(e).__name__}: {e}")
    else:
        cmd = build_copy_command(tool, source, tmp)
        result = subprocess.run(cmd, text=True, capture_output=True)
        if result.returncode != 0:
            if tmp.exists():
                tmp.unlink()
            message = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
            return ("failed", source, message)

    if not tmp.exists() or tmp.stat().st_size == 0:
        if tmp.exists():
            tmp.unlink()
        return ("failed", source, "download produced an empty/missing file")

    if dest.exists() and dest.stat().st_size > 0:
        tmp.unlink()
        return ("exists-after-download", source, str(dest))

    tmp.replace(dest)
    chmod_group_writable(dest)
    return ("downloaded", source, str(dest))


def write_manifest_row(path: Path | None, row: dict[str, str]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "instance_id", "episode_id", "status", "source", "dest"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--bucket_uri", default=DEFAULT_BUCKET_URI)
    parser.add_argument("--target_root", default=DEFAULT_TARGET_ROOT)
    parser.add_argument(
        "--source_template",
        default=DEFAULT_SOURCE_TEMPLATE,
        help=(
            "Python format template for source URI. Available fields: bucket_uri, " "task_id, instance_id, episode_id."
        ),
    )
    parser.add_argument("--instance_ids", default="", help="Comma-separated instance ids.")
    parser.add_argument("--instance_ids_file", default="", help="One instance id per line, or CSV with id first.")
    parser.add_argument(
        "--gcs_manifest_csv",
        default="",
        help="CSV with task_id, instance_id, episode_id, object columns from generate_gcs_rawdata_manifest.py.",
    )
    parser.add_argument("--first_instance_id", type=int, default=0)
    parser.add_argument("--num_instances", type=int, default=200)
    parser.add_argument("--instance_stride", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tool", default="auto", help="auto, python-gcs, gcloud, gsutil, or full path.")
    parser.add_argument(
        "--credentials",
        default="",
        help="Path to a Google credentials JSON file for --tool python-gcs. Defaults to ADC env/config.",
    )
    parser.add_argument("--manifest", default="", help="CSV status manifest path.")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    tool = "python-gcs" if args.dry_run and args.tool == "auto" else resolve_tool(args.tool)
    client = None if args.dry_run or tool != "python-gcs" else PythonGCSClient(args.credentials)
    manifest = Path(args.manifest) if args.manifest else None

    if args.gcs_manifest_csv:
        jobs = load_manifest_jobs(args)
    else:
        bucket_uri = args.bucket_uri.rstrip("/")
        task_dir = Path(args.target_root) / f"task-{args.task_id:04d}"
        jobs = []
        for instance_id in load_instance_ids(args):
            episode_id = format_episode_id(args.task_id, instance_id)
            source = args.source_template.format(
                bucket_uri=bucket_uri,
                task_id=args.task_id,
                instance_id=instance_id,
                episode_id=episode_id,
            )
            dest = task_dir / f"episode_{episode_id}.hdf5"
            jobs.append((instance_id, episode_id, source, dest))

    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        futures = {
            executor.submit(download_one, tool, client, source, dest, args.dry_run): (
                instance_id,
                episode_id,
                source,
                dest,
            )
            for instance_id, episode_id, source, dest in jobs
        }
        for future in as_completed(futures):
            instance_id, episode_id, source, dest = futures[future]
            status, resolved_source, resolved_dest = future.result()
            counts[status] = counts.get(status, 0) + 1
            print(
                f"{status}: task={args.task_id:04d} instance={instance_id:04d} {resolved_source} -> {resolved_dest}",
                flush=True,
            )
            write_manifest_row(
                manifest,
                {
                    "task_id": f"{args.task_id:04d}",
                    "instance_id": f"{instance_id:04d}",
                    "episode_id": episode_id,
                    "status": status,
                    "source": source,
                    "dest": str(dest),
                },
            )

    failed = counts.get("failed", 0)
    print(f"summary task={args.task_id:04d} " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
