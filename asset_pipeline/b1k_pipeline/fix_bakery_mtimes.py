"""Fix mtimes of files under cad/*/*/bakery/* using upload times from the DVC GCS remote.

Files in the bakery directories are symlinks into the DVC cache (cache.type=symlink).
The DVC cache layout (DVC 3.x: files/md5/<XX>/<rest>) matches the layout in the GCS
remote bucket, so the symlink's path relative to .dvc/cache is also the GCS object name.

We update the mtime of the cache target (not the symlink), because:
  * os.path.getmtime follows symlinks, so this is what readers observe.
  * Multiple bakery files may point at the same content-addressed cache entry; updating
    the target dedupes the work and keeps mtimes consistent across all references.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import tqdm
from google.cloud import storage

DEFAULT_REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PROJECT = "lucid-inquiry-205018"
DEFAULT_BUCKET = "ig-pipeline-cache"


def iter_bakery_files(repo_root: pathlib.Path):
    """Yield paths matching cad/*/*/bakery/* (one level)."""
    for file in (repo_root / "cad").glob("*/*/bakery/*"):
        yield file


def symlink_to_cache_relpath(
    symlink_path: pathlib.Path, cache_root: pathlib.Path
) -> pathlib.PurePosixPath | None:
    """Resolve a bakery symlink to a path relative to the DVC cache root.

    Uses os.path.realpath, which on Windows strips any ``\\\\?\\`` extended-path
    prefix that os.readlink may return for DVC symlinks.
    """
    try:
        real = pathlib.Path(os.path.realpath(symlink_path))
    except OSError:
        return None
    try:
        rel = real.relative_to(cache_root)
    except ValueError:
        return None
    return pathlib.PurePosixPath(*rel.parts)


def fetch_mtime(bucket: storage.Bucket, relpath: pathlib.PurePosixPath) -> float | None:
    blob = bucket.get_blob(str(relpath))
    if blob is None or blob.updated is None:
        return None
    return blob.updated.timestamp()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo",
        type=pathlib.Path,
        default=DEFAULT_REPO,
        help="Path to the asset_pipeline repo root (default: parent of this script).",
    )
    p.add_argument("--project", default=DEFAULT_PROJECT, help="GCP project name.")
    p.add_argument("--bucket", default=DEFAULT_BUCKET, help="GCS bucket name.")
    p.add_argument("--workers", type=int, default=32, help="Parallel GCS metadata workers.")
    p.add_argument("--dry-run", action="store_true", help="Don't actually call os.utime.")
    args = p.parse_args()

    repo = pathlib.Path(os.path.realpath(args.repo))
    cache_root = pathlib.Path(os.path.realpath(repo / ".dvc" / "cache"))
    if not cache_root.is_dir():
        print(f"DVC cache not found at {cache_root}", file=sys.stderr)
        return 1

    print(f"Scanning bakery files under {repo} ...")
    by_rel: dict[pathlib.PurePosixPath, list[pathlib.Path]] = {}
    n_total = 0
    n_not_symlink = 0
    for f in iter_bakery_files(repo):
        n_total += 1
        if not f.is_symlink():
            n_not_symlink += 1
            continue
        rel = symlink_to_cache_relpath(f, cache_root)
        if rel is None:
            print(f"WARN: cannot derive cache relpath for {f}", file=sys.stderr)
            continue
        by_rel.setdefault(rel, []).append(f)
    n_links = sum(len(v) for v in by_rel.values())
    print(
        f"  {n_total} matched files, {n_not_symlink} non-symlinks skipped, "
        f"{n_links} symlinks pointing to {len(by_rel)} unique cache entries."
    )
    if not by_rel:
        if n_total > 0 and n_not_symlink == n_total:
            print(
                "No symlinks found among matched files. On Windows, DVC only creates\n"
                "symlinks when Developer Mode is enabled or the process runs as admin;\n"
                "otherwise it falls back to copy/hardlink and there is no cache target\n"
                "to look up. Re-run `dvc checkout` with symlinks enabled, or adapt this\n"
                "script to look up hashes via the .dvc files directly.",
                file=sys.stderr,
            )
        return 0

    print(f"Querying GCS bucket gs://{args.bucket} ...")
    client = storage.Client(project=args.project)
    bucket = client.bucket(args.bucket)

    mtimes: dict[pathlib.PurePosixPath, float] = {}
    missing: list[pathlib.PurePosixPath] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_mtime, bucket, rel): rel for rel in by_rel}
        for fut in tqdm.tqdm(as_completed(futs), total=len(futs), desc="GCS"):
            rel = futs[fut]
            try:
                t = fut.result()
            except Exception as e:
                print(f"WARN: failed to fetch {rel}: {e}", file=sys.stderr)
                continue
            if t is None:
                missing.append(rel)
                continue
            mtimes[rel] = t
    if missing:
        print(
            f"WARN: {len(missing)} cache entries not found in bucket "
            f"(first few: {[str(x) for x in missing[:5]]})",
            file=sys.stderr,
        )

    print(
        f"{'(dry-run) ' if args.dry_run else ''}"
        f"Setting mtime on {len(mtimes)} cache files ..."
    )
    n_done = 0
    n_missing_local = 0
    affected_links = 0
    for rel, t in mtimes.items():
        target = cache_root / pathlib.Path(*rel.parts)
        if not target.exists():
            n_missing_local += 1
            continue
        if not args.dry_run:
            os.utime(target, (t, t))
        n_done += 1
        affected_links += len(by_rel[rel])

    print(f"Updated {n_done} cache entries (affects {affected_links} bakery symlinks).")
    if n_missing_local:
        print(
            f"WARN: {n_missing_local} cache entries had GCS mtimes but no local file.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
