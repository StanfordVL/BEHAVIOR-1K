#!/usr/bin/env python3
"""Delete run directories under coop2/runs/.

A run folder is `<topology>_agents<N>_repair_<on|off>_seed<S>_<timestamp>`; the
process log alone is tens of MB, so these accumulate fast. Anything that is not
shaped like a run folder (this script, README.md, a directory you parked here by
hand) is left alone -- the filter is on the name, not on "everything but the
files I know about", so a stray folder is never collateral.

    python coop2/runs/clean_runs.py            # list what would go, delete nothing
    python coop2/runs/clean_runs.py --yes      # delete it
    python coop2/runs/clean_runs.py --keep 3 --yes        # keep the 3 newest
    python coop2/runs/clean_runs.py --topology individual --yes
    python coop2/runs/clean_runs.py --older-than 7 --yes  # older than 7 days
"""

import argparse
import os
import re
import shutil
import time

RUNS_DIR = os.path.dirname(os.path.abspath(__file__))

# topology_agents<N>_repair_<on|off>_seed<S>_<YYYYmmdd>_<HHMMSS>_<micros>
RUN_NAME = re.compile(
    r"^(?P<topology>[a-z_]+)_agents\d+_repair_(?:on|off)_seed\d+_"
    r"(?P<stamp>\d{8}_\d{6})(?:_\d+)?$"
)


def find_runs(runs_dir=RUNS_DIR, topology=None):
    """Run folders, newest first (by the timestamp in the name)."""
    found = []
    for name in os.listdir(runs_dir):
        path = os.path.join(runs_dir, name)
        if not os.path.isdir(path):
            continue
        match = RUN_NAME.match(name)
        if match is None:
            continue
        if topology and match.group("topology") != topology:
            continue
        found.append((match.group("stamp"), name, path))
    found.sort(reverse=True)
    return [(name, path) for _stamp, name, path in found]


def directory_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def human(num_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.0f}{unit}" if unit == "B" else f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024.0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", "-y", action="store_true",
                        help="actually delete; without it this is a dry run")
    parser.add_argument("--keep", type=int, default=0, metavar="N",
                        help="keep the N most recent runs that match")
    parser.add_argument("--topology", default=None,
                        help="only individual / centralized / broadcast_chain")
    parser.add_argument("--older-than", type=float, default=None, metavar="DAYS",
                        help="only runs whose folder is older than DAYS days")
    parser.add_argument("--runs-dir", default=RUNS_DIR)
    args = parser.parse_args()

    runs = find_runs(args.runs_dir, args.topology)
    kept_recent = runs[:args.keep] if args.keep else []
    doomed = runs[args.keep:] if args.keep else runs

    if args.older_than is not None:
        cutoff = time.time() - args.older_than * 86400
        doomed = [(n, p) for n, p in doomed if os.path.getmtime(p) < cutoff]

    if not doomed:
        print(f"nothing to delete ({len(runs)} run folder(s) match, {len(kept_recent)} kept)")
        return

    total = 0
    for name, path in doomed:
        size = directory_size(path)
        total += size
        print(f"  {'delete' if args.yes else 'would delete'}  {name}  ({human(size)})")
        if args.yes:
            shutil.rmtree(path)

    verb = "deleted" if args.yes else "would free"
    print(f"{verb} {len(doomed)} run folder(s), {human(total)}")
    if kept_recent:
        print(f"kept {len(kept_recent)} most recent: " + ", ".join(n for n, _ in kept_recent))
    if not args.yes:
        print("dry run -- pass --yes to actually delete")


if __name__ == "__main__":
    main()
