#!/usr/bin/env python3
"""Run pyright on the files changed between two git refs, but only fail CI on
errors that land on lines actually touched by the diff. Pre-existing errors
in the rest of a changed file are reported but do not fail the build, so we
can turn on type checking without first fixing the entire legacy codebase.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def changed_python_files(base, head, package_dir):
    result = run(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{base}", f"{head}", "--", package_dir]
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(2)
    files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return [f for f in files if f.endswith(".py") and Path(f).exists()]


def added_lines_for_file(base, head, repo_relative_path):
    """Return the set of 1-indexed line numbers introduced/modified in `head`."""
    result = run(["git", "diff", "--no-color", "--unified=0", f"{base}", f"{head}", "--", repo_relative_path])
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(2)
    lines = set()
    for line in result.stdout.splitlines():
        match = HUNK_RE.match(line)
        if not match:
            continue
        new_start = int(match.group(1))
        new_count = int(match.group(2)) if match.group(2) is not None else 1
        if new_count == 0:
            # Pure deletion at this point in the file; nothing was added.
            continue
        lines.update(range(new_start, new_start + new_count))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True, help="Repo-relative package directory, e.g. OmniGibson")
    parser.add_argument("--base", required=True, help="Base git ref/sha to diff against")
    parser.add_argument("--head", required=True, help="Head git ref/sha being checked")
    args = parser.parse_args()

    package_dir = args.package_dir.rstrip("/")
    changed_files = changed_python_files(args.base, args.head, package_dir)

    if not changed_files:
        print(f"No changed Python files under {package_dir}/ — skipping type check.")
        return 0

    print(f"Type-checking {len(changed_files)} changed file(s) under {package_dir}/:")
    for f in changed_files:
        print(f"  {f}")

    relative_files = [str(Path(f).relative_to(package_dir)) for f in changed_files]
    pyright_result = run(["pyright", "--outputjson", *relative_files], cwd=package_dir)

    try:
        report = json.loads(pyright_result.stdout)
    except json.JSONDecodeError:
        print("Failed to parse pyright output as JSON:", file=sys.stderr)
        print(pyright_result.stdout, file=sys.stderr)
        print(pyright_result.stderr, file=sys.stderr)
        return 2

    added_lines = {f: added_lines_for_file(args.base, args.head, f) for f in changed_files}

    blocking, pre_existing = [], []
    for diag in report.get("generalDiagnostics", []):
        if diag.get("severity") != "error":
            continue
        diag_file = str(Path(package_dir) / Path(diag["file"]).relative_to(Path(package_dir).resolve()))
        line = diag["range"]["start"]["line"] + 1
        entry = f"{diag_file}:{line}: {diag['message'].splitlines()[0]}"
        if line in added_lines.get(diag_file, ()):
            blocking.append(entry)
        else:
            pre_existing.append(entry)

    if pre_existing:
        print(f"\n{len(pre_existing)} pre-existing type error(s) in touched files (not blocking):")
        for entry in pre_existing:
            print(f"  {entry}")

    if blocking:
        print(f"\n{len(blocking)} new type error(s) on lines you touched (blocking):")
        for entry in blocking:
            print(f"  {entry}")
        return 1

    print("\nNo new type errors on touched lines.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
