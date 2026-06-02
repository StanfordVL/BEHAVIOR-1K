#!/usr/bin/env python3
"""Submit and monitor sharded LeRobot v3 replay jobs on Stanford Slurm.

This script is intentionally dry-run by default. Add --submit to call sbatch,
and add --loop to keep watching and relaunching incomplete shards.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ACCOUNTS = "sc-loprio,viscam"
DEFAULT_PARTITION = "svl,sc"
JOB_NAME_PREFIX = "rsc"


@dataclass(frozen=True)
class Shard:
    task_name: str
    task_index: int
    shard_index: int
    demo_ids: tuple[int, ...]

    @property
    def shard_id(self) -> str:
        return f"{self.shard_index:03d}"

    @property
    def job_name(self) -> str:
        return f"{JOB_NAME_PREFIX}_{self.task_index:03d}_{self.shard_index:03d}"

    @property
    def key(self) -> str:
        return f"{self.task_index:03d}:{self.shard_index:03d}"

    @property
    def demo_ids_arg(self) -> str:
        return ",".join(str(demo_id) for demo_id in self.demo_ids)


def load_task_mapping(repo_root: Path) -> dict[str, int]:
    eval_utils = repo_root / "OmniGibson" / "omnigibson" / "learning" / "utils" / "eval_utils.py"
    tree = ast.parse(eval_utils.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TASK_NAMES_TO_INDICES":
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"TASK_NAMES_TO_INDICES not found in {eval_utils}")


def parse_task_names(args: argparse.Namespace, task_mapping: dict[str, int]) -> list[str]:
    if args.task_names_file:
        names = [
            line.strip()
            for line in Path(args.task_names_file).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    elif args.tasks:
        names = [name.strip() for name in args.tasks.split(",") if name.strip()]
    else:
        names = [name for name, _ in sorted(task_mapping.items(), key=lambda item: item[1])]

    unknown = sorted(set(names) - set(task_mapping))
    if unknown:
        raise ValueError(f"Unknown task names: {unknown}")
    return names


def build_shards(
    task_names: list[str],
    task_mapping: dict[str, int],
    first_demo_id: int,
    num_demos: int,
    shard_size: int,
) -> list[Shard]:
    shards = []
    demo_ids = list(range(first_demo_id, first_demo_id + num_demos))
    for task_name in task_names:
        task_index = task_mapping[task_name]
        for shard_index, start in enumerate(range(0, len(demo_ids), shard_size)):
            shards.append(
                Shard(
                    task_name=task_name,
                    task_index=task_index,
                    shard_index=shard_index,
                    demo_ids=tuple(demo_ids[start : start + shard_size]),
                )
            )
    return shards


def shard_repo_id(shard: Shard) -> str:
    return f"b1k_shards/{shard.task_name}/shard-{shard.shard_id}"


def marker_dir(lerobot_root_dir: Path, shard: Shard) -> Path:
    return lerobot_root_dir / ".replay_status" / shard.task_name / f"shard-{shard.shard_id}"


def is_complete(lerobot_root_dir: Path, shard: Shard) -> bool:
    return (marker_dir(lerobot_root_dir, shard) / "complete").exists()


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"shards": {}}
    return json.loads(path.read_text())


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp_path.replace(path)


def run_command(cmd: list[str], required: bool) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, check=required, text=True, capture_output=True)
    except FileNotFoundError:
        if required:
            raise
        return subprocess.CompletedProcess(cmd, returncode=127, stdout="", stderr=f"{cmd[0]} not found")


def active_replay_jobs(user: str) -> dict[str, str]:
    cmd = ["squeue", "-h", "-u", user, "-o", "%A|%j|%T"]
    result = run_command(cmd, required=False)
    if result.returncode != 0:
        return {}

    active = {}
    for line in result.stdout.splitlines():
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        job_id, job_name, state = parts
        if job_name.startswith(f"{JOB_NAME_PREFIX}_"):
            active[job_name] = f"{job_id}:{state}"
    return active


def submit_shard(
    shard: Shard,
    args: argparse.Namespace,
    account: str,
) -> str:
    script_path = Path(args.sbatch_script).resolve()
    cmd = [
        "sbatch",
        "--parsable",
        f"--account={account}",
        f"--partition={args.partition}",
        f"--job-name={shard.job_name}",
        str(script_path),
        "--data_folder",
        args.data_folder,
        "--task_name",
        shard.task_name,
        "--shard_id",
        shard.shard_id,
        "--demo_ids",
        shard.demo_ids_arg,
        "--lerobot_root_dir",
        args.lerobot_root_dir,
        "--lerobot_repo_id",
        shard_repo_id(shard),
        "--flush_every_n_steps",
        str(args.flush_every_n_steps),
        "--repo_root",
        args.repo_root,
    ]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def run_once(args: argparse.Namespace) -> bool:
    repo_root = Path(args.repo_root).resolve()
    task_mapping = load_task_mapping(repo_root)
    task_names = parse_task_names(args, task_mapping)
    shards = build_shards(task_names, task_mapping, args.first_demo_id, args.num_demos, args.shard_size)

    lerobot_root_dir = Path(args.lerobot_root_dir).resolve()
    state_path = Path(args.state_file).resolve()
    state = load_state(state_path)
    state.setdefault("shards", {})

    active = active_replay_jobs(args.user)
    active_count = len(active)
    accounts = [account.strip() for account in args.accounts.split(",") if account.strip()]
    if not accounts:
        raise ValueError("--accounts must include at least one Slurm account")

    complete = 0
    active_shards = 0
    submitted = 0
    blocked_by_retries = 0

    for shard in shards:
        shard_state = state["shards"].setdefault(shard.key, {"attempts": 0, "job_ids": []})
        if is_complete(lerobot_root_dir, shard):
            complete += 1
            continue
        if shard.job_name in active:
            active_shards += 1
            continue
        if shard_state["attempts"] >= args.max_retries:
            blocked_by_retries += 1
            continue
        if active_count + submitted >= args.max_active:
            continue

        account = accounts[(shard.task_index + shard.shard_index + shard_state["attempts"]) % len(accounts)]
        if args.submit:
            job_id = submit_shard(shard, args, account)
            shard_state["attempts"] += 1
            shard_state["job_ids"].append(job_id)
            shard_state["last_submit_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            shard_state["last_account"] = account
            submitted += 1
            print(f"submitted {shard.job_name} job_id={job_id} account={account} demos={shard.demo_ids_arg}")
        else:
            submitted += 1
            print(f"dry-run submit {shard.job_name} account={account} demos={shard.demo_ids_arg}")

    if args.submit:
        save_state(state_path, state)

    total = len(shards)
    incomplete = total - complete
    print(
        "summary: "
        f"total_shards={total} complete={complete} incomplete={incomplete} "
        f"active={active_shards} submitted={submitted} retry_blocked={blocked_by_retries}"
    )
    return incomplete == 0


def main() -> None:
    default_repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_folder", required=True)
    parser.add_argument("--repo_root", default=str(default_repo_root))
    parser.add_argument(
        "--sbatch_script",
        default=str(default_repo_root / "OmniGibson" / "scripts" / "learning" / "replay_data_sc.sbatch"),
    )
    parser.add_argument("--lerobot_root_dir", default=None)
    parser.add_argument("--state_file", default=None)
    parser.add_argument("--tasks", default="", help="Comma-separated task names. Defaults to all tasks.")
    parser.add_argument("--task_names_file", default="")
    parser.add_argument("--first_demo_id", type=int, default=0)
    parser.add_argument("--num_demos", type=int, default=200)
    parser.add_argument("--shard_size", type=int, default=10)
    parser.add_argument("--flush_every_n_steps", type=int, default=1000)
    parser.add_argument("--accounts", default=DEFAULT_ACCOUNTS)
    parser.add_argument("--partition", default=DEFAULT_PARTITION)
    parser.add_argument("--user", default=os.environ.get("USER", "wsai"))
    parser.add_argument("--max_active", type=int, default=160)
    parser.add_argument("--max_retries", type=int, default=8)
    parser.add_argument("--interval_seconds", type=int, default=900)
    parser.add_argument("--submit", action="store_true", help="Actually call sbatch. Default is dry-run.")
    parser.add_argument("--loop", action="store_true", help="Keep monitoring until all shards complete.")
    args = parser.parse_args()

    if args.lerobot_root_dir is None:
        args.lerobot_root_dir = str(Path(args.data_folder) / "lerobot_shards")
    if args.state_file is None:
        args.state_file = str(Path(args.lerobot_root_dir) / "monitor" / "replay_data_sc_state.json")

    while True:
        done = run_once(args)
        if done or not args.loop:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
