"""
Run the MA-Crafter COOP2 experiment grid.

Default grid:
  topology: individual, centralized, broadcast_chain
  agent count: 3, 6
  repair: off, on
  seed: 42
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = ROOT / "experiment"


TOPOLOGY_SCRIPTS = {
    "individual": EXPERIMENT_DIR / "run_individual.py",
    "centralized": EXPERIMENT_DIR / "run_centralized.py",
    "broadcast_chain": EXPERIMENT_DIR / "run_broadcast_chain.py",
}


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_command(args, topology: str, agent_count: int, repair_enabled: bool, seed: int) -> list[str]:
    command = [
        sys.executable,
        str(TOPOLOGY_SCRIPTS[topology]),
        "--agents",
        str(agent_count),
        "--steps",
        str(args.steps),
        "--time-limit-seconds",
        str(args.time_limit_seconds),
        "--model",
        args.model,
        "--seed",
        str(seed),
    ]
    if args.output_root:
        command.extend(["--output-root", str(args.output_root)])
    if args.quiet:
        command.append("--quiet")
    if args.llm_quiet:
        command.append("--llm-quiet")
    if args.show:
        command.append("--show")
    if not args.record_video:
        command.append("--no-video")
    if repair_enabled:
        command.append("--coop2-repair")
    return command


def save_manifest(manifest_path: Path, records: list[dict]) -> None:
    with open(manifest_path, "w") as f:
        json.dump(records, f, indent=2)


def run_sequential(args, runs: list[dict], manifest_path: Path) -> int:
    records = []
    for run in runs:
        print(
            f"[{run['index']}/{len(runs)}] {run['topology']}, agents={run['agents']}, "
            f"repair={'on' if run['repair'] else 'off'}, seed={run['seed']}"
        )
        if args.dry_run:
            records.append({**run, "status": "dry_run"})
            continue

        started = time.monotonic()
        result = subprocess.run(run["command"], cwd=ROOT)
        duration = time.monotonic() - started
        record = {
            **run,
            "status": "completed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "duration_seconds": round(duration, 3),
        }
        records.append(record)
        save_manifest(manifest_path, records)

        if result.returncode != 0 and not args.continue_on_error:
            print(f"Stopping after failed run {run['index']}.")
            return result.returncode

    save_manifest(manifest_path, records)
    return 0


def dry_run_schedule(args, runs: list[dict], manifest_path: Path) -> int:
    records = []
    wave = 1
    wave_slots = 0
    for run in runs:
        status = "dry_run"
        if run["agents"] > args.max_concurrent_agents:
            status = "dry_run_over_budget"
        elif wave_slots and wave_slots + run["agents"] > args.max_concurrent_agents:
            wave += 1
            wave_slots = 0
        wave_slots += run["agents"]
        print(
            f"[{run['index']}/{len(runs)}] wave={wave}, {run['topology']}, "
            f"agents={run['agents']}, repair={'on' if run['repair'] else 'off'}, "
            f"seed={run['seed']} (wave slots {wave_slots}/{args.max_concurrent_agents})"
        )
        records.append({
            **run,
            "status": status,
            "wave": wave,
            "slot_budget": args.max_concurrent_agents,
        })
    save_manifest(manifest_path, records)
    return 0


def run_parallel(args, runs: list[dict], manifest_dir: Path, manifest_path: Path) -> int:
    pending = list(runs)
    active = []
    records = []
    safe_model = str(args.model).replace("/", "_").replace(":", "_")
    runner_logs_dir = manifest_dir / f"runner_logs_{safe_model}"
    runner_logs_dir.mkdir(parents=True, exist_ok=True)
    stop_launching = False

    def active_slots() -> int:
        return sum(item["agents"] for item in active)

    def can_launch(run: dict) -> bool:
        if stop_launching:
            return False
        if args.max_parallel_runs and len(active) >= args.max_parallel_runs:
            return False
        return active_slots() + run["agents"] <= args.max_concurrent_agents

    while pending or active:
        launched = False
        while pending and can_launch(pending[0]):
            run = pending.pop(0)
            print(
                f"[{run['index']}/{len(runs)}] launch {run['topology']}, "
                f"agents={run['agents']}, repair={'on' if run['repair'] else 'off'}, "
                f"seed={run['seed']} "
                f"(slots {active_slots() + run['agents']}/{args.max_concurrent_agents})"
            )

            log_name = (
                f"{run['index']:03d}_{run['topology']}_agents{run['agents']}_"
                f"repair{'on' if run['repair'] else 'off'}_seed{run['seed']}.log"
            )
            log_path = runner_logs_dir / log_name
            log_file = open(log_path, "w")
            started = time.monotonic()
            process = subprocess.Popen(
                run["command"],
                cwd=ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
            active.append({
                **run,
                "process": process,
                "started": started,
                "runner_log": str(log_path),
                "log_file": log_file,
            })
            launched = True

        finished = []
        for item in active:
            returncode = item["process"].poll()
            if returncode is None:
                continue
            item["log_file"].close()
            duration = time.monotonic() - item["started"]
            record = {
                key: value
                for key, value in item.items()
                if key not in {"process", "started", "log_file"}
            }
            record.update({
                "status": "completed" if returncode == 0 else "failed",
                "returncode": returncode,
                "duration_seconds": round(duration, 3),
            })
            records.append(record)
            finished.append(item)
            print(
                f"[{item['index']}/{len(runs)}] done {item['topology']}, "
                f"agents={item['agents']}, repair={'on' if item['repair'] else 'off'} "
                f"-> {record['status']}"
            )
            if returncode != 0 and not args.continue_on_error:
                stop_launching = True

        for item in finished:
            active.remove(item)

        if finished:
            save_manifest(manifest_path, records + [
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"process", "started", "log_file"}
                }
                for item in active
            ])

        if stop_launching and active:
            for item in active:
                item["process"].terminate()
            for item in active:
                try:
                    item["process"].wait(timeout=30)
                except subprocess.TimeoutExpired:
                    item["process"].kill()
                    item["process"].wait()
                item["log_file"].close()
            return 1
        if stop_launching:
            return 1

        if not launched and not finished:
            if pending and not active and pending[0]["agents"] > args.max_concurrent_agents:
                run = pending[0]
                print(
                    f"Run {run['index']} needs {run['agents']} slots, "
                    f"but max-concurrent-agents is {args.max_concurrent_agents}."
                )
                return 2
            time.sleep(args.poll_seconds)

    save_manifest(manifest_path, records)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MA-Crafter COOP2 experiment grid")
    parser.add_argument(
        "--topologies",
        nargs="+",
        default=["individual", "centralized", "broadcast_chain"],
        choices=sorted(TOPOLOGY_SCRIPTS),
        help="Communication structures evaluated in the paper.",
    )
    parser.add_argument("--agent-counts", type=parse_int_list, default=parse_int_list("3,6"))
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("42"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--time-limit-seconds", type=float, default=150)
    parser.add_argument("--model", type=str, default="gpt-5.4-mini")
    parser.add_argument(
        "--repair",
        choices=["both", "off", "on"],
        default="both",
        help="Repair conditions to run",
    )
    parser.add_argument("--quiet", action="store_true", default=True)
    parser.add_argument("--verbose", dest="quiet", action="store_false")
    parser.add_argument("--llm-quiet", action="store_true", default=True)
    parser.add_argument("--llm-verbose", dest="llm_quiet", action="store_false")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--record-video", action="store_true", help="Record episode GIFs. Disabled by default for batch runs.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Experiment folder for manifest, runner logs, and run result directories.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--max-concurrent-agents",
        type=int,
        default=1,
        help="Weighted parallelism budget. A run consumes its agent count as slots.",
    )
    parser.add_argument(
        "--max-parallel-runs",
        type=int,
        default=0,
        help="Optional cap on simultaneous subprocesses. 0 means no separate cap.",
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()

    repair_modes = [False, True]
    if args.repair == "off":
        repair_modes = [False]
    elif args.repair == "on":
        repair_modes = [True]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_model = str(args.model).replace("/", "_").replace(":", "_")
    if args.output_root is None:
        manifest_dir = SCRIPT_DIR / "results" / f"coop2_traces_{safe_model}_{timestamp}"
        args.output_root = manifest_dir
    else:
        manifest_dir = args.output_root.resolve()
        args.output_root = manifest_dir
    manifest_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    index = 1
    for seed in args.seeds:
        for topology in args.topologies:
            for agent_count in args.agent_counts:
                for repair_enabled in repair_modes:
                    command = build_command(args, topology, agent_count, repair_enabled, seed)
                    runs.append({
                        "index": index,
                        "topology": topology,
                        "agents": agent_count,
                        "repair": repair_enabled,
                        "seed": seed,
                        "command": command,
                    })
                    index += 1

    print(f"Planned runs: {len(runs)}")
    manifest_path = manifest_dir / f"manifest_{safe_model}.json"
    print(f"Manifest: {manifest_path}")
    print(f"Agent concurrency budget: {args.max_concurrent_agents}")

    if args.dry_run:
        return dry_run_schedule(args, runs, manifest_path)
    if args.max_concurrent_agents <= 1:
        return run_sequential(args, runs, manifest_path)
    return run_parallel(args, runs, manifest_dir, manifest_path)


if __name__ == "__main__":
    raise SystemExit(main())
