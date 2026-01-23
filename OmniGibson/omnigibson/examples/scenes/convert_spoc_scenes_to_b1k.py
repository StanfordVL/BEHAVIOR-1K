"""
Convert SPOC scenes into BEHAVIOR-1K scene JSON + layout maps.
"""
import argparse
import hashlib
import json
import pathlib
import time
import traceback

from omnigibson.macros import gm

# Set OmniGibson macros before importing og
gm.HEADLESS = True
gm.ENABLE_FLATCACHE = False
gm.USE_GPU_DYNAMICS = False
gm.USE_ENCRYPTED_ASSETS = True

import omnigibson as og

from omnigibson.examples.scenes.load_spoc_scene import load_spoc_scene
from b1k_pipeline.usd_conversion.make_maps import generate_maps_for_current_scene


def should_process(scene_key, task_id, total_tasks, seed="potato"):
    if total_tasks <= 1:
        return True
    digest = hashlib.md5(f"{scene_key}{seed}".encode()).hexdigest()
    return int(digest, 16) % total_tasks == task_id


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def write_error(error_dir, scene_name, exc):
    ensure_dir(error_dir)
    error_path = error_dir / f"{scene_name}.txt"
    error_path.write_text(str(exc))


def output_scene_name(scene_name):
    split, idx = scene_name.rsplit("_", 1)
    split_path = pathlib.Path(split)
    return f"{split_path.stem}_{idx}"


def iter_spoc_scenes(jsonl_paths):
    for jsonl_path in jsonl_paths:
        jsonl_path = pathlib.Path(jsonl_path)
        with jsonl_path.open("r") as f:
            for i, _ in enumerate(f):
                yield f"{jsonl_path}_{i}"


def strip_init_info(json_path):
    with open(json_path, "r") as f:
        scene_info = json.load(f)
    scene_info.pop("init_info", None)
    with open(json_path, "w") as f:
        json.dump(scene_info, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-dir", required=True, help="Directory containing SPOC .jsonl files")
    parser.add_argument("--jsonl-glob", default="*.jsonl")
    parser.add_argument("--dataset-root", required=True, help="Root folder containing datasets (e.g., .../datasets)")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--total-tasks", type=int, default=1)
    parser.add_argument("--restart-every", type=int, default=0)
    parser.add_argument("--keep-init-info", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--success-prefix", default="", help="Prefix for success files (e.g., scriptname_jobid)")
    args = parser.parse_args()

    dataset_root = pathlib.Path(args.dataset_root)
    output_root = dataset_root / "spoc"
    jsonl_paths = sorted(pathlib.Path(args.jsonl_dir).glob(args.jsonl_glob))
    print(f"Found {len(jsonl_paths)} jsonl files under {args.jsonl_dir}")

    gm.DATA_PATH = str(dataset_root)
    gm.DATASET_PATH = str(output_root)

    og.launch()

    errors_dir = output_root / "errors"
    jobs_dir = output_root / "jobs"
    ensure_dir(errors_dir)
    ensure_dir(jobs_dir)
    processed = 0
    for scene_name in iter_spoc_scenes(jsonl_paths):
        out_name = output_scene_name(scene_name)
        if not should_process(out_name, args.task_id, args.total_tasks):
            continue

        output_scene_dir = output_root / "scenes" / out_name
        json_dir = output_scene_dir / "json"
        layout_dir = output_scene_dir / "layout"
        json_path = json_dir / f"{out_name}_best.json"
        success_path = output_scene_dir / "import.success"

        if json_path.exists() and not args.overwrite:
            print(f"Skipping {out_name}: output exists")
            continue

        print(f"Processing {out_name}")
        try:
            ensure_dir(json_dir)
            ensure_dir(layout_dir)

            load_spoc_scene(scene_name)
            og.sim.step()
            og.sim.save(json_paths=[str(json_path)])
            if not args.keep_init_info:
                strip_init_info(json_path)

            map_start = time.time()
            generate_maps_for_current_scene(str(layout_dir))
            map_end = time.time()
            print(f"Generated maps in {map_end - map_start:.2f}s")

            success_path.touch()
            processed += 1
        except Exception:
            exc = traceback.format_exc()
            print(f"Error processing {out_name}:\n{exc}")
            write_error(errors_dir, out_name, exc)
        finally:
            og.clear()

        if args.restart_every and processed >= args.restart_every:
            break

    og.shutdown()
    print(f"Processed {processed} scenes")
    success_filename = f"{args.success_prefix}_{args.task_id}.success" if args.success_prefix else f"{args.task_id}.success"
    (jobs_dir / success_filename).touch()


if __name__ == "__main__":
    main()

