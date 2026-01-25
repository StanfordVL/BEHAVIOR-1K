"""
Convert Habitat-style scenes (e.g., HSSD / AI2-THOR) into BEHAVIOR-1K scene JSON + layout maps.
"""

import argparse
import hashlib
import json
import pathlib
import time
import traceback

import sys
sys.path.append(str(pathlib.Path(__file__).parents[4] / "asset_pipeline"))

from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_dataset_path

# Set OmniGibson macros before importing og
gm.HEADLESS = True
gm.ENABLE_FLATCACHE = False
gm.USE_GPU_DYNAMICS = False
gm.USE_ENCRYPTED_ASSETS = True

import omnigibson as og

from omnigibson.examples.scenes.load_habitat_scene import load_habitat_scene
from b1k_pipeline.usd_conversion.make_maps import generate_maps_for_current_scene


def normalize_scene_name(scene_path):
    name = pathlib.Path(scene_path).stem
    if name.endswith(".scene_instance"):
        name = name[: -len(".scene_instance")]
    return name


def should_process(scene_key, task_id, total_tasks, seed="potato"):
    if total_tasks <= 1:
        return True
    digest = hashlib.md5(f"{scene_key}{seed}".encode()).hexdigest()
    return int(digest, 16) % total_tasks == task_id


def collect_scene_files(scene_dir, pattern):
    scene_dir = pathlib.Path(scene_dir)
    return sorted(scene_dir.glob(pattern))


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def write_error(error_dir, scene_name, exc):
    ensure_dir(error_dir)
    error_path = error_dir / f"{scene_name}.txt"
    error_path.write_text(str(exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="hssd", choices=["hssd", "ai2thor"])
    parser.add_argument("--scene-dir", default="/checkpoint/clear/cgokmen/habitat-data/scene_datasets/hssd-hab", help="Directory containing hssd-hab dataset.")
    parser.add_argument("--scene-glob", default="**/*.scene_instance.json")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--total-tasks", type=int, default=1)
    parser.add_argument("--restart-every", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--success-prefix", default="", help="Prefix for success files (e.g., scriptname_jobid)")
    args = parser.parse_args()

    output_root = pathlib.Path(get_dataset_path(args.dataset_name))
    scene_files = collect_scene_files(args.scene_dir, args.scene_glob)[:1]
    print(f"Found {len(scene_files)} scene files under {args.scene_dir}")
    scene_files.sort(key=lambda x: hashlib.md5((str(x) + "potato").encode()).hexdigest())

    og.launch()

    errors_dir = output_root / "errors"
    jobs_dir = output_root / "jobs"
    ensure_dir(errors_dir)
    ensure_dir(jobs_dir)
    processed = 0
    for scene_path in scene_files:
        scene_name = normalize_scene_name(scene_path)
        if not should_process(scene_name, args.task_id, args.total_tasks):
            continue

        output_scene_dir = output_root / "scenes" / scene_name
        json_dir = output_scene_dir / "json"
        layout_dir = output_scene_dir / "layout"
        json_path = json_dir / f"{scene_name}_best.json"
        success_path = output_scene_dir / "import.success"

        if json_path.exists() and not args.overwrite:
            print(f"Skipping {scene_name}: output exists")
            continue

        print(f"Processing {scene_name}")
        try:
            ensure_dir(json_dir)
            ensure_dir(layout_dir)

            load_habitat_scene(args.dataset_name, str(scene_path))
            og.sim.step()
            og.sim.save(json_paths=[str(json_path)])

            map_start = time.time()
            generate_maps_for_current_scene(str(layout_dir))
            map_end = time.time()
            print(f"Generated maps in {map_end - map_start:.2f}s")

            success_path.touch()
            processed += 1
        except Exception:
            exc = traceback.format_exc()
            print(f"Error processing {scene_name}:\n{exc}")
            write_error(errors_dir, scene_name, exc)
        finally:
            og.clear()

        if args.restart_every and processed >= args.restart_every:
            break

    og.shutdown()
    print(f"Processed {processed} scenes")
    success_filename = (
        f"{args.success_prefix}_{args.task_id}.success" if args.success_prefix else f"{args.task_id}.success"
    )
    (jobs_dir / success_filename).touch()


if __name__ == "__main__":
    main()
