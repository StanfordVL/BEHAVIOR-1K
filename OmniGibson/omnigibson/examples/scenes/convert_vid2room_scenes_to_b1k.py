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

import sys

sys.path.append(str(pathlib.Path(__file__).parents[4] / "asset_pipeline"))

# Set OmniGibson macros before importing og
gm.HEADLESS = True
gm.ENABLE_FLATCACHE = False
gm.USE_GPU_DYNAMICS = False
gm.USE_ENCRYPTED_ASSETS = True

import omnigibson as og
from omnigibson.utils.asset_utils import get_dataset_path

from omnigibson.examples.scenes.load_vid2room_scene import get_scene_id, load_vid2room_scene
from b1k_pipeline.usd_conversion.make_maps import generate_maps_for_current_scene

DEFAULT_INTERESTING_SCENES_JSON = "/home/cgokmen/projects/BEHAVIOR-1K/slurm/interesting_scenes.json"


def should_process(room_dir, task_id, total_tasks, seed="potato"):
    if total_tasks <= 1:
        return True
    digest = hashlib.md5(f"{room_dir}-{seed}".encode()).hexdigest()
    return int(digest, 16) % total_tasks == task_id


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def write_error(error_dir, scene_name, exc):
    ensure_dir(error_dir)
    error_path = error_dir / f"{scene_name}.txt"
    error_path.write_text(str(exc))


def iter_vid2room_scenes(scene_list):
    with open(scene_list, "r") as f:
        room_dirs = json.load(f)

    # Convert to pathlib paths
    room_dirs = [pathlib.Path(k) for k in room_dirs]
    # room_dirs = [pathlib.Path("/checkpoint/clear/cgokmen/vid2room/RealEstate10K/vid_1vdXN7X4Af4/rooms/living_room_0")]

    # Exclude bathrooms
    room_dirs = [x for x in room_dirs if "bathroom" not in str(x)]

    # Filter to only rooms where floorplan.success exists (floorplan generation is complete)
    room_dirs = [x for x in room_dirs if (x / "floorplan.success").exists()]

    return room_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-list",
        type=str,
        default=DEFAULT_INTERESTING_SCENES_JSON,
        help="Path to JSON file with list of scene directories to process",
    )
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--total-tasks", type=int, default=1)
    parser.add_argument("--restart-every", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--success-prefix", default="", help="Prefix for success files (e.g., scriptname_jobid)")
    args = parser.parse_args()

    output_root = pathlib.Path(get_dataset_path("vid2room"))

    og.launch()

    errors_dir = output_root / "errors"
    jobs_dir = output_root / "jobs"
    ensure_dir(errors_dir)
    ensure_dir(jobs_dir)
    processed = 0
    for room_dir in iter_vid2room_scenes(args.scene_list):
        if not should_process(room_dir, args.task_id, args.total_tasks):
            continue

        scene_id = get_scene_id(room_dir)
        scene_success_file = output_root / "objects" / "vid2room_structures" / f"{scene_id}.success"
        if not scene_success_file.exists():
            print(f"Skipping {room_dir}: structures not imported")
            continue

        output_scene_dir = output_root / "scenes" / scene_id
        json_dir = output_scene_dir / "json"
        layout_dir = output_scene_dir / "layout"
        json_path = json_dir / f"{scene_id}_best.json"
        success_path = output_scene_dir / "import.success"

        if json_path.exists() and not args.overwrite:
            print(f"Skipping {scene_id}: output exists")
            continue

        print(f"Processing {scene_id}")
        try:
            ensure_dir(json_dir)
            ensure_dir(layout_dir)

            load_vid2room_scene(room_dir)
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
            print(f"Error processing {scene_id}:\n{exc}")
            write_error(errors_dir, scene_id, exc)
        finally:
            og.clear()

        if args.restart_every and processed >= args.restart_every:
            break

    print(f"Processed {processed} scenes")
    success_filename = (
        f"{args.success_prefix}_{args.task_id}.success" if args.success_prefix else f"{args.task_id}.success"
    )
    (jobs_dir / success_filename).touch()
    og.shutdown()


if __name__ == "__main__":
    main()
