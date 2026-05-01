import json
import math
import traceback

from concurrent.futures import as_completed
import fs.copy
import fs.path
from fs.tempfs import TempFS
from fs.zipfs import ZipFS
import tqdm

from b1k_pipeline.utils import (
    PipelineFS,
    TMP_DIR,
    launch_cluster,
    submit_og_task,
)

WORKER_COUNT = 1

MAX_POS_DELTA = 0.05  # 5cm
MAX_ORN_DELTA = math.radians(5)  # 5 degrees
MAX_LINEAR_VEL = 0.01  # 0.01 m/s
MAX_ANGULAR_VEL = math.radians(1)  # 1 degree/s

OG_MACROS = {
    "HEADLESS": True,
    "USE_GPU_DYNAMICS": False,
    "USE_ENCRYPTED_ASSETS": True,
}


def process_scene(dataset_root, scene):
    """Validate that the scene is at rest after a few hundred sim steps."""
    import torch as th

    import omnigibson as og
    from omnigibson.macros import gm
    import omnigibson.utils.transform_utils as T

    gm.DATASET_PATH = str(dataset_root)

    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene,
        },
    }

    env = og.Environment(configs=cfg)

    objs = env.scene.objects
    links = {
        obj.name + "-" + link_name: link
        for obj in objs
        for link_name, link in obj.links.items()
    }

    initial_poses = {
        link_name: link.get_position_orientation() for link_name, link in links.items()
    }

    print("Stepping simulation.")
    for _ in range(300):
        env.step([])
    print("Done stepping simulation.")

    mismatches = []
    for link_name, link in links.items():
        old_pos, old_orn = initial_poses[link_name]
        new_pos, new_orn = link.get_position_orientation()
        lin_vel = link.get_linear_velocity()
        ang_vel = link.get_angular_velocity()

        delta_pos = th.linalg.norm(new_pos - old_pos).item()
        if delta_pos > MAX_POS_DELTA:
            mismatches.append(
                f"{link_name} position changed by {delta_pos} meters from {old_pos} to {new_pos}."
            )
        delta_orn_mag = T.get_orientation_diff_in_radian(old_orn, new_orn)
        if delta_orn_mag > MAX_ORN_DELTA:
            mismatches.append(
                f"{link_name} orientation changed by {delta_orn_mag} rads from {old_orn} to {new_orn}."
            )
        if th.any(th.abs(lin_vel) > MAX_LINEAR_VEL):
            mismatches.append(f"{link_name} linear velocity is {lin_vel}.")
        if th.any(th.abs(ang_vel) > MAX_ANGULAR_VEL):
            mismatches.append(f"{link_name} angular velocity is {ang_vel}.")

    return mismatches


def main():
    with (
        PipelineFS() as pipeline_fs,
        pipeline_fs.open("artifacts/og_dataset.zip", "rb") as og_dataset_zip,
        ZipFS(og_dataset_zip) as objects_fs,
        TempFS(temp_dir=str(TMP_DIR)) as dataset_fs,
    ):
        # Copy everything over to the dataset FS
        print("Copying input to dataset fs...")

        total_files = sum(1 for f in objects_fs.walk.files())
        with tqdm.tqdm(total=total_files) as pbar:
            fs.copy.copy_fs(
                objects_fs, dataset_fs, on_copy=lambda *args: pbar.update(1)
            )

        print("Launching cluster...")
        executor = launch_cluster(WORKER_COUNT, og_macros=OG_MACROS)

        # Start the batched run
        scenes = list(dataset_fs.opendir("scenes").listdir("/"))
        print("Queueing scenes.")
        print("Total count: ", len(scenes))
        futures = {}
        for scene in scenes:
            worker_future = submit_og_task(
                executor,
                process_scene,
                dataset_fs.getsyspath("/"),
                scene,
            )
            futures[worker_future] = scene

        print("Queued all scenes. Waiting for them to finish...")
        scene_results = {}
        for future in tqdm.tqdm(as_completed(futures.keys()), total=len(futures)):
            scene = futures[future]
            scene_results[scene] = {"success": False, "issues": [], "logs": ""}
            try:
                scene_results[scene]["issues"] = future.result()
                scene_results[scene]["success"] = not scene_results[scene]["issues"]
            except Exception:
                scene_results[scene]["logs"] = traceback.format_exc()

        results = {
            "success": all([x["success"] for x in scene_results.values()]),
            "scenes": scene_results,
        }
        with pipeline_fs.pipeline_output().open("validate_scenes.json", "w") as f:
            json.dump(results, f, indent=4)

        pipeline_fs.pipeline_output().touch("usdify_scenes.success")


if __name__ == "__main__":
    main()
