import json
import pathlib
import traceback

from concurrent.futures import as_completed
import fs.copy
from fs.multifs import MultiFS
from fs.tempfs import TempFS
import tqdm

from b1k_pipeline.utils import (
    ParallelZipFS,
    PipelineFS,
    TMP_DIR,
    launch_cluster,
    submit_og_task,
)

WORKER_COUNT = 2

OG_MACROS = {
    "HEADLESS": True,
    "USE_GPU_DYNAMICS": False,
    "USE_ENCRYPTED_ASSETS": True,
}


def process_scene(dataset_root, scene):
    """Convert a scene URDF to JSON and (for *_best.urdf) generate maps."""
    import time

    from omnigibson.macros import gm
    from omnigibson.utils.asset_conversion_utils import convert_scene_urdf_to_json
    from b1k_pipeline.usd_conversion.make_maps import generate_maps_for_current_scene

    gm.DATASET_PATH = str(dataset_root)

    urdf_path = pathlib.Path(dataset_root) / scene
    scene_basename = urdf_path.stem
    json_path = urdf_path.parent.parent / "json" / f"{scene_basename}.json"

    convert_scene_urdf_to_json(urdf=str(urdf_path), json_path=str(json_path))

    if urdf_path.name.endswith("_best.urdf"):
        print("Starting map generation")
        map_start = time.time()
        save_path = urdf_path.parent.parent / "layout"
        generate_maps_for_current_scene(str(save_path))
        map_end = time.time()
        print("Generated maps in ", map_end - map_start, "seconds")


def main():
    with (
        PipelineFS() as pipeline_fs,
        ParallelZipFS("objects_usd.zip") as objects_fs,
        ParallelZipFS("metadata.zip") as metadata_fs,
        ParallelZipFS("scenes.zip") as scenes_fs,
        TempFS(temp_dir=str(TMP_DIR)) as dataset_fs,
    ):
        with ParallelZipFS("scenes_json.zip", write=True) as out_fs:
            # Copy everything over to the dataset FS
            print("Copying input to dataset fs...")
            multi_fs = MultiFS()
            multi_fs.add_fs("metadata", metadata_fs, priority=1)
            multi_fs.add_fs("objects", objects_fs, priority=1)
            multi_fs.add_fs("scenes", scenes_fs, priority=1)

            # Copy all the files to the output zip filesystem.
            total_files = sum(1 for f in multi_fs.walk.files())
            with tqdm.tqdm(total=total_files) as pbar:
                fs.copy.copy_fs(
                    multi_fs, dataset_fs, on_copy=lambda *args: pbar.update(1)
                )

            print("Launching cluster...")
            executor = launch_cluster(WORKER_COUNT, og_macros=OG_MACROS)

            # Start the batched run. We remove the leading / so that pathlib can append it to dataset path correctly.
            scenes = [x.path[1:] for x in dataset_fs.glob("scenes/*/urdf/*.urdf")]
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

            # Wait for all the workers to finish
            print("Queued all scenes. Waiting for them to finish...")
            errors = {}
            for future in tqdm.tqdm(as_completed(futures.keys()), total=len(futures)):
                try:
                    future.result()
                except Exception:
                    errors[futures[future]] = traceback.format_exc()

            # Move the USDs to the output FS
            print("Copying scene JSONs to output FS...")
            usd_glob = sorted(
                {x.path for x in dataset_fs.glob("scenes/*/json/")}
                | {x.path for x in dataset_fs.glob("scenes/*/layout/")}
            )
            for item in tqdm.tqdm(usd_glob):
                fs.copy.copy_fs(dataset_fs.opendir(item), out_fs.makedirs(item))

            print("Done processing. Archiving things now.")

        # Save the logs
        success = len(errors) == 0
        with pipeline_fs.pipeline_output().open("usdify_scenes.json", "w") as f:
            json.dump({"success": success, "errors": errors}, f)

        # At this point, out_temp_fs's contents will be zipped. Save the success file.
        if success:
            pipeline_fs.pipeline_output().touch("usdify_scenes.success")


if __name__ == "__main__":
    main()
