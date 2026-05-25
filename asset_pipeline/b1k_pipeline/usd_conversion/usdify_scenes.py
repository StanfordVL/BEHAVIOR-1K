import json
import os
import pathlib
import signal
import subprocess
import sys
import traceback
from concurrent.futures import as_completed
import fs.copy
from fs.multifs import MultiFS
from fs.osfs import OSFS
from fs.tempfs import TempFS
import tqdm

from b1k_pipeline.utils import (
    ParallelZipFS,
    PipelineFS,
    TMP_DIR,
    make_og_pool_executor,
    worker_subprocess_env,
)

WORKER_COUNT = 2
MAX_TIME_PER_PROCESS = 60 * 60  # 1 hour


def run_on_scene(dataset_path, scene):
    try:
        basename = pathlib.Path(scene).stem
        print("Running on scene:", basename)
        cmd = [
            sys.executable,
            "-m",
            "b1k_pipeline.usd_conversion.usdify_scenes_process",
            dataset_path,
            scene,
        ]
        os.makedirs("/scr/BEHAVIOR-1K/asset_pipeline/logs", exist_ok=True)
        with (
            open(f"/scr/BEHAVIOR-1K/asset_pipeline/logs/{basename}.log", "w") as f,
            open(f"/scr/BEHAVIOR-1K/asset_pipeline/logs/{basename}.err", "w") as ferr,
        ):
            try:
                # process_group=0 puts the subprocess in its OWN process
                # group (so os.killpg below only reaches the subprocess and
                # its descendants — not the pool worker that called us) but
                # leaves it in the parent's session, so a terminal SIGHUP /
                # head-process crash still tears it down. Requires Python
                # 3.11+ (we're on 3.11.x).
                p = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=ferr,
                    cwd="/scr/BEHAVIOR-1K/asset_pipeline",
                    process_group=0,
                    env=worker_subprocess_env(),
                )
                p.wait(timeout=MAX_TIME_PER_PROCESS)
            except subprocess.TimeoutExpired:
                ferr.write(
                    f"\n{basename} did not finish within {MAX_TIME_PER_PROCESS}s. Killing\n"
                )
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    ferr.write(f"Process {p.pid} already exited.\n")
                p.wait()

        # Check if the success file exists.
        success_file = (pathlib.Path(dataset_path) / scene).with_suffix(".success")
        if not success_file.exists():
            raise ValueError(
                f"Scene {scene} processing failed: no success file found. Check the logs."
            )

        return None
    except:
        return traceback.format_exc()


def main():
    with (
        PipelineFS() as pipeline_fs,
        ParallelZipFS("objects_usd.zip") as objects_fs,
        ParallelZipFS("metadata.zip") as metadata_fs,
        ParallelZipFS("scenes.zip") as scenes_fs,
        ParallelZipFS("systems.zip") as systems_fs,
        TempFS(temp_dir=str(TMP_DIR)) as dataset_fs,
    ):
        with ParallelZipFS("scenes_json.zip", write=True) as out_fs:
            # Copy everything over to the dataset FS, under a behavior-1k-assets/ subdir
            # so that OmniGibson's get_dataset_path("behavior-1k-assets") (which resolves
            # to gm.DATA_PATH/behavior-1k-assets) finds the assets.
            print("Copying input to dataset fs...")
            multi_fs = MultiFS()
            multi_fs.add_fs("metadata", metadata_fs, priority=1)
            multi_fs.add_fs("objects", objects_fs, priority=1)
            multi_fs.add_fs("scenes", scenes_fs, priority=1)
            multi_fs.add_fs("systems", systems_fs, priority=1)

            dataset_subdir = "behavior-1k-assets"
            staged_fs = dataset_fs.makedirs(dataset_subdir)

            # Copy all the files to the output zip filesystem.
            total_files = sum(1 for f in multi_fs.walk.files())
            with tqdm.tqdm(total=total_files) as pbar:
                fs.copy.copy_fs(
                    multi_fs, staged_fs, on_copy=lambda *args: pbar.update(1)
                )

            # Copy omnigibson-robot-assets/ and the key
            fs.copy.copy_fs(
                OSFS("/scr/BEHAVIOR-1K/datasets/omnigibson-robot-assets"),
                dataset_fs.makedirs("omnigibson-robot-assets"),
            )
            with open("/scr/BEHAVIOR-1K/datasets/omnigibson.key", "rb") as f:
                dataset_fs.writefile("omnigibson.key", f)
            with open("/scr/BEHAVIOR-1K/asset_pipeline/VERSION", "rb") as f:
                staged_fs.writefile("VERSION", f)

            print("Launching cluster...")
            with make_og_pool_executor(WORKER_COUNT) as executor:
                # Start the batched run. We remove the leading / so that pathlib can append it to dataset path correctly.
                scenes = [x.path[1:] for x in dataset_fs.glob(f"{dataset_subdir}/scenes/*/urdf/*_best.urdf")]
                # scenes_to_process = {
                #     "hotel_suite_large",
                #     "restaurant_diner",
                #     "office_cubicles_right",
                # }
                # scenes = [
                #     x.path[1:]
                #     for x in dataset_fs.glob(f"{dataset_subdir}/scenes/*/urdf/*_best.urdf")
                #     if pathlib.Path(x.path).parts[3] in scenes_to_process
                # ]
                print("Queueing scenes.")
                print("Total count: ", len(scenes))
                futures = {}
                for scene in scenes:
                    worker_future = executor.submit(
                        run_on_scene,
                        dataset_fs.getsyspath("/"),
                        scene,
                    )
                    futures[worker_future] = scene

                # Wait for all the workers to finish
                print("Queued all scenes. Waiting for them to finish...")
                errors = {}
                for future in tqdm.tqdm(as_completed(futures.keys()), total=len(futures)):
                    exc = future.result()
                    if exc:
                        errors[futures[future]] = str(exc)

                print("Finished processing. Shutting down executor...")

            # Move the USDs to the output FS. Strip the dataset_subdir prefix so the
            # output zip preserves the original scenes/<scene>/{json,layout}/ layout.
            print("Copying scene JSONs to output FS...")
            usd_glob = sorted(
                {x.path for x in dataset_fs.glob(f"{dataset_subdir}/scenes/*/json/")}
                | {x.path for x in dataset_fs.glob(f"{dataset_subdir}/scenes/*/layout/")}
            )
            for item in tqdm.tqdm(usd_glob):
                out_item = item[len(f"/{dataset_subdir}"):]
                fs.copy.copy_fs(dataset_fs.opendir(item), out_fs.makedirs(out_item))

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
