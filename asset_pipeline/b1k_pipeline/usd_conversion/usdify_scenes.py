import json
import os
import signal
import subprocess
import pathlib
import traceback
from dask.distributed import Client, as_completed
import fs.copy
from fs.multifs import MultiFS
from fs.tempfs import TempFS
import tqdm

from b1k_pipeline.utils import ParallelZipFS, PipelineFS, TMP_DIR, launch_cluster

WORKER_COUNT = 2
MAX_TIME_PER_PROCESS = 60 * 60  # 1 hour


def run_on_scene(dataset_path, scene):
    try:
        basename = pathlib.Path(scene).stem
        print("Running on scene:", basename)
        python_cmd = [
            "python",
            "-m",
            "b1k_pipeline.usd_conversion.usdify_scenes_process",
            dataset_path,
            scene,
        ]
        cmd = [
            "micromamba",
            "run",
            "-n",
            "omnigibson",
            "/bin/bash",
            "-c",
            "source /isaac-sim/setup_conda_env.sh && rm -rf /root/.cache/ov/texturecache && "
            + " ".join(python_cmd),
        ]
        with (
            open(f"/scr/BEHAVIOR-1K/asset_pipeline/logs/{basename}.log", "w") as f,
            open(f"/scr/BEHAVIOR-1K/asset_pipeline/logs/{basename}.err", "w") as ferr,
        ):
            try:
                p = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=ferr,
                    cwd="/scr/BEHAVIOR-1K/asset_pipeline",
                    start_new_session=True,
                )
                pid = p.pid
                p.wait(timeout=MAX_TIME_PER_PROCESS)
            except subprocess.TimeoutExpired as e:
                ferr.write(
                    f"\nTimeout for {basename} ({MAX_TIME_PER_PROCESS}s) expired. Killing\n"
                )
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except ProcessLookupError:
                    ferr.write(f"Process {pid} already exited.\n")
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
            dask_client = launch_cluster(WORKER_COUNT)

            # Start the batched run. We remove the leading / so that pathlib can append it to dataset path correctly.
            scenes = [x.path[1:] for x in dataset_fs.glob("scenes/*/urdf/*.urdf")]
            print("Queueing scenes.")
            print("Total count: ", len(scenes))
            futures = {}
            for scene in scenes:
                worker_future = dask_client.submit(
                    run_on_scene,
                    dataset_fs.getsyspath("/"),
                    scene,
                    # retries=2,
                    pure=False,
                )
                futures[worker_future] = scene

            # Wait for all the workers to finish
            print("Queued all scenes. Waiting for them to finish...")
            errors = {}
            for future in tqdm.tqdm(as_completed(futures.keys()), total=len(futures)):
                exc = future.result()
                if exc:
                    errors[futures[future]] = str(exc)

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
