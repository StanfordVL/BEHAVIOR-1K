import json
import os
import signal
import subprocess
import sys
from dask.distributed import Client, as_completed
import fs.copy
from fs.multifs import MultiFS
import fs.path
from fs.tempfs import TempFS
import tqdm

from b1k_pipeline.utils import ParallelZipFS, PipelineFS, TMP_DIR, launch_cluster

WORKER_COUNT = 1
MAX_TIME_PER_PROCESS = 60 * 60  # 1 hour

def run_on_scene(dataset_path, scene):
    python_cmd = ["python", "-m", "b1k_pipeline.usd_conversion.usdify_scenes_process", dataset_path, scene]
    cmd = ["micromamba", "run", "-n", "omnigibson", "/bin/bash", "-c", "source /isaac-sim/setup_conda_env.sh && rm -rf /root/.cache/ov/texturecache && " + " ".join(python_cmd)]
    with open(f"/scr/BEHAVIOR-1K/asset_pipeline/logs/{scene}.log", "w") as f, open(f"/scr/BEHAVIOR-1K/asset_pipeline/logs/{scene}.err", "w") as ferr:
        try:
            p = subprocess.Popen(cmd, stdout=f, stderr=ferr, cwd="/scr/BEHAVIOR-1K/asset_pipeline", start_new_session=True)
            p.wait(timeout=MAX_TIME_PER_PROCESS)
        except subprocess.TimeoutExpired:
            ferr.write(f'\nTimeout for {scene} ({MAX_TIME_PER_PROCESS}s) expired. Killing\n')
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            p.wait()

    # Check if the success file exists.
    if not os.path.exists(f"{dataset_path}/scenes/{scene}/usdify_scenes.success"):
        raise ValueError(f"Scene {scene} processing failed: no success file found. Check the logs.")

def main():
    with PipelineFS() as pipeline_fs, \
         ParallelZipFS("objects_usd.zip") as objects_fs, \
         ParallelZipFS("metadata.zip") as metadata_fs, \
         ParallelZipFS("scenes.zip") as scenes_fs, \
         TempFS(temp_dir=str(TMP_DIR)) as dataset_fs:
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
                fs.copy.copy_fs(multi_fs, dataset_fs, on_copy=lambda *args: pbar.update(1))

            print("Launching cluster...")
            dask_client = launch_cluster(WORKER_COUNT)

            # Start the batched run
            scenes = [x for x in dataset_fs.listdir("scenes")]
            print("Queueing scenes.")
            print("Total count: ", len(scenes))
            futures = {}
            for scene in scenes:
                worker_future = dask_client.submit(
                    run_on_scene,
                    dataset_fs.getsyspath("/"),
                    scene,
                    # retries=2,
                    pure=False)
                futures[worker_future] = scene

            # Wait for all the workers to finish
            print("Queued all scenes. Waiting for them to finish...")
            errors = {}
            for future in tqdm.tqdm(as_completed(futures.keys()), total=len(futures)):
                try:
                    future.result()
                except Exception as e:
                    raise
                    errors[futures[future]] = str(e)

            # Move the USDs to the output FS
            print("Copying scene JSONs to output FS...")
            usd_glob = sorted(
                {x.path for x in dataset_fs.glob("scenes/*/json/")} |
                {x.path for x in dataset_fs.glob("scenes/*/layout/")})
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
