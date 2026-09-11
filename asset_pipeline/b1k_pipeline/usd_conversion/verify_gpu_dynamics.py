import json
import math
import os
import random
import subprocess
from dask.distributed import Client, as_completed
import fs.copy
from fs.osfs import OSFS
import fs.path
from fs.tempfs import TempFS
import tqdm

from b1k_pipeline.utils import ParallelZipFS, TMP_DIR, launch_cluster

WORKER_COUNT = 4


def run_on_batch(dataset_path, path):
    python_cmd = [
        "python",
        "-m",
        "b1k_pipeline.usd_conversion.verify_gpu_dynamics_process",
        dataset_path,
        path,
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
    obj = path[:-1].split("/")[-1]
    with (
        open(f"/scr/BEHAVIOR-1K/asset_pipeline/logs/{obj}.log", "w") as f,
        open(f"/scr/BEHAVIOR-1K/asset_pipeline/logs/{obj}.err", "w") as ferr,
    ):
        return subprocess.run(
            cmd,
            stdout=f,
            stderr=ferr,
            check=True,
            cwd="/scr/BEHAVIOR-1K/asset_pipeline",
        )


def main():
    with (
        ParallelZipFS("objects_usd.zip") as objects_fs,
        ParallelZipFS("metadata.zip") as metadata_fs,
        ParallelZipFS("systems.zip") as systems_fs,
        TempFS(temp_dir=str(TMP_DIR)) as dataset_fs,
    ):
        # Copy everything over to the dataset FS, under a behavior-1k-assets/ subdir
        # so that OmniGibson's get_dataset_path("behavior-1k-assets") (which resolves
        # to gm.DATA_PATH/behavior-1k-assets) finds the assets.
        print("Copying input to dataset fs...")
        staged_fs = dataset_fs.makedirs("behavior-1k-assets")
        fs.copy.copy_fs(metadata_fs, staged_fs)
        fs.copy.copy_fs(systems_fs, staged_fs)
        fs.copy.copy_fs(objects_fs, staged_fs)

        # Copy omnigibson-robot-assets/ and the key alongside the dataset (at gm.DATA_PATH root).
        fs.copy.copy_fs(
            OSFS("/scr/BEHAVIOR-1K/datasets/omnigibson-robot-assets"),
            dataset_fs.makedirs("omnigibson-robot-assets"),
        )
        with open("/scr/BEHAVIOR-1K/datasets/omnigibson.key", "rb") as f:
            dataset_fs.writefile("omnigibson.key", f)

        print("Launching cluster...")
        dask_client = launch_cluster(WORKER_COUNT)

        # Start the batched run
        object_glob = [x.path for x in staged_fs.glob("objects/*/*/")]
        print("Queueing batches.")
        print("Total count: ", len(object_glob))

        futures = {}
        for path in object_glob:
            worker_future = dask_client.submit(
                run_on_batch, dataset_fs.getsyspath("/"), path, pure=False
            )
            futures[worker_future] = path

        # Wait for all the workers to finish
        print("Queued all batches. Waiting for them to finish...")
        for future in tqdm.tqdm(as_completed(futures.keys()), total=len(futures)):
            # Check the batch results.
            path = futures[future]
            obj = path[:-1].split("/")[-1]
            if future.exception():
                print(f"Exception in {futures[future]}: {future.exception()}")
                with open(
                    f"/scr/BEHAVIOR-1K/asset_pipeline/logs/{obj}.exception", "w"
                ) as f:
                    f.write(str(future.exception()))
            else:
                with open(
                    f"/scr/BEHAVIOR-1K/asset_pipeline/logs/{obj}.success", "w"
                ) as f:
                    pass


if __name__ == "__main__":
    main()
