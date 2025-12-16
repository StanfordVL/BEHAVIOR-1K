import json
import math
import os
import random
import subprocess
from dask.distributed import Client, as_completed
import fs.copy
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
        # Copy everything over to the dataset FS
        print("Copying input to dataset fs...")
        fs.copy.copy_fs(metadata_fs, dataset_fs)
        fs.copy.copy_fs(systems_fs, dataset_fs)
        fs.copy.copy_fs(objects_fs, dataset_fs)
        # fs.copy.copy_fs(objects_fs.opendir("objects/sink"), dataset_fs.makedirs("objects/sink"))

        print("Launching cluster...")
        dask_client = launch_cluster(WORKER_COUNT)

        # Start the batched run
        object_glob = [x.path for x in dataset_fs.glob("objects/*/*/")]
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
