import os
import signal
import subprocess
import sys
from dask.distributed import as_completed

import fs.path
from fs.copy import copy_fs
from fs.tempfs import TempFS
from fs.osfs import OSFS
from fs.zipfs import ZipFS

from b1k_pipeline.utils import PipelineFS, TMP_DIR, launch_cluster

import tqdm


BATCH_SIZE = 8
assert BATCH_SIZE % 2 == 0
WORKER_COUNT = 6
MAX_TIME_PER_PROCESS = 5 * 60  # 5 minutes


def run_on_batch(dataset_path, out_path, batch):
    python_cmd = [
        "python",
        "-m",
        "b1k_pipeline.generate_object_images_og",
        dataset_path,
        out_path,
    ] + batch
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
    obj = batch[0].split("/")[-1]
    with (
        open(f"/scr/BEHAVIOR-1K/asset_pipeline/logs/{obj}.log", "w") as f,
        open(f"/scr/BEHAVIOR-1K/asset_pipeline/logs/{obj}.err", "w") as ferr,
    ):
        try:
            p = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=ferr,
                cwd="/scr/BEHAVIOR-1K/asset_pipeline",
                start_new_session=True,
            )
            return p.wait(timeout=MAX_TIME_PER_PROCESS)
        except subprocess.TimeoutExpired:
            print(
                f"Timeout for {batch} ({MAX_TIME_PER_PROCESS}s) expired. Killing",
                file=sys.stderr,
            )
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            return p.wait()


def main():
    with PipelineFS() as pipeline_fs:
        with (
            ZipFS(pipeline_fs.open("artifacts/og_dataset.zip", "rb")) as dataset_zip_fs,
            TempFS(temp_dir=str(TMP_DIR)) as dataset_fs,
            OSFS(
                pipeline_fs.makedirs(
                    "artifacts/pipeline/object_images", recreate=True
                ).getsyspath("/")
            ) as out_temp_fs,
        ):
            # Copy everything over to the dataset FS
            print("Copy everything over to the dataset FS...")
            objdir_glob = list(dataset_zip_fs.glob("objects/*/*/"))
            for item in tqdm.tqdm(objdir_glob):
                if (
                    dataset_zip_fs.opendir(item.path).glob("usd/*.usd").count().files
                    == 0
                ):
                    continue
                objdir_normalized = fs.path.normpath(item.path)
                obj_id = fs.path.basename(objdir_normalized)
                if out_temp_fs.exists(f"{obj_id}.success"):
                    continue
                copy_fs(
                    dataset_zip_fs.opendir(item.path), dataset_fs.makedirs(item.path)
                )

            # Launch the cluster
            dask_client = launch_cluster(WORKER_COUNT)

            # Start the batched run
            object_glob = [
                fs.path.normpath(x.path) for x in dataset_fs.glob("objects/*/*/")
            ]

            print("Queueing batches.")
            print("Total count: ", len(object_glob))
            futures = {}
            batch_size = min(BATCH_SIZE, len(object_glob) // WORKER_COUNT)
            for start in range(0, len(object_glob), batch_size):
                end = start + batch_size
                batch = object_glob[start:end]
                if batch:
                    worker_future = dask_client.submit(
                        run_on_batch,
                        dataset_fs.getsyspath("/"),
                        out_temp_fs.getsyspath("/"),
                        batch,
                        pure=False,
                    )
                    futures[worker_future] = batch

            # Wait for all the workers to finish
            print("Queued all batches. Waiting for them to finish...")
            while True:
                for future in tqdm.tqdm(
                    as_completed(futures.keys()), total=len(futures)
                ):
                    # Check the batch results.
                    batch = futures[future]
                    return_code = future.result()  # we dont use the return code since we check the output files directly

                    # Remove everything that failed and make a new batch from them.
                    new_batch = []
                    for item in batch:
                        item_basename = fs.path.basename(item)
                        expected_output = f"{item_basename}.success"
                        if not out_temp_fs.exists(expected_output):
                            print("Could not find", expected_output)
                            new_batch.append(item)

                    # If there's nothing to requeue, we are good!
                    if not new_batch:
                        continue

                    # Otherwise, decide if we are going to requeue or just skip.
                    if len(batch) == 1:
                        print(f"Failed on a single item {batch[0]}. Skipping.")
                    else:
                        print(f"Subdividing batch of length {len(new_batch)}")
                        batch_size = len(new_batch) // 2
                        subbatches = [new_batch[:batch_size], new_batch[batch_size:]]
                        for subbatch in subbatches:
                            if not subbatch:
                                continue
                            worker_future = dask_client.submit(
                                run_on_batch,
                                dataset_fs.getsyspath("/"),
                                subbatch,
                                pure=False,
                            )
                            futures[worker_future] = subbatch
                        del futures[future]

                        # Restart the for loop so that the counter can update
                        break
                else:
                    # Completed successfully - break out of the while loop.
                    break

        print("Archiving results...")

        # At this point, out_temp_fs's contents will be zipped. Save the success file.
        pipeline_fs.pipeline_output().touch("generate_images.success")


if __name__ == "__main__":
    main()
