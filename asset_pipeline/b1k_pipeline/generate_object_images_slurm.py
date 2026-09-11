import os
import signal
import subprocess
import sys
from concurrent.futures import as_completed

import fs.path
from fs.copy import copy_fs
from fs.tempfs import TempFS
from fs.osfs import OSFS
from fs.zipfs import ZipFS

from b1k_pipeline.utils import (
    PipelineFS,
    TMP_DIR,
    make_og_pool_executor,
    worker_subprocess_env,
)

import tqdm


BATCH_SIZE = 8
assert BATCH_SIZE % 2 == 0
WORKER_COUNT = 6
MAX_TIME_PER_PROCESS = 5 * 60  # 5 minutes


def run_on_batch(dataset_path, out_path, batch):
    cmd = [
        sys.executable,
        "-m",
        "b1k_pipeline.generate_object_images_og",
        dataset_path,
        out_path,
    ] + batch
    obj = batch[0].split("/")[-1]
    os.makedirs("/scr/BEHAVIOR-1K/asset_pipeline/logs", exist_ok=True)
    with (
        open(f"/scr/BEHAVIOR-1K/asset_pipeline/logs/{obj}.log", "w") as f,
        open(f"/scr/BEHAVIOR-1K/asset_pipeline/logs/{obj}.err", "w") as ferr,
    ):
        try:
            # process_group=0 puts the subprocess in its OWN process group
            # (so os.killpg below only reaches the subprocess and its
            # descendants — not the pool worker that called us) but leaves
            # it in the parent's session, so a terminal SIGHUP /
            # head-process crash still tears it down.
            p = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=ferr,
                cwd="/scr/BEHAVIOR-1K/asset_pipeline",
                process_group=0,
                env=worker_subprocess_env(),
            )
            return p.wait(timeout=MAX_TIME_PER_PROCESS)
        except subprocess.TimeoutExpired:
            print(
                f"Timeout for {batch} ({MAX_TIME_PER_PROCESS}s) expired. Killing",
                file=sys.stderr,
            )
            os.killpg(p.pid, signal.SIGKILL)
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
            print("Launching cluster...")
            with make_og_pool_executor(WORKER_COUNT) as executor:
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
                        worker_future = executor.submit(
                            run_on_batch,
                            dataset_fs.getsyspath("/"),
                            out_temp_fs.getsyspath("/"),
                            batch,
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
                                worker_future = executor.submit(
                                    run_on_batch,
                                    dataset_fs.getsyspath("/"),
                                    out_temp_fs.getsyspath("/"),
                                    subbatch,
                                )
                                futures[worker_future] = subbatch
                            del futures[future]

                            # Restart the for loop so that the counter can update
                            break
                    else:
                        # Completed successfully - break out of the while loop.
                        break

                print("Finished processing. Shutting down executor...")

        print("Archiving results...")

        # At this point, out_temp_fs's contents will be zipped. Save the success file.
        pipeline_fs.pipeline_output().touch("generate_images.success")


if __name__ == "__main__":
    main()
