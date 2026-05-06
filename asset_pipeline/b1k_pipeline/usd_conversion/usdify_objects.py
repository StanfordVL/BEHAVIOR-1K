import json
import math
import os
import signal
import subprocess
import sys
import fs.copy
from concurrent.futures import as_completed
import fs.path
from fs.tempfs import TempFS
import tqdm

from b1k_pipeline.utils import (
    ParallelZipFS,
    PipelineFS,
    TMP_DIR,
    make_og_pool_executor,
    worker_subprocess_env,
)

WORKER_COUNT = 6
BATCH_SIZE = 64
MAX_TIME_PER_PROCESS = 10 * 60  # 10 minutes


def run_on_batch(dataset_path, batch):
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "b1k_pipeline.usd_conversion.usdify_objects_process",
        dataset_path,
    ] + batch
    obj = batch[0][:-1].split("/")[-1]
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
    failed_objects = set()
    with (
        PipelineFS() as pipeline_fs,
        ParallelZipFS("objects.zip") as objects_fs,
        TempFS(temp_dir=str(TMP_DIR)) as dataset_fs,
    ):
        with ParallelZipFS("objects_usd.zip", write=True) as out_fs:
            # Copy everything over to the dataset FS
            print("Copying input to dataset fs...")
            objdir_glob = list(objects_fs.glob("objects/*/*/"))
            for item in tqdm.tqdm(objdir_glob):
                if (
                    objects_fs.opendir(item.path)
                    .opendir("urdf")
                    .glob("*.urdf")
                    .count()
                    .files
                    == 0
                ):
                    continue
                fs.copy.copy_fs(
                    objects_fs.opendir(item.path),
                    dataset_fs.makedirs(item.path, recreate=True),
                )

            print("Launching cluster...")

            with make_og_pool_executor(WORKER_COUNT) as executor:
                # Start the batched run
                object_glob = [x.path for x in dataset_fs.glob("objects/*/*/")]
                print("Queueing batches.")
                print("Total count: ", len(object_glob))

                # Make sure workers don't idle by reducing batch size when possible.
                batch_size = min(BATCH_SIZE, math.ceil(len(object_glob) / WORKER_COUNT))

                futures = {}
                for start in range(0, len(object_glob), batch_size):
                    end = start + batch_size
                    batch = object_glob[start:end]
                    worker_future = executor.submit(
                        run_on_batch, dataset_fs.getsyspath("/"), batch
                    )
                    futures[worker_future] = batch

                # Wait for all the workers to finish
                print("Queued all batches. Waiting for them to finish...")
                logs = []
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
                            item_dir = dataset_fs.opendir(item)
                            if item_dir.glob("usd/*.encrypted.usd").count().files != 1:
                                print("Could not find", item)
                                print("Available items:", list(item_dir.walk.files()))
                                new_batch.append(item)
                                if item_dir.exists("usd"):
                                    item_dir.removetree("usd")

                        # If there's nothing to requeue, we are good!
                        if not new_batch:
                            continue

                        # Otherwise, decide if we are going to requeue or just skip.
                        if len(batch) == 1:
                            print(f"Failed on a single item {batch[0]}. Skipping.")
                            failed_objects.add(batch[0])
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

            # Move the USDs to the output FS
            print("Copying USDs to output FS...")
            usd_glob = [
                x.path for x in dataset_fs.glob("objects/*/*/usd/*.usd")
            ]
            for item in tqdm.tqdm(usd_glob):
                itemdir = fs.path.dirname(item)
                fs.copy.copy_fs(dataset_fs.opendir(itemdir), out_fs.makedirs(itemdir))

            print("Done processing. Archiving things now.")

        # Save the logs
        with pipeline_fs.pipeline_output().open("usdify_objects.json", "w") as f:
            json.dump(
                {
                    "success": len(failed_objects) == 0,
                    "failed_objects": sorted(failed_objects),
                    "logs": logs,
                },
                f,
            )


if __name__ == "__main__":
    main()
