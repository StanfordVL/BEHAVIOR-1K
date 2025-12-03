import json
import math
import os
import random
import subprocess
from dask.distributed import Client, as_completed
import fs.copy
from fs.multifs import MultiFS
import fs.path
from fs.tempfs import TempFS
import tqdm

from b1k_pipeline.utils import ParallelZipFS, PipelineFS, TMP_DIR, launch_cluster

WORKER_COUNT = 1
BATCH_SIZE = 32


def run_on_batch(dataset_path, batch, mode):
    if mode == "ray":
        script = "b1k_pipeline.usd_conversion.generate_fillable_volumes_process_ray"
    elif mode == "dip":
        script = "b1k_pipeline.usd_conversion.generate_fillable_volumes_process_dip"
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose either ray or dip.")
    python_cmd = ["python", "-m", script, dataset_path] + batch
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
    obj = batch[0][:-1].split("/")[-1]
    with (
        open(f"/scr/BEHAVIOR-1K/asset_pipeline/logs/{obj}-{mode}.log", "w") as f,
        open(f"/scr/BEHAVIOR-1K/asset_pipeline/logs/{obj}-{mode}.err", "w") as ferr,
    ):
        return subprocess.run(
            cmd,
            stdout=f,
            stderr=ferr,
            check=True,
            cwd="/scr/BEHAVIOR-1K/asset_pipeline",
        )


def main():
    failed_objects = set()
    with (
        PipelineFS() as pipeline_fs,
        ParallelZipFS("objects_usd.zip") as objects_fs,
        ParallelZipFS("metadata.zip") as metadata_fs,
        ParallelZipFS("systems.zip") as systems_fs,
        TempFS(temp_dir=str(TMP_DIR)) as dataset_fs,
    ):
        # Get the list of objects we need to process
        with pipeline_fs.open("metadata/fillables.json") as f:
            ids = {x.split("-")[1] for x in json.load(f)}

        with pipeline_fs.open("metadata/fillable_assignments.json") as f:
            assignments = json.load(f)

        has_dip_log = {
            obj
            for obj in ids
            if os.path.exists("/scr/BEHAVIOR-1K/asset_pipeline/logs/{obj}-dip.err")
        }
        no_need_dip = {k for k, v in assignments.items() if v == "ray"}  # | has_dip_log
        no_need_ray = {k for k, v in assignments.items() if v == "dip"}

        # with ParallelZipFS("fillable_volumes.zip", write=True) as out_fs:
        with pipeline_fs.makedirs(
            "artifacts/parallels/fillable_volumes", recreate=True
        ) as out_fs:
            # Copy everything over to the dataset FS
            print("Copying input to dataset fs...")
            fs.copy.copy_fs(metadata_fs, dataset_fs)
            fs.copy.copy_fs(systems_fs, dataset_fs)
            objdir_glob = list(objects_fs.glob("objects/*/*/"))
            for item in tqdm.tqdm(objdir_glob):
                if fs.path.parts(item.path)[-1] not in ids:
                    continue
                fs.copy.copy_fs(
                    objects_fs.opendir(item.path), dataset_fs.makedirs(item.path)
                )

            print("Launching cluster...")
            dask_client = launch_cluster(WORKER_COUNT)

            # Start the batched run
            object_glob = [x.path for x in dataset_fs.glob("objects/*/*/")]
            print("Queueing batches.")
            print("Total count: ", len(object_glob))

            # First the logic for the ray method
            ray_outputs = [fs.path.join(x, "fillable_ray.obj") for x in object_glob]
            ray_remaining = [
                (x, y)
                for x, y in zip(object_glob, ray_outputs)
                if fs.path.parts(x)[-2] not in no_need_ray and not out_fs.exists(y)
            ]
            random.shuffle(ray_remaining)
            ray_batch_size = min(
                BATCH_SIZE, math.ceil(len(ray_remaining) / WORKER_COUNT)
            )

            # Then the dip method.
            dip_outputs = [fs.path.join(x, "fillable_dip.obj") for x in object_glob]
            dip_remaining = [
                (x, y)
                for x, y in zip(object_glob, dip_outputs)
                if fs.path.parts(x)[-2] not in no_need_dip and not out_fs.exists(y)
            ]
            random.shuffle(dip_remaining)
            dip_batch_size = min(
                BATCH_SIZE, math.ceil(len(dip_remaining) / WORKER_COUNT)
            )

            futures = {}

            if ray_remaining:
                for start in range(0, len(ray_remaining), ray_batch_size):
                    end = start + ray_batch_size
                    batch = ray_remaining[start:end]

                    if batch:
                        ray_batch, ray_outputs = zip(*batch)
                        worker_future = dask_client.submit(
                            run_on_batch,
                            dataset_fs.getsyspath("/"),
                            list(ray_batch),
                            "ray",
                            pure=False,
                        )
                        futures[worker_future] = list(ray_outputs)

            if dip_remaining:
                for start in range(0, len(dip_remaining), dip_batch_size):
                    end = start + dip_batch_size
                    batch = dip_remaining[start:end]

                    if batch:
                        dip_batch, dip_outputs = zip(*batch)
                        worker_future = dask_client.submit(
                            run_on_batch,
                            dataset_fs.getsyspath("/"),
                            list(dip_batch),
                            "dip",
                            pure=False,
                        )
                        futures[worker_future] = list(dip_outputs)

            # Wait for all the workers to finish
            print("Queued all batches. Waiting for them to finish...")
            logs = []
            for future in tqdm.tqdm(as_completed(futures.keys()), total=len(futures)):
                # Check the batch results.
                batch = futures[future]
                if future.exception():
                    e = future.exception()
                    # logs.append({"stdout": e.stdout.decode("utf-8"), "stderr": e.stderr.decode("utf-8")})
                    print(e)
                else:
                    out = future.result()
                    # logs.append({"stdout": out.stdout.decode("utf-8"), "stderr": out.stderr.decode("utf-8")})

                # Copy the object to the output fs
                for item in batch:
                    dirpath = fs.path.dirname(item)
                    basename = fs.path.basename(item)
                    dataset_dir = dataset_fs.opendir(dirpath)
                    if dataset_dir.exists(basename):
                        fs.copy.copy_file(
                            dataset_dir,
                            basename,
                            out_fs.makedirs(dirpath, recreate=True),
                            basename,
                        )

            # Finish up.
            usd_glob = [x.path for x in dataset_fs.glob("objects/*/*/*.obj")]
            print(
                f"Done processing. Added {len(usd_glob)} objects. Archiving things now."
            )

        # Save the logs
        with pipeline_fs.pipeline_output().open(
            "generate_fillable_volumes_flat.json", "w"
        ) as f:
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
