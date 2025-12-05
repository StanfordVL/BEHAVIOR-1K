import uuid
import hydra
import json
import logging
import os
import subprocess
import requests
import threading
import time
from inspect import getsourcefile
from omegaconf import OmegaConf
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.learning.utils.obs_utils import (
    create_video_writer,
)
from omnigibson.macros import gm
from pathlib import Path
from omnigibson.learning.eval import Evaluator

# Get the ID of the 0th GPU from nvidia-smi
gpu_id = (
    subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"]).decode("utf-8").strip()
)

# set global variables to boost performance
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True
gm.HEADLESS = True
scratch_disk = "/scr-ssd" if os.path.exists("/scr-ssd") else "/scr"
user = os.environ["USER"]
gm.APPDATA_PATH = os.path.join(scratch_disk, user, "omnigibson-cache", gpu_id)
os.makedirs(gm.APPDATA_PATH, exist_ok=True)

# create module logger
logger = logging.getLogger("evaluator")
logger.setLevel(logging.INFO)

# Job queue server URL
JOB_QUEUE_URL = os.environ.get("JOB_QUEUE_URL", "http://cgokmen-lambda.stanford.edu:8000")
PRINT_INTERVAL = 100  # steps


def request_job(worker_id: str) -> dict:
    """Request a job from the job queue server."""
    try:
        response = requests.get(f"{JOB_QUEUE_URL}/job", params={"worker": worker_id})
        response.raise_for_status()
        data = response.json()
        return data.get("job")  # Returns None if no job available
    except Exception as e:
        logger.error(f"Failed to request job: {e}")
        return None


def send_heartbeat_all_jobs_and_resources(worker_id: str) -> None:
    """Send heartbeat for all jobs and resources assigned to this worker."""
    try:
        response = requests.post(f"{JOB_QUEUE_URL}/heartbeat", params={"worker": worker_id})
        response.raise_for_status()
        logger.debug(f"Heartbeat sent successfully: {response.json()}")
    except Exception as e:
        logger.warning(f"Failed to send heartbeat: {e}")


def reserve_resource(resource_type: str, worker_id: str, job_id: str) -> dict:
    """Reserve a resource of the specified type for the job. Retries every 20 seconds until successful."""
    start_time = time.time()
    while True:
        try:
            response = requests.post(
                f"{JOB_QUEUE_URL}/resource/{resource_type}/acquire", params={"worker": worker_id, "job_id": job_id}
            )
            response.raise_for_status()
            logger.info(
                f"Resource {resource_type} reserved successfully after {(time.time() - start_time) / 60:.2f} minutes"
            )
            return response.json()  # Returns {"index": ..., "resource": ...}
        except Exception as e:
            logger.warning(
                f"Failed to reserve resource: {e}. Total wait time: {(time.time() - start_time) / 60:.2f} minutes. Retrying in 20 seconds..."
            )
            time.sleep(20)


def release_resource(resource_type: str, resource_idx: int, worker_id: str, job_id: str) -> None:
    """Release a reserved resource."""
    try:
        response = requests.post(
            f"{JOB_QUEUE_URL}/resource/{resource_type}/release",
            params={"worker": worker_id, "job_id": job_id, "resource_idx": resource_idx},
        )
        response.raise_for_status()
        logger.info(f"Resource {resource_type}[{resource_idx}] released successfully")
    except Exception as e:
        logger.warning(f"Failed to release resource: {e}")


def mark_job_as_completed(job_id: str, worker_id: str) -> None:
    """Mark a job as completed in the job queue."""
    try:
        response = requests.post(f"{JOB_QUEUE_URL}/done/{job_id}", params={"worker": worker_id})
        response.raise_for_status()
        logger.info(f"Job {job_id} marked as completed")
    except Exception as e:
        logger.error(f"Failed to mark job as completed: {e}")


def heartbeat_thread_func(worker_id: str, stop_event: threading.Event) -> None:
    """Background thread that sends heartbeats every 30 seconds."""
    while not stop_event.is_set():
        send_heartbeat_all_jobs_and_resources(worker_id)
        # Wait for 10 seconds or until stop event is set
        stop_event.wait(30)


def main():
    # Generate a unique worker ID for this worker
    user = os.environ.get("USER", "")
    worker_id = user + "-" + os.environ.get("SLURM_JOB_ID", "") + "-" + str(uuid.uuid4())
    logger.info(f"Worker ID: {worker_id}")

    # Start heartbeat thread
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(target=heartbeat_thread_func, args=(worker_id, stop_event), daemon=True)
    heartbeat_thread.start()
    logger.info("Heartbeat thread started")

    # Request a job from the job queue
    job_data = request_job(worker_id)
    if job_data is None:
        logger.info("No job available. Exiting.")
        stop_event.set()
        heartbeat_thread.join()
        return
    job_start_time = time.time()

    job_id = job_data["id"]
    payload = job_data["payload"]
    resource_type = job_data["resource_type"]

    logger.info(f"Received job {job_id} with resource_type={resource_type}")
    logger.info(f"Job payload: {payload}")

    # Extract task parameters from payload
    team_slug = payload.get("team_slug")
    task_name = payload.get("task")
    instance_basename = payload.get("instance_basename")
    idx = int(instance_basename.replace("_template-tro_state.json", "").rsplit("_", 1)[-1])
    log_path = os.path.join("/vision/group/behavior/eval-results/", team_slug)

    # Inject the instance basename into the config
    overrides = [
        "policy=websocket",
        f"env_wrapper._target_=omnigibson.learning.wrappers.challenge_submissions.submission_{team_slug}.WRAPPER_CLASS",
        f"task.name={task_name}",
        f"log_path={log_path}",
        f"model.host=None",
        f"model.port=None",
        "test_hidden=true",
    ]
    register_omegaconf_resolvers()
    # open yaml from task path
    with hydra.initialize_config_dir(f"{Path(getsourcefile(lambda: 0)).parents[0]}/configs", version_base="1.1"):
        config = hydra.compose("base_config.yaml", overrides=overrides)
    OmegaConf.resolve(config)
    # set video path
    if config.write_video:
        video_path = Path(config.log_path).expanduser() / "videos" / user
        video_path.mkdir(parents=True, exist_ok=True)

    # establish metrics
    metrics = {}
    metrics_path = Path(config.log_path).expanduser() / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    # Note that when used as a context manager, the evaluator will be closed automatically when the context is exited,
    # and it also handles SIGINT and SIGTERM and prints a bunch of stuff. We don't want that here so we don't use it as a context manager.
    evaluator = Evaluator(config)
    logger.info("Starting evaluation...")
    # This reset is needed to update the robot initial pose
    evaluator.reset()
    evaluator.load_task_instance(idx, test_hidden=True)

    logger.info(f"Starting task instance {idx} for evaluation...")

    # Reserve a resource from the job queue
    logger.info(f"Reserving resource of type {resource_type}...")
    resource_data = reserve_resource(resource_type, worker_id, job_id)
    resource_idx = resource_data["index"]
    resource_info = resource_data["resource"]
    logger.info(f"Reserved resource {resource_type}[{resource_idx}]: {resource_info}")

    # Update the websocket policy config with the resource info
    evaluator.policy.update_host(resource_info["host"], int(resource_info["port"]))
    # This reset is needed to send a to reset the websocket policy
    evaluator.reset()

    done = False
    if config.write_video:
        video_name = str(video_path) + f"/{config.task.name}_{idx}.mp4"
        evaluator.video_writer = create_video_writer(
            fpath=video_name,
            resolution=(448, 672),
        )

    try:
        # run metric start callbacks
        for metric in evaluator.metrics:
            metric.start_callback(evaluator.env)

        # Print first step time
        first_step_time = time.time()
        logger.info(f"First step time: {first_step_time - job_start_time:.2f} seconds")

        step_idx = 0
        last_block_time = time.time()
        while not done:
            terminated, truncated = evaluator.step()
            if terminated or truncated:
                done = True
            if config.write_video:
                evaluator._write_video()

            step_idx += 1
            if step_idx % PRINT_INTERVAL == 0:
                current_time = time.time()
                time_per_step = (current_time - last_block_time) / PRINT_INTERVAL
                logger.info(f"Step {step_idx} completed. Average FPS: {1 / time_per_step:.2f}")
                logger.info(
                    f"Estimated time remaining: {(evaluator.human_stats['length'] - step_idx) * time_per_step:.2f} seconds"
                )
                last_block_time = current_time

        # run metric end callbacks
        for metric in evaluator.metrics:
            metric.end_callback(evaluator.env)

        end_time = time.time()
        logger.info(f"Evaluation finished at step {evaluator.env._current_step}.")
        logger.info(f"Evaluation exit state: {terminated}, {truncated}")
        logger.info(f"Total trials: {evaluator.n_trials}")
        logger.info(f"Total success trials: {evaluator.n_success_trials}")
        logger.info(f"Total job time: {end_time - job_start_time:.2f} seconds")
        logger.info(f"Total stepping time: {end_time - first_step_time:.2f} seconds")
        logger.info(f"Average FPS: {step_idx / (end_time - first_step_time):.2f}")

        # gather metric results and write to file
        for metric in evaluator.metrics:
            metrics.update(metric.gather_results())
        with open(metrics_path / f"{config.task.name}_{idx}.json", "w") as f:
            json.dump(metrics, f)

        # reset video writer
        if config.write_video:
            evaluator.video_writer = None
            logger.info(f"Saved video to {video_name}")

        # Mark job as completed
        mark_job_as_completed(job_id, worker_id)
    finally:
        # Release the resource
        release_resource(resource_type, resource_idx, worker_id, job_id)

    # Stop heartbeat thread and wait for it to finish
    logger.info("Stopping heartbeat thread...")
    stop_event.set()
    heartbeat_thread.join()
    logger.info("Heartbeat thread stopped")


if __name__ == "__main__":
    main()
