import json
import logging
import os
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query

# --- Config ---
JOB_HEARTBEAT_TIMEOUT = 60 * 60  # seconds before reclaiming a job
RESOURCE_HEARTBEAT_TIMEOUT = 60 * 20  # seconds before freeing a resource
JOB_SWEEP_INTERVAL = 60  # how often to sweep job heartbeats
RESOURCE_SWEEP_INTERVAL = 10  # how often to sweep resource heartbeats
JOBS_FILE = Path(__file__).resolve().parent / "jobs.json"
RESOURCES_FILE = Path(__file__).resolve().parent / "resources.json"

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("jobqueue")


# --- Job data structures ---
class Job:
    def __init__(self, raw: dict):
        if "id" not in raw:
            raise ValueError("Job entry missing 'id'")
        self.id: str = str(raw["id"])
        self.payload = raw.get("payload", {})
        self.resource_type: str = raw["resource_type"]
        self.status: str = raw.get("status", "pending")
        self.assigned_to: Optional[str] = raw.get("assigned_to")
        self.assigned_at: Optional[float] = raw.get("assigned_at")
        self.last_heartbeat: Optional[float] = raw.get("last_heartbeat")
        self.attempts: int = int(raw.get("attempts", 0))

    def assign(self, worker_id: str):
        self.attempts += 1
        self.status = "in_progress"
        self.assigned_to = worker_id
        self.assigned_at = time.time()
        self.last_heartbeat = self.assigned_at

    def touch(self):
        self.last_heartbeat = time.time()

    def requeue(self):
        self.status = "pending"
        self.assigned_to = None
        self.assigned_at = None
        self.last_heartbeat = None

    def finish(self):
        self.status = "done"
        self.assigned_to = None
        self.assigned_at = None
        self.last_heartbeat = None

    def to_dict(self):
        return {
            "id": self.id,
            "payload": self.payload,
            "resource_type": self.resource_type,
            "status": self.status,
            "assigned_to": self.assigned_to,
            "assigned_at": self.assigned_at,
            "last_heartbeat": self.last_heartbeat,
            "attempts": self.attempts,
        }


# --- Resource data structures ---
class ResourceLease:
    def __init__(self, worker_id: str, job_id: str, idx: int):
        self.worker_id = worker_id
        self.job_id = job_id
        self.idx = idx
        self.last_heartbeat = time.time()

    def touch(self):
        self.last_heartbeat = time.time()


class ResourcePool:
    def __init__(self, resource_type: str, resources: List[dict]):
        if not resources:
            raise ValueError(f"No resources defined for type {resource_type}")
        self.resource_type = resource_type
        self.resources = resources
        self.capacity = len(resources)
        self.lock = threading.Lock()
        self.semaphore = threading.BoundedSemaphore(self.capacity)
        self.assignments: Dict[int, ResourceLease] = {}
        self.holder_index: Dict[Tuple[str, str], int] = {}

    def acquire(
        self, worker_id: str, job_id: str
    ) -> Tuple[Optional[int], Optional[dict], bool]:
        holder_key = (worker_id, job_id)
        with self.lock:
            if holder_key in self.holder_index:
                idx = self.holder_index[holder_key]
                lease = self.assignments[idx]
                lease.touch()
                return idx, self.resources[idx], False

            if not self.semaphore.acquire(blocking=False):
                return None, None, False

            for idx in range(self.capacity):
                if idx not in self.assignments:
                    lease = ResourceLease(worker_id, job_id, idx)
                    self.assignments[idx] = lease
                    self.holder_index[holder_key] = idx
                    return idx, self.resources[idx], True

            # Should never happen, but release semaphore if we didn't allocate.
            self.semaphore.release()
            return None, None, False

    def release(
        self, worker_id: str, job_id: str, resource_idx: int, reason: str = "released"
    ) -> bool:
        holder_key = (worker_id, job_id)
        with self.lock:
            lease = self.assignments.get(resource_idx)
            if not lease:
                return False
            if lease.worker_id != worker_id or lease.job_id != job_id:
                return False
            self._drop_assignment(holder_key, resource_idx, reason)
            return True

    def heartbeat(self, worker_id: str, job_id: str, resource_idx: int) -> bool:
        with self.lock:
            lease = self.assignments.get(resource_idx)
            if not lease:
                return False
            if lease.worker_id != worker_id or lease.job_id != job_id:
                return False
            lease.touch()
            return True

    def heartbeat_all_for_worker(self, worker_id: str) -> int:
        """Heartbeat all resources assigned to a worker. Returns count of heartbeated resources."""
        count = 0
        with self.lock:
            for idx, lease in self.assignments.items():
                if lease.worker_id == worker_id:
                    lease.touch()
                    count += 1
        return count

    def sweep(self) -> List[int]:
        now = time.time()
        dropped = []
        with self.lock:
            for idx, lease in list(self.assignments.items()):
                if now - lease.last_heartbeat > RESOURCE_HEARTBEAT_TIMEOUT:
                    holder_key = (lease.worker_id, lease.job_id)
                    self._drop_assignment(holder_key, idx, "heartbeat timeout")
                    dropped.append(idx)
        return dropped

    def _drop_assignment(self, holder_key: Tuple[str, str], idx: int, reason: str):
        self.assignments.pop(idx, None)
        self.holder_index.pop(holder_key, None)
        self.semaphore.release()
        log.warning(
            "Resource %s[%s] freed (%s)",
            self.resource_type,
            idx,
            reason,
        )


# --- In-memory state ---
jobs_lock = threading.Lock()
jobs: List[Job] = []
job_index: Dict[str, Job] = {}
pending_jobs: List[Job] = []
in_progress: Dict[str, Job] = {}
finished_jobs: set = set()
resource_pools: Dict[str, ResourcePool] = {}
resource_counts: Dict[str, int] = {}
in_progress_resource_counts: Counter = Counter()


def load_jobs():
    raw_jobs = json.loads(JOBS_FILE.read_text())
    if not isinstance(raw_jobs, list):
        raise ValueError("Jobs file must contain a list of jobs")
    for idx, entry in enumerate(raw_jobs):
        job_data = dict(entry)
        job_data.setdefault("id", str(idx))
        job = Job(job_data)
        jobs.append(job)
        job_index[job.id] = job
        if job.status == "done":
            finished_jobs.add(job.id)
            continue
        job.requeue()
        pending_jobs.append(job)


def load_resources():
    raw_resources = json.loads(RESOURCES_FILE.read_text())
    if not isinstance(raw_resources, dict):
        raise ValueError("Resources file must contain a dict of resource lists")
    for resource_type, entries in raw_resources.items():
        resource_pools[resource_type] = ResourcePool(resource_type, entries)
        resource_counts[resource_type] = len(entries)


def persist_jobs_locked():
    serialized = [job.to_dict() for job in jobs]
    data = json.dumps(serialized, indent=2)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(JOBS_FILE.parent)
    ) as tmp:
        tmp.write(data)
        temp_name = tmp.name
    os.replace(temp_name, JOBS_FILE)


def select_next_job() -> Optional[Job]:
    if not pending_jobs:
        return None
    pending_jobs.sort(
        key=lambda job: (
            in_progress_resource_counts.get(job.resource_type, 0)
            / max(resource_counts.get(job.resource_type, 1), 1),
            job.attempts,
            job.id,
        )
    )

    for i, job in enumerate(pending_jobs):
        # Only pick a job if there is at least one instance of the required resource type that exists,
        # even if it's not available right this moment.
        if resource_counts.get(job.resource_type, 0) > 0:
            return pending_jobs.pop(i)

    return None


def assign_job(worker_id: str) -> Optional[Job]:
    with jobs_lock:
        job = select_next_job()
        if not job:
            return None
        job.assign(worker_id)
        in_progress[job.id] = job
        in_progress_resource_counts[job.resource_type] += 1
        persist_jobs_locked()
        return job


def complete_job(job_id: str, worker_id: str):
    with jobs_lock:
        job = in_progress.get(job_id)
        if not job or job.assigned_to != worker_id:
            raise HTTPException(status_code=400, detail="Job not assigned to worker")
        job.finish()
        finished_jobs.add(job_id)
        in_progress.pop(job_id, None)
        in_progress_resource_counts[job.resource_type] -= 1
        persist_jobs_locked()


def heartbeat_job(job_id: str, worker_id: str):
    with jobs_lock:
        job = in_progress.get(job_id)
        if not job or job.assigned_to != worker_id:
            raise HTTPException(status_code=404, detail="Job not assigned to worker")
        job.touch()
        persist_jobs_locked()


def heartbeat_all_jobs_for_worker(worker_id: str) -> int:
    """Heartbeat all jobs assigned to a worker. Returns count of heartbeated jobs."""
    count = 0
    with jobs_lock:
        for job in in_progress.values():
            if job.assigned_to == worker_id:
                job.touch()
                count += 1
        if count > 0:
            persist_jobs_locked()
    return count


def heartbeat_all_resources_for_worker(worker_id: str) -> Dict[str, int]:
    """Heartbeat all resources assigned to a worker across all resource pools.
    Returns a dict mapping resource_type to count of heartbeated resources."""
    results = {}
    for resource_type, pool in resource_pools.items():
        count = pool.heartbeat_all_for_worker(worker_id)
        if count > 0:
            results[resource_type] = count
    return results


def reclaim_stale_jobs():
    while True:
        time.sleep(JOB_SWEEP_INTERVAL)
        now = time.time()
        reclaimed = []
        with jobs_lock:
            for job_id, job in list(in_progress.items()):
                if (
                    job.last_heartbeat
                    and (now - job.last_heartbeat) > JOB_HEARTBEAT_TIMEOUT
                ):
                    reclaimed.append(job)
            for job in reclaimed:
                log.warning(
                    "Job %s reclaimed after heartbeat timeout (worker=%s)",
                    job.id,
                    job.assigned_to,
                )
                in_progress.pop(job.id, None)
                in_progress_resource_counts[job.resource_type] -= 1
                job.requeue()
                pending_jobs.append(job)
            if reclaimed:
                persist_jobs_locked()


def sweep_resources():
    while True:
        time.sleep(RESOURCE_SWEEP_INTERVAL)
        for pool in resource_pools.values():
            pool.sweep()


def log_stats():
    while True:
        utilization = {}
        for resource_type, pool in resource_pools.items():
            with pool.lock:
                used = len(pool.assignments)
                total = pool.capacity
            utilization[resource_type] = f"{used}/{total} ({used / total * 100:.1f}%)"

        message = [
            "Jobs: %d" % len(jobs),
            "Pending jobs: %d" % len(pending_jobs),
            "In progress: %d" % len(in_progress),
            "Finished jobs: %d" % len(finished_jobs),
            "Resource pools: %d" % len(resource_pools),
            "Resource counts: %s" % str(resource_counts),
            "Resource utilization by resource type: %s" % str(utilization),
            "In-progress jobs per resource type: %s" % str(in_progress_resource_counts),
        ]
        log.info(" | ".join(message))
        time.sleep(60)


# --- Initialization ---
load_jobs()
load_resources()
threading.Thread(target=reclaim_stale_jobs, daemon=True).start()
threading.Thread(target=sweep_resources, daemon=True).start()
threading.Thread(target=log_stats, daemon=True).start()

app = FastAPI()


# --- API Endpoints ---
@app.get("/job")
def get_job(worker: str = Query(...)):
    job = assign_job(worker)
    if not job:
        return {"job": None}
    log.info("Job %s assigned to %s (attempt %s)", job.id, worker, job.attempts)
    return {
        "job": {
            "id": job.id,
            "payload": job.payload,
            "resource_type": job.resource_type,
        }
    }


@app.post("/job/{job_id}/heartbeat")
def job_heartbeat(job_id: str, worker: str = Query(...)):
    heartbeat_job(job_id, worker)
    return {"status": "ok"}


@app.post("/done/{job_id}")
def done(job_id: str, worker: str = Query(...)):
    complete_job(job_id, worker)
    log.info("Job %s finished by %s", job_id, worker)
    return {"status": "ok"}


@app.post("/resource/{resource_type}/acquire")
def acquire_resource(
    resource_type: str, worker: str = Query(...), job_id: str = Query(...)
):
    pool = resource_pools.get(resource_type)
    if not pool:
        raise HTTPException(status_code=404, detail="Unknown resource type")
    idx, resource, is_new = pool.acquire(worker, job_id)
    if resource is None:
        raise HTTPException(status_code=429, detail="No resources available")
    action = "allocated" if is_new else "heartbeat"
    log.info(
        "Resource %s[%s] %s to %s for job %s",
        resource_type,
        idx,
        action,
        worker,
        job_id,
    )
    return {"index": idx, "resource": resource}


@app.post("/resource/{resource_type}/heartbeat")
def resource_heartbeat(
    resource_type: str,
    worker: str = Query(...),
    job_id: str = Query(...),
    resource_idx: int = Query(...),
):
    pool = resource_pools.get(resource_type)
    if not pool or not pool.heartbeat(worker, job_id, resource_idx):
        raise HTTPException(status_code=404, detail="Resource lease not found")
    return {"status": "ok"}


@app.post("/resource/{resource_type}/release")
def release_resource(
    resource_type: str,
    worker: str = Query(...),
    job_id: str = Query(...),
    resource_idx: int = Query(...),
):
    pool = resource_pools.get(resource_type)
    if not pool or not pool.release(
        worker, job_id, resource_idx, reason="released by worker"
    ):
        raise HTTPException(
            status_code=404, detail="Resource lease not found or mismatch"
        )
    return {"status": "ok"}


@app.post("/heartbeat")
def heartbeat_all(worker: str = Query(...)):
    """Heartbeat all jobs and resources assigned to a worker."""
    job_count = heartbeat_all_jobs_for_worker(worker)
    resource_counts = heartbeat_all_resources_for_worker(worker)
    total_resources = sum(resource_counts.values())
    return {
        "status": "ok",
        "jobs_heartbeated": job_count,
        "resources_heartbeated": resource_counts,
        "total_resources_heartbeated": total_resources,
    }
