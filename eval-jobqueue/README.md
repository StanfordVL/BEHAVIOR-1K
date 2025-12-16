# eval-jobqueue

A distributed job queue system for evaluating submissions in the BEHAVIOR-1K challenge.

## Architecture Overview

The system consists of three main components:

1. **Job Queue Server** (`jobqueue.py`) - A FastAPI server that manages job distribution and resource allocation
2. **Workers** (`eval_with_jobqueue.py`) - SLURM jobs that process evaluation tasks
3. **Job Generator** (`generate_jobs.py`) - Utility to generate jobs and resource configurations

### How It Works

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Job Generator  │────▶│  Job Queue       │◀────│  SLURM Workers      │
│                 │     │  Server          │     │                     │
│ - jobs.json     │     │  (FastAPI)       │     │ - Request jobs      │
│ - resources.json│     │                  │     │ - Reserve resources │
└─────────────────┘     │ - Job management │     │ - Run evaluations   │
                        │ - Resource pools │     │ - Send heartbeats   │
                        └──────────────────┘     └─────────────────────┘
```

The job queue runs on a non-SLURM node accessible to all SLURM nodes. Workers connect to request `(team, task, instance)` jobs, launch the simulator, then reserve a "resource" (websocket policy server). This two-phase approach keeps policy servers 100% saturated by having workers pre-load tasks while waiting for resources.

## Directory Structure

```
eval-jobqueue/
├── jobqueue.py              # FastAPI job queue server
├── generate_jobs.py         # Job and resource generation script
├── stream_logs.py           # Google Cloud log streaming utility
├── jobs.json                # Generated job definitions (auto-generated)
├── resources.json           # Generated resource pools (auto-generated)
├── resources-raw.json       # Input: websocket URLs and constraints
├── challenge_submissions/   # Input: team submission JSON files
└── test_instances/          # Input: task instance files organized by task
    └── <task_name>/
        └── *_template-tro_state.json
```

## Setup

### Prerequisites

```bash
pip install fastapi uvicorn python-slugify
```

### Input Files

1. **`challenge_submissions/`** - Place team submission JSON files here. Each file should contain:
   - `team`: Team name
   - `overall_scores.q_score`: Overall quality score
   - `per_task_scores.q_score`: Dict mapping task names to scores

2. **`test_instances/`** - Task instance files organized by task name subdirectories

3. **`resources-raw.json`** - Websocket policy server URLs with optional task filters:
   ```json
   {
     "team_slug": [
       {
         "host": "server1.example.com",
         "port": 8080
       },
       {
         "host": "server2.example.com",
         "port": 8080,
         "compatible_task": ["task_a", "task_b"],
         "not_compatible_task": ["task_c"]
       }
     ]
   }
   ```

### Generate Jobs and Resources

```bash
python generate_jobs.py
```

This creates:
- `jobs.json` - Job definitions for each `(team, task, instance)` triple
- `resources.json` - Resource pools grouped by task compatibility constraints

**Options:**
```bash
python generate_jobs.py \
  --test-instances-dir ./test_instances \
  --submissions-dir ./challenge_submissions \
  --resources-raw-file ./resources-raw.json \
  --output-jobs-file ./jobs.json \
  --output-resources-file ./resources.json \
  --dry-run  # Preview without writing files
```

**Job Selection Criteria:**
- Only teams with `q_score > 0` for a task get jobs for that task
- Limited to top 5 standard submissions and top 5 privileged submissions by overall `q_score`
- Skips instances that already have results in `/vision/group/behavior/eval-results/<team_slug>/metrics/`

## Running the System

### 1. Start the Job Queue Server

```bash
uvicorn jobqueue:app --host 0.0.0.0 --port 8000
```

With Google Cloud log streaming:
```bash
uvicorn jobqueue:app --host 0.0.0.0 2>&1 | \
  GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json \
  python stream_logs.py jobqueue
```

### 2. Launch SLURM Workers

Update `JOB_QUEUE_URL` environment variable or use the default, then submit jobs:

```bash
# Set job queue URL (optional, defaults to http://cgokmen-lambda.stanford.edu:8000)
export JOB_QUEUE_URL="http://your-server:8000"

# Submit SLURM jobs
sbatch eval.sh
```

## Configuration

### Server Configuration (`jobqueue.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `JOB_HEARTBEAT_TIMEOUT` | 60 min | Time before reclaiming an unresponsive job |
| `RESOURCE_HEARTBEAT_TIMEOUT` | 20 min | Time before freeing an unresponsive resource |
| `JOB_SWEEP_INTERVAL` | 60 sec | How often to check for stale jobs |
| `RESOURCE_SWEEP_INTERVAL` | 10 sec | How often to check for stale resources |

### Worker Configuration (`eval_with_jobqueue.py`)

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `JOB_QUEUE_URL` | `http://cgokmen-lambda.stanford.edu:8000` | Job queue server URL |

Workers send heartbeats every 30 seconds and retry resource acquisition every 20 seconds.

## API Endpoints

### Job Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/job?worker=<id>` | GET | Request next available job |
| `/job/{job_id}/heartbeat?worker=<id>` | POST | Send heartbeat for a specific job |
| `/done/{job_id}?worker=<id>` | POST | Mark job as completed |

### Resource Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/resource/{type}/acquire?worker=<id>&job_id=<id>` | POST | Reserve a resource |
| `/resource/{type}/heartbeat?worker=<id>&job_id=<id>&resource_idx=<idx>` | POST | Heartbeat a resource |
| `/resource/{type}/release?worker=<id>&job_id=<id>&resource_idx=<idx>` | POST | Release a resource |

### Unified Heartbeat

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/heartbeat?worker=<id>` | POST | Heartbeat all jobs and resources for a worker |

## Job Lifecycle

```
┌─────────┐    assign    ┌─────────────┐   complete   ┌──────┐
│ pending │─────────────▶│ in_progress │─────────────▶│ done │
└─────────┘              └─────────────┘              └──────┘
     ▲                         │
     │                         │ heartbeat timeout
     └─────────────────────────┘
           (requeue)
```

1. **pending** - Job waiting to be assigned
2. **in_progress** - Job assigned to a worker, receiving heartbeats
3. **done** - Job completed successfully

Jobs are automatically requeued if heartbeats stop (worker crash/timeout).

## Resource Pool Management

Resources are organized into pseudo-groups based on task compatibility constraints. Each group operates as a semaphore-controlled pool:

- Workers acquire resources before starting evaluation
- Resources are held until explicitly released or heartbeat timeout
- Multiple workers can wait for the same resource type
- Job scheduling prioritizes resource types with lower utilization

## Monitoring

The server logs statistics every 60 seconds including:
- Total/pending/in-progress/finished job counts
- Resource pool utilization by type
- In-progress jobs per resource type

## Output

Evaluation results are written to:
- **Metrics**: `/vision/group/behavior/eval-results/<team_slug>/metrics/<task>_<instance_id>.json`
- **Videos**: `/vision/group/behavior/eval-results/<team_slug>/videos/<user>/<task>_<instance_id>.mp4`
