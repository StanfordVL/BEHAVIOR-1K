#!/bin/bash
#SBATCH --job-name=replay_data_sc
#SBATCH --account=svl
#SBATCH --partition=svl,sc-loprio
#SBATCH --nodes=1
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/vision/u/%u/BEHAVIOR-1K/outputs/sc/replay_data/%x-%j.out
#SBATCH --error=/vision/u/%u/BEHAVIOR-1K/outputs/sc/replay_data/%x-%j.err

set -euo pipefail
umask 0002

ensure_shared_permissions() {
  local path
  local resolved
  for path in "$@"; do
    [[ -e "$path" ]] || continue
    resolved="$(realpath -m "$path")"
    case "$resolved" in
      /vision/group/behavior|/vision/group/behavior/*)
        chgrp -R behavior "$resolved" || true
        chmod -R g+rwX "$resolved" || true
        find "$resolved" -type d -exec chmod g+s {} + || true
        ;;
    esac
  done
}

usage() {
  cat <<'EOF'
Usage:
  sbatch replay_data_sc.sbatch --data_folder PATH --task_name TASK --shard_id ID --demo_ids 00000000,00000010,00000020

Required:
  --data_folder        Root containing 2026-challenge-rawdata/
  --task_name          BEHAVIOR task name (used for shard output paths / markers)
  --shard_id           Stable shard id, e.g. 0007
  --demo_ids           Comma-separated *episode demo ids* (8-digit: task_id + instance_id + traj_id) to replay serially in this shard

Optional:
  --lerobot_root_dir   LeRobot shard root. Defaults to <data_folder>/lerobot_shards
  --lerobot_repo_id    LeRobot repo id. Defaults to b1k_shards/<task_name>/shard-<shard_id>
  --flush_every_n_steps  Defaults to 1000
  --repo_root          BEHAVIOR-1K checkout. Defaults to this script's repository root
EOF
}

DATA_FOLDER=""
TASK_NAME=""
SHARD_ID=""
DEMO_IDS=""
LEROBOT_ROOT_DIR=""
LEROBOT_REPO_ID=""
FLUSH_EVERY_N_STEPS="1000"
REPO_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_folder)
      DATA_FOLDER="$2"
      shift 2
      ;;
    --task_name)
      TASK_NAME="$2"
      shift 2
      ;;
    --shard_id)
      SHARD_ID="$2"
      shift 2
      ;;
    --demo_ids)
      DEMO_IDS="$2"
      shift 2
      ;;
    --lerobot_root_dir)
      LEROBOT_ROOT_DIR="$2"
      shift 2
      ;;
    --lerobot_repo_id)
      LEROBOT_REPO_ID="$2"
      shift 2
      ;;
    --flush_every_n_steps)
      FLUSH_EVERY_N_STEPS="$2"
      shift 2
      ;;
    --repo_root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$DATA_FOLDER" || -z "$TASK_NAME" || -z "$SHARD_ID" || -z "$DEMO_IDS" ]]; then
  usage >&2
  exit 2
fi

if [[ -z "$REPO_ROOT" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

if [[ -z "$LEROBOT_ROOT_DIR" ]]; then
  LEROBOT_ROOT_DIR="${DATA_FOLDER}/lerobot_shards"
fi

if [[ -z "$LEROBOT_REPO_ID" ]]; then
  LEROBOT_REPO_ID="b1k_shards/${TASK_NAME}/shard-${SHARD_ID}"
fi

MARKER_DIR="${LEROBOT_ROOT_DIR}/.replay_status/${TASK_NAME}/shard-${SHARD_ID}"
RUNNING_MARKER="${MARKER_DIR}/running"
COMPLETE_MARKER="${MARKER_DIR}/complete"
FAILED_MARKER="${MARKER_DIR}/failed"
LEROBOT_OUTPUT_DIR="${LEROBOT_ROOT_DIR}/${LEROBOT_REPO_ID}"

cleanup_shared_permissions() {
  ensure_shared_permissions "$MARKER_DIR" "$LEROBOT_OUTPUT_DIR"
}

mkdir -p "/vision/u/${USER}/BEHAVIOR-1K/outputs/sc/replay_data"
mkdir -p "$MARKER_DIR"
ensure_shared_permissions "$MARKER_DIR"

echo "SLURM_JOBID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_NNODES=${SLURM_NNODES:-}"
echo "SLURMTMPDIR=${SLURMTMPDIR:-}"
echo "working directory=${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "repo root=${REPO_ROOT}"
echo "task=${TASK_NAME}"
echo "shard=${SHARD_ID}"
echo "demo_ids=${DEMO_IDS}"
echo "lerobot_root_dir=${LEROBOT_ROOT_DIR}"
echo "lerobot_repo_id=${LEROBOT_REPO_ID}"
echo "Current time: $(date)"

if [[ -f "$COMPLETE_MARKER" ]]; then
  echo "Shard already complete: ${COMPLETE_MARKER}"
  exit 0
fi

mark_failed() {
  local exit_code="$?"
  mkdir -p "$MARKER_DIR"
  {
    echo "exit_code=${exit_code}"
    echo "job_id=${SLURM_JOB_ID:-}"
    echo "host=$(hostname)"
    echo "time=$(date --iso-8601=seconds)"
	  } > "$FAILED_MARKER"
	  rm -f "$RUNNING_MARKER"
	  cleanup_shared_permissions
	  exit "$exit_code"
	}
trap cleanup_shared_permissions EXIT
trap mark_failed ERR TERM INT

GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -n 1 || true)"
GPU_NAME="$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)"
GPU_MEM_MB="$(echo "$GPU_INFO" | cut -d',' -f2 | xargs)"
GPU_NAME_LOWER="$(echo "$GPU_NAME" | tr '[:upper:]' '[:lower:]')"

if [[ -z "$GPU_NAME" || -z "$GPU_MEM_MB" ]]; then
  echo "Could not read GPU information from nvidia-smi." >&2
  exit 42
fi

if (( GPU_MEM_MB < 24000 )); then
  echo "GPU has less than 24GB VRAM: ${GPU_NAME} (${GPU_MEM_MB} MB)." >&2
  exit 42
fi

if [[ ! "$GPU_NAME_LOWER" =~ (titan[[:space:]]*rtx|3090|4090|5090|a6000|a5000|a40|l40s|rtx.*6000) ]]; then
  echo "GPU is not in the allowed RTX-class replay set: ${GPU_NAME}." >&2
  exit 42
fi

echo "Using GPU: ${GPU_NAME} (${GPU_MEM_MB} MB)"

CONDA_SH="/vision/u/${USER}/miniconda3/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
  CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
fi
set +u
source "$CONDA_SH"
conda activate behavior
set -u

cd "$REPO_ROOT"

{
  echo "job_id=${SLURM_JOB_ID:-}"
  echo "host=$(hostname)"
  echo "time=$(date --iso-8601=seconds)"
  echo "demo_ids=${DEMO_IDS}"
} > "$RUNNING_MARKER"
rm -f "$FAILED_MARKER"
ensure_shared_permissions "$MARKER_DIR"

IFS=',' read -r -a DEMO_ID_ARRAY <<< "$DEMO_IDS"
FIRST_DEMO=1
for DEMO_ID in "${DEMO_ID_ARRAY[@]}"; do
  DEMO_ID="$(echo "$DEMO_ID" | xargs)"
  if [[ -z "$DEMO_ID" ]]; then
    continue
  fi

  RESUME_ARGS=()
  if [[ "$FIRST_DEMO" -eq 0 ]]; then
    RESUME_ARGS=(--resume_lerobot)
  fi

  echo "Replaying task=${TASK_NAME} demo_id=${DEMO_ID} shard=${SHARD_ID} resume=${RESUME_ARGS[*]:-false}"
  OMNIGIBSON_HEADLESS=1 python OmniGibson/scripts/learning/replay_obs.py \
    --data_folder "$DATA_FOLDER" \
    --demo_id "$DEMO_ID" \
    --output_format lerobot \
    --flush_every_n_steps "$FLUSH_EVERY_N_STEPS" \
    --lerobot_root_dir "$LEROBOT_ROOT_DIR" \
    --lerobot_repo_id "$LEROBOT_REPO_ID" \
    "${RESUME_ARGS[@]}"

  FIRST_DEMO=0
done

{
  echo "job_id=${SLURM_JOB_ID:-}"
  echo "host=$(hostname)"
  echo "time=$(date --iso-8601=seconds)"
  echo "demo_ids=${DEMO_IDS}"
} > "$COMPLETE_MARKER"
rm -f "$RUNNING_MARKER" "$FAILED_MARKER"
cleanup_shared_permissions

echo "Shard finished successfully."
