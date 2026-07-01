#!/bin/bash
#SBATCH --job-name=download_gcs_rawdata
#SBATCH --account=vision
#SBATCH --partition=svl
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=/vision/u/%u/BEHAVIOR-1K/outputs/sc/download_gcs_rawdata/%x-%A_%a.out
#SBATCH --error=/vision/u/%u/BEHAVIOR-1K/outputs/sc/download_gcs_rawdata/%x-%A_%a.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch --array=0-99 download_gcs_rawdata.sbatch [options]
  sbatch download_gcs_rawdata.sbatch --task_id 16 [options]

Options:
  --task_id ID                 Overrides SLURM_ARRAY_TASK_ID.
  --bucket_uri URI             Default: gs://behavior-demos
  --target_root PATH           Default: /vision/group/behavior/2026-challenge-rawdata
  --source_template TEMPLATE   Default: {bucket_uri}/task-{task_id:04d}/episode_{episode_id}.hdf5
  --instance_ids IDS           Comma-separated instance ids.
  --instance_ids_file PATH     One instance id per line, or CSV with id first.
  --instance_ids_file_template PATH
                               Template with {task_id}, e.g. /path/task-{task_id}.txt.
  --gcs_manifest_csv PATH      CSV from generate_gcs_rawdata_manifest.py.
  --first_instance_id ID       Default: 0
  --num_instances N            Default: 200
  --instance_stride N          Default: 1
  --workers N                  Default: 4
  --tool TOOL                  auto, python-gcs, gcloud, gsutil, or full path.
  --credentials PATH           Google credentials JSON for --tool python-gcs.
  --repo_root PATH             BEHAVIOR-1K checkout.
  --dry_run                    Print planned downloads only.
EOF
}

TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
BUCKET_URI="gs://behavior-demos"
TARGET_ROOT="/vision/group/behavior/2026-challenge-rawdata"
SOURCE_TEMPLATE="{bucket_uri}/task-{task_id:04d}/episode_{episode_id}.hdf5"
INSTANCE_IDS=""
INSTANCE_IDS_FILE=""
INSTANCE_IDS_FILE_TEMPLATE=""
GCS_MANIFEST_CSV=""
FIRST_INSTANCE_ID="0"
NUM_INSTANCES="200"
INSTANCE_STRIDE="1"
WORKERS="4"
TOOL="auto"
CREDENTIALS=""
REPO_ROOT=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task_id)
      TASK_ID="$2"
      shift 2
      ;;
    --bucket_uri)
      BUCKET_URI="$2"
      shift 2
      ;;
    --target_root)
      TARGET_ROOT="$2"
      shift 2
      ;;
    --source_template)
      SOURCE_TEMPLATE="$2"
      shift 2
      ;;
    --instance_ids)
      INSTANCE_IDS="$2"
      shift 2
      ;;
    --instance_ids_file)
      INSTANCE_IDS_FILE="$2"
      shift 2
      ;;
    --instance_ids_file_template)
      INSTANCE_IDS_FILE_TEMPLATE="$2"
      shift 2
      ;;
    --gcs_manifest_csv)
      GCS_MANIFEST_CSV="$2"
      shift 2
      ;;
    --first_instance_id)
      FIRST_INSTANCE_ID="$2"
      shift 2
      ;;
    --num_instances)
      NUM_INSTANCES="$2"
      shift 2
      ;;
    --instance_stride)
      INSTANCE_STRIDE="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --tool)
      TOOL="$2"
      shift 2
      ;;
    --credentials)
      CREDENTIALS="$2"
      shift 2
      ;;
    --repo_root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --dry_run)
      DRY_RUN=1
      shift
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

if [[ -z "$TASK_ID" ]]; then
  echo "Missing --task_id or SLURM_ARRAY_TASK_ID." >&2
  usage >&2
  exit 2
fi

if [[ -z "$REPO_ROOT" ]]; then
  if [[ -d "/vision/u/${USER}/BEHAVIOR-1K" ]]; then
    REPO_ROOT="/vision/u/${USER}/BEHAVIOR-1K"
  else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
  fi
fi

TASK_ID_PADDED="$(printf '%04d' "$TASK_ID")"
if [[ -z "$INSTANCE_IDS_FILE" && -n "$INSTANCE_IDS_FILE_TEMPLATE" ]]; then
  INSTANCE_IDS_FILE="${INSTANCE_IDS_FILE_TEMPLATE//\{task_id\}/$TASK_ID_PADDED}"
fi

mkdir -p "/vision/u/${USER}/BEHAVIOR-1K/outputs/sc/download_gcs_rawdata"
mkdir -p "$TARGET_ROOT"

echo "SLURM_JOBID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "task_id=${TASK_ID}"
echo "bucket_uri=${BUCKET_URI}"
echo "target_root=${TARGET_ROOT}"
echo "source_template=${SOURCE_TEMPLATE}"
echo "repo_root=${REPO_ROOT}"
echo "Current time: $(date)"

CONDA_SH="/vision/u/${USER}/miniconda3/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
  CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
fi
set +u
source "$CONDA_SH"
conda activate behavior
set -u

cd "$REPO_ROOT"

ARGS=(
  --task_id "$TASK_ID"
  --bucket_uri "$BUCKET_URI"
  --target_root "$TARGET_ROOT"
  --source_template "$SOURCE_TEMPLATE"
  --first_instance_id "$FIRST_INSTANCE_ID"
  --num_instances "$NUM_INSTANCES"
  --instance_stride "$INSTANCE_STRIDE"
  --workers "$WORKERS"
  --tool "$TOOL"
  --manifest "$TARGET_ROOT/download_manifests/task-$(printf '%04d' "$TASK_ID").csv"
)

if [[ -n "$INSTANCE_IDS" ]]; then
  ARGS+=(--instance_ids "$INSTANCE_IDS")
fi
if [[ -n "$INSTANCE_IDS_FILE" ]]; then
  ARGS+=(--instance_ids_file "$INSTANCE_IDS_FILE")
fi
if [[ -n "$GCS_MANIFEST_CSV" ]]; then
  ARGS+=(--gcs_manifest_csv "$GCS_MANIFEST_CSV")
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
  ARGS+=(--dry_run)
fi
if [[ -n "$CREDENTIALS" ]]; then
  ARGS+=(--credentials "$CREDENTIALS")
fi

python OmniGibson/scripts/learning/download_gcs_rawdata.py "${ARGS[@]}"
