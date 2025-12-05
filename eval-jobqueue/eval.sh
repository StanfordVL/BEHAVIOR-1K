#!/bin/bash
#SBATCH --job-name=b1k-eval
#SBATCH --output=/vision/group/behavior/eval-logs/%j.log
#SBATCH --error=/vision/group/behavior/eval-logs/%j.log
#SBATCH --time=14-00:00:00
#SBATCH --account=cvgl
#SBATCH --partition=svl
#SBATCH --nodes=1
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=60G
#SBATCH --array=0-47

# Use the base drive as /scr-ssd if it exists, otherwise use /scr
BASE_DRIVE=/scr-ssd
if [ ! -d "$BASE_DRIVE" ]; then
    BASE_DRIVE=/scr
fi

# Set the tempdir to be the base drive /tmp/$USER
TMPDIR=$BASE_DRIVE/$USER/tmp
mkdir -p $TMPDIR
export TMPDIR

GOOGLE_APPLICATION_CREDENTIALS=/vision/group/behavior/credentials.json
export GOOGLE_APPLICATION_CREDENTIALS

eval "$(conda shell.bash hook)"
conda activate /vision/group/behavior/b1k-eval-condaenv
while true; do
    echo "Starting a new run"
    python -u -m omnigibson.learning.eval_with_jobqueue 2>&1 | python /vision/group/behavior/b1k-eval-mainrepo/stream_logs.py
    echo "Evaluation job finished. Sleeping for 10 seconds..."
    sleep 10
done
