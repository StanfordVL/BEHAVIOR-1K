#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=h100_core_shared
#SBATCH --account=clear
#SBATCH --job-name=test_behavior1k_on_h100
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

export OMNIGIBSON_HEADLESS=1
python -u -c "import omnigibson as og; og.launch()"