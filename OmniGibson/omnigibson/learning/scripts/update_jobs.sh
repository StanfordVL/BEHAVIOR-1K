#!/bin/bash
# This script is called by cron
mkdir -p /vision/u/$(whoami)/BEHAVIOR-1K/outputs/sc
cd /vision/u/$(whoami)/BEHAVIOR-1K && /usr/local/bin/sbatch OmniGibson/omnigibson/learning/scripts/update_jobs.sbatch.sh