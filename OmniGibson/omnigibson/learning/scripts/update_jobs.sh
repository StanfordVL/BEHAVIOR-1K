#!/bin/bash
source /home/svl/miniconda3/etc/profile.d/conda.sh
conda activate behavior
cd /vision/u/$USER/BEHAVIOR-1K && python OmniGibson/omnigibson/learning/scripts/update_jobs.py