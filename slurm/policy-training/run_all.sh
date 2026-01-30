#!/bin/bash

# Submit all policy training jobs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting all policy training jobs..."

sbatch "$SCRIPT_DIR/act.sh"
sbatch "$SCRIPT_DIR/actddp.sh"
sbatch "$SCRIPT_DIR/customdp.sh"
sbatch "$SCRIPT_DIR/dp.sh"
sbatch "$SCRIPT_DIR/dpddp.sh"
sbatch "$SCRIPT_DIR/pi0.sh"
sbatch "$SCRIPT_DIR/pi0ddp.sh"

echo "All jobs submitted."

