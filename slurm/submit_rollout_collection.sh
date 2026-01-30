#!/bin/bash
# This script submits rollout_collection.sh to multiple QOSes with different array sizes.
# Each submission gets a unique START_FROM value so they process different CSV files.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLLOUT_SCRIPT="${SCRIPT_DIR}/rollout_collection.sh"

# --- QOS Configuration ---
QOS_LIST=(
    "h200_lowest:1024"
    # "h100_lowest:1024"
    # "h100_core_shared:256"
)

# --- Submission Logic ---
START_FROM=0

echo "=== Rollout Collection Multi-QOS Submission ==="
echo "Script: ${ROLLOUT_SCRIPT}"
echo ""

for entry in "${QOS_LIST[@]}"; do
    # Parse QOS name and array size
    qos_name="${entry%%:*}"
    array_size="${entry##*:}"
    
    # Calculate array range (0-indexed)
    array_max=$((array_size - 1))
    
    echo "Submitting to QOS: ${qos_name}"
    echo "  Array size: ${array_size} (0-${array_max})"
    echo "  START_FROM: ${START_FROM}"
    
    # Submit the job with overridden QOS and array range
    job_id=$(sbatch \
        --qos="${qos_name}" \
        --array="0-${array_max}" \
        "${ROLLOUT_SCRIPT}" "${START_FROM}" \
        | awk '{print $4}')
    
    echo "  Submitted job ID: ${job_id}"
    echo ""
    
    # Increment START_FROM for the next QOS
    # Each array task processes NUM_JOBS=1 file, so total files = array_size
    START_FROM=$((START_FROM + array_size))
done

echo "=== All jobs submitted ==="
echo "Total CSV files covered: $((START_FROM - 1))"

