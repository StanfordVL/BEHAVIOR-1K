#!/usr/bin/env bash
set -eo pipefail

# Check that the first argument is either og_dataset or og_dataset_demo
if [ "$1" != "og_dataset" ] && [ "$1" != "og_dataset_demo" ]; then
    echo "ERROR: First argument must be either og_dataset or og_dataset_demo"
    exit 1
fi

# Get the pipeline path
SCRIPTDIR="$(dirname "$0")"
SCRIPTDIR="$(cd "$SCRIPTDIR" && pwd)"
PIPELINEDIR="${SCRIPTDIR}/.."
PIPELINEDIR="$(cd "$PIPELINEDIR" && pwd)"

# Create a temp dir
TMP_DIR=$(mktemp -d)
echo "TMP_DIR: $TMP_DIR"

# Change to the temp dir
cd $TMP_DIR

# Unpack the full dataset into the og_dataset directory
mkdir og_dataset
cd og_dataset
unzip -q $PIPELINEDIR/artifacts/${1}.zip
echo "Unpacked og_dataset"

# Unpack the sampled tasks over the og_dataset
# If $1 is og_dataset_demo, then we need to unzip only a subset.
if [ "$1" == "og_dataset_demo" ]; then
    unzip -l $PIPELINEDIR/sampled_tasks.zip | grep 'Rs_int' | awk '{print $4}' | xargs -I{} unzip $PIPELINEDIR/sampled_tasks.zip '{}'
else
    unzip -q $PIPELINEDIR/sampled_tasks.zip
fi

# Check that both _best.json and _template.json files exist
NUM_BEST=$(find . -type f | grep "_best\.json" | wc -l)
NUM_TEMPLATE=$(find . -type f | grep "_template\.json" | wc -l)
if [ $NUM_BEST -eq 0 ] || [ $NUM_TEMPLATE -eq 0 ]; then
    echo "ERROR: _best.json and _template.json files not found in ${TMP_DIR}"
    exit 1
fi

# Tar the og dataset using 7za
cd $TMP_DIR
7za a ${1}_sampled.tar og_dataset > /dev/null 2>&1
echo "Tared og_dataset"
gzip -9 ${1}_sampled.tar
echo "Gzipped og_dataset"

# Check that the tar.gz file exists
if [ ! -f ${1}_sampled.tar.gz ]; then
    echo "ERROR: ${1}_sampled.tar.gz not found in ${TMP_DIR}"
    exit 1
fi

# Move the tar.gz file to the artifacts directory
mv ${1}_sampled.tar.gz $PIPELINEDIR/artifacts/
echo "Moved ${1}_sampled.tar.gz to $PIPELINEDIR/artifacts/"

# Clean up
rm -rf $TMP_DIR
echo "Cleaned up"
