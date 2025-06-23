#!/bin/bash
set -e

apt-get update
apt-get install -y zip unzip
pip install tqdm

OMNIGIBSON_NO_OMNIVERSE=1 python -c "from omnigibson.utils.asset_utils import download_key; download_key()"

# First, unzip the dataset into the temp directory
unzip /ig_pipeline/artifacts/og_dataset.zip -d /tmp/og_dataset

# Then, run the conversion script
cd /ig_pipeline
object_list=( artifacts/aggregate/objects/*/* )
object_count=${#object_list[@]}
echo "Object count: $object_count"
USDIFY_BATCH_SIZE=100
for ((batch_from=0; batch_from<object_count; batch_from+=USDIFY_BATCH_SIZE)); do
  batch_to=$(( $batch_from+$USDIFY_BATCH_SIZE ))
  python -m b1k_pipeline.usd_conversion.usdify_objects_process $batch_from $batch_to /tmp/og_dataset
done
python -m b1k_pipeline.usd_conversion.usdify_scenes /tmp/og_dataset

# Do some cleanup to reduce file size
rm -rf /tmp/og_dataset/objects/*/*/shape
rm -rf /tmp/og_dataset/objects/*/*/material
rm -rf /tmp/og_dataset/objects/*/*/*.urdf
rm -rf /tmp/og_dataset/scenes/*/urdf

# Then, re-zip the dataset
cd /tmp
tar -czvf /ig_pipeline/artifacts/og_dataset_encrypted.tar.gz og_dataset
