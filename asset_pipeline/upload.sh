#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
VERSION=$(cat VERSION)

hf upload behavior-1k/zipped-datasets artifacts/og_dataset.zip "behavior-1k-assets-${VERSION}.zip" --repo-type=dataset
