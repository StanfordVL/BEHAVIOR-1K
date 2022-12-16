#!/usr/bin/env bash

# Make sure you run this script EXACTLY as `bash ./build_docs.sh`

# Activate conda env
source activate omnigibson

# Remove source directory and copy over source files to docs folder
rm -r docs/src
mkdir docs/src
cp -r omnigibson/* docs/src

# Update code source references
python docs/gen_ref_pages.py

# Build the docs (written to ./site)
mkdocs build
