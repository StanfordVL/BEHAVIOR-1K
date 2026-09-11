#!/usr/bin/env bash

# Remove source directory and copy over source files to docs folder
rm -rf docs/src
mkdir docs/src
cp -r omnigibson/* docs/src

# Update code source references
rm -rf docs/reference
python docs/gen_ref_pages.py

# Build the docs (written to ./site)
mkdocs build
