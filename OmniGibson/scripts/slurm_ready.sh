#!/bin/bash

# create texture cache on scr-ssd
mkdir -p /scr-ssd/wsai/texturecache
rm -rf ~/.cache/ov/texturecache
ln -s /scr-ssd/wsai/texturecache ~/.cache/ov/texturecache