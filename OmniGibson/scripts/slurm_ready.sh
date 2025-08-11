#!/bin/bash

# create texture cache on scr-ssd
mkdir -p /scr-ssd/$(whoami)/texturecache
rm -rf ~/.cache/ov/texturecache
ln -s /scr-ssd/$(whoami)/texturecache ~/.cache/ov/texturecache