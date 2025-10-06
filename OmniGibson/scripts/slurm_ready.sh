#!/bin/bash

# create texture cache on scr-ssd
target="/scr-ssd/$(whoami)/texturecache"
mkdir -p "$target"
if [ "$(readlink -f ~/.cache/ov/texturecache 2>/dev/null)" != "$(realpath "$target")" ]; then
    ln -sfn "$target" ~/.cache/ov/texturecache
fi