# **Known Issues & Troubleshooting**

## 🤔 **Known Issues**

??? question "How can I parallelize running multiple scenes in OmniGibson?"

    Currently, to run multiple scenes in parallel, you will need to launch separate instances of the OmniGibson environment. While this introduces some overhead due to running multiple instances of IsaacSim, we are actively working on implementing parallelization capabilities. Our goal is to enable running multiple scenes within a single instance, streamlining the process and reducing the associated overhead.

## 🧯 **Troubleshooting**

??? question "I cannot open Omniverse Launcher AppImage on Linux"

    You probably need to [install FUSE](https://github.com/AppImage/AppImageKit/wiki/FUSE) to run the Omniverse Launcher AppImage.

??? question "OmniGibson is stuck at `HydraEngine rtx failed creating scene renderer.`"

    `OmniGibson` is likely using an unsupported GPU (default is id 0). Run `nvidia-smi` to see the active list of GPUs, and select an NVIDIA-supported GPU and set its corresponding ID when running `OmniGibson` with `export OMNIGIBSON_GPU_ID=<ID NUMBER>`.