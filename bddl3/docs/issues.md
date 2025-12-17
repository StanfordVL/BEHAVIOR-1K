
## Trouble Shooting

### Unable to initialize EGL

1. Is nvidia driver properly installed? Nvidia drivers prior to 460.80 and 465.27 have a bug with cgroups that prevent utilize all but GPU 0 on a node. You can check your driver version by running nvidia-smi and update if necessary.
2. Is libegl1 installed? You can determine this by `apt list --installed | grep libegl1`
3. Are openGL libraries visible? You can do so by
`export LD_LIBRARY_PATH=/usr/lib/nvidia-<vvv>:$LD_LIBRARY_PATH`
4. There are two ways of setting up openGL library, if the current installation doesn't work, you can try to install with USE_GLAD set to FALSE in [here](https://github.com/StanfordVL/iGibson/blob/master/igibson/render/CMakeLists.txt)
5. If you want to render in headless mode, make sure `$DISPLAY` environment variable is unset, otherwise you might have error `Failed to EGL with glad`, because EGL is sensitive to `$DISPLAY` environment variable.

It is a good idea to run `ldconfig -p | grep EGL` and you should be able to see `libEGL_nvidia` installed.

```
	libEGL_nvidia.so.0 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0
	libEGL_nvidia.so.0 (libc6) => /usr/lib/i386-linux-gnu/libEGL_nvidia.so.0
	libEGL_mesa.so.0 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0
	libEGL.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libEGL.so.1
	libEGL.so.1 (libc6) => /usr/lib/i386-linux-gnu/libEGL.so.1
	libEGL.so (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libEGL.so
	libEGL.so (libc6) => /usr/lib/i386-linux-gnu/libEGL.so
```

Make sure EGL is not linked to mesa, because in order for gibson to work, linking to Nvidia's EGL is required. For example, this will cause problems: `/usr/lib/x86_64-linux-gnu/libEGL.so -> mesa-egl/libEGL.so`

The EGL setup part is borrowed from Erwin Coumans [egl_example](https://github.com/erwincoumans/egl_example). It would be informative to see if that repository can run on your machine.

### Other issues
For other issues, please submit an issue in our [github repository](https://github.com/StanfordVL/iGibson/issues). 
