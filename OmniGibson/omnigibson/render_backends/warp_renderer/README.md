# Experimental Warp render backend

Select physics and rendering independently before the simulator is created:

```bash
conda activate newton
OMNIGIBSON_PHYSICS_BACKEND=newton OMNIGIBSON_RENDER_BACKEND=warp \
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 taskset -c 0-7 \
python -m omnigibson.examples.objects.visualize_object
```

These environment variables supply global defaults; explicit assignments to `gm.PHYSICS_BACKEND`
or `gm.RENDER_BACKEND` in an example still take precedence. Without overrides, defaults remain
PhysX + Kit. `OMNIGIBSON_DEVICE=cuda:0` optionally selects GPU physics for environments that do
not specify a device themselves; some examples still assume CPU tensors. Warp rendering uses CUDA
even with CPU physics. `OMNIGIBSON_HEADLESS=1` disables the window, not RGB camera capture.

The robot-control example also retains its convenience option:

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 taskset -c 0-7 \
python -m omnigibson.examples.robots.robot_control_example --quickstart --renderer warp
```

The global viewer appears as ordinary `env.step()`, `og.sim.step()`, or `og.sim.render()` calls
tick the renderer. Existing `og.sim.viewer_camera.set_position_orientation()` calls still work.
Left-drag orbits, right-drag looks around, middle-drag pans, and scrolling dollies. Resize the
window or double-click its title bar to maximize; the image scales without changing the camera's
render resolution. Escape / closing the window exits the process. No camera WASD bindings are added.
Robot keyboard control still needs an example's controller; choosing a renderer does not create one.

## Requirements and limits

- The existing Newton-compatible environment, a CUDA GPU, `pyglet`, `OpenEXR`, and CUDA-enabled
  Open Image Denoise 2.5.1. Extract its official Linux archive under `$CONDA_PREFIX/opt`, or set
  `OIDN_LIBRARY` to `libOpenImageDenoise.so.2`. No Kit application or RT cores are needed.
- Default rendering is `pt1_oidn`: one sample per pixel, eight bounces, high-quality OIDN, fixed
  seed, ambient 160, exposure 0.0007717915155850135.
- Currently supports Newton rigid geometry and RGB cameras (opaque RGBA observations). Other
  sensor modalities raise `NotImplementedError`; tiled rendering is not supported.
- Geometry is re-extracted when the physics model changes, including object additions/removals.
  Dynamic light/material edits and particle/fluid surfaces are not yet supported. Images are
  not pixel-identical to Kit/RTX.
- Examples that directly call Kit UI/input APIs, use Kit's camera-controller helpers, or request
  unsupported modalities need adaptation. This backend does not emulate the full Isaac Sim UI.

The simulator uses shared render-backend hooks for camera navigation and cleanup. Newton ticks
the selected renderer and consults `requires_physics_visual_shapes` to avoid building duplicate
visual meshes for renderers that extract directly from USD.

## Smoke test

From `OmniGibson/`, with the `newton` environment active:

```bash
OMNIGIBSON_HEADLESS=1 OMNIGIBSON_PHYSICS_BACKEND=newton OMNIGIBSON_RENDER_BACKEND=warp \
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 taskset -c 0-7 \
pytest -q tests/test_warp_backend.py
```
