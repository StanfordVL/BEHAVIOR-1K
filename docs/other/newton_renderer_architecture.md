# Newton Renderer Architecture Handoff

This document defines the implementation boundary for interchangeable RGB
renderers on the Newton backend. Rendering must remain independent of the
physics solver so users can select a fast training renderer or a higher-quality
path tracer without changing environments, tasks, objects, or robots.

Last updated: July 22, 2026.

## Decision

Implement a camera-renderer interface, not a viewer interface:

```text
Newton model and state
        -> renderer-neutral scene and frame data
        -> selected camera renderer
        -> RGB [environment, camera, height, width, channel]
```

A viewer is an optional human-facing visualization tool. A camera renderer is
an environment sensor that produces observations. `ViewerGL` may remain the
debug viewer, but task observations must not depend on it.

Initial backends:

| Backend | Purpose | Priority |
| --- | --- | --- |
| `null` | Physics-only execution | Required |
| `newton_warp` | Fast, batched RGB using `newton.sensors.SensorTiledCamera` | Implement first |
| `ovrtx` | Higher-quality path-traced RGB using OVRTX | Implement second |
| `nyx` | Optional batched path-traced RGB adapter | Feasibility prototype after the common interface works |

Nyx is currently a Genesis camera plugin, not a Newton renderer. Do not make it
a required dependency or design the common API around its private types.

## Public Configuration

Preserve one renderer-independent configuration shape. Backend-specific
settings belong in a nested block and must not leak into higher-level code.

```yaml
render:
  backend: newton_warp       # null, newton_warp, ovrtx, or nyx
  device: cuda:0
  resolution: [256, 256]
  cameras_per_env: 1
  frequency: 1               # render every N environment steps
  outputs: [rgb]

  newton_warp:
    textures: true
    shadows: false

  ovrtx:
    preset: performance
    asynchronous: true

  nyx:
    samples_per_pixel: 2
    denoiser: true
```

Unsupported backends or outputs must fail with a clear error. Optional backend
packages must be imported lazily.

## Common Interface

The exact types may follow existing OmniGibson conventions, but the lifecycle
and responsibilities should match:

```python
class CameraRenderer(ABC):
    capabilities: RendererCapabilities

    def build(self, scene: RenderScene) -> None: ...
    def update(self, frame: RenderFrame) -> None: ...
    def render(self, cameras: CameraBatch) -> RenderOutput: ...
    def reset(self, env_ids=None) -> None: ...
    def close(self) -> None: ...
```

`build()` uploads static resources once. `update()` synchronizes only dynamic
state. `render()` returns device-resident observations. `reset()` clears any
per-environment temporal accumulation or renderer history.

`RendererCapabilities` should report, at minimum:

- supported outputs, initially `rgb`
- batched-world and batched-camera support
- rigid, deformable-mesh, and particle support
- textures, transparency, and path tracing
- asynchronous rendering
- required device and optional package constraints

Keep capability fields for depth, normals, segmentation, optical flow, and
object IDs even though RGB is the initial requirement. A backend need not
support every capability.

## Renderer-Neutral Data

Separate static scene data from per-frame data.

`RenderScene` contains:

- mesh geometry, UVs, normals, and instance relationships
- canonical materials and texture references
- lights and environment maps
- stable object, shape, and environment identifiers
- fixed particle capacities and deformable topology

`RenderFrame` contains:

- rigid instance transforms
- deformable vertex positions
- active particle positions, radii, and visibility masks
- object activation and visibility masks
- light updates when required

`CameraBatch` contains:

- per-environment camera transforms
- intrinsics, resolution, clipping range, and projection model
- camera-to-environment ownership

The primary output contract is:

```python
RenderOutput.rgb  # [num_envs, num_cameras, height, width, 3]
```

Internally this should remain a Warp or backend-native GPU array. Convert to the
Torch observation API with zero-copy interop where possible. Define whether RGB
is `uint8` sRGB or floating-point linear RGB in the common contract; do not let
backends silently disagree.

## Performance Requirements

The steady-state path must not copy Newton state through the CPU:

```text
Newton GPU state -> CPU -> renderer GPU   # prohibited in the normal frame loop
```

Required behavior:

- Upload static meshes and textures once during `build()`.
- Update transforms and dynamic geometry in device memory.
- Allocate observation buffers once and reuse them.
- Render only requested outputs; RGB-only execution must not compute depth or
  segmentation unnecessarily.
- Support rendering every N physics steps.
- Support selecting a subset of environments for rendering.
- Keep physics and rendering devices configurable for future multi-GPU use.
- Avoid per-environment Python loops in build-independent frame operations.

Asynchronous rendering is optional and backend-specific. The environment must
define whether an asynchronous observation represents the current physics frame
or a deliberately delayed frame.

## Material and Feature Semantics

Define a small canonical material representation and lower it into each
backend. At minimum it should include base color/texture, roughness, metallic,
opacity, emissive color, and two-sidedness.

Pixel equivalence across renderers is not a goal. The contract is consistent
camera geometry, object identity, image shape, color space, and lifecycle.
Lighting, transparency, particles, and path-traced effects may differ and
should be documented through capabilities.

Dynamic topology should not be required in the initial interface. Slicing,
dicing, and transition-rule outputs should use the migration's preallocated
activation pools. Deformables update fixed-topology vertices, and particle
systems update within fixed capacities.

## Implementation Sequence

1. Add the common types, factory, configuration validation, and `null` backend.
2. Implement `newton_warp` with one RGB camera in one environment.
3. Extend `newton_warp` to batched environments and cameras with device-resident
   output and partial reset coverage.
4. Connect RGB observations to the existing environment observation API.
5. Add deformable and particle frame updates.
6. Implement OVRTX as a camera backend. Reuse useful `ViewerRTX` scene-loading
   code, but do not expose the viewer as the observation abstraction.
7. Prototype a Nyx adapter and retain it only if Newton state and RGB can remain
   device-resident and its batched performance justifies the dependency.

Do not block the common interface or fast renderer on Nyx integration.

## Acceptance Criteria

The first implementation is complete when:

- A configuration switch selects `null` or `newton_warp` without branching in
  environment, scene, object, robot, or task code.
- Headless RGB works for one and multiple environments.
- Output shape, dtype, color space, device, and frame timing are tested.
- Multiple cameras preserve correct environment/camera ordering.
- Partial environment reset produces no stale transforms or image history.
- Object activation and visibility changes appear on the next specified render
  frame.
- The steady-state batched path has no required CPU readback or per-environment
  Python loop.
- Closing and rebuilding releases renderer resources safely.
- Missing optional dependencies produce actionable errors.
- Existing physics-only and OpenGL debugging workflows continue to work.

Benchmark each real backend at common resolutions and batch sizes, including
`128x128` and `256x256` at 1, 16, 64, and 256 environments where memory permits.
Report warm steady-state render time, total observation latency, peak GPU
memory, and whether output remains device-resident.

## Non-Goals for the First Pass

- Identical images across renderers
- Full RTX material parity
- Every sensor modality
- Dynamic mesh topology
- Simultaneous multi-renderer training
- Committing to Nyx before its Newton adapter is measured

