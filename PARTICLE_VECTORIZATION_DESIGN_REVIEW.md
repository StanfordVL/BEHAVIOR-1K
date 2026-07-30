# Particle-State Vectorization — Implemented Design and Call Flow

**Branch:** `vec/particle-system`

## 1. Review scope

This document describes only the code implemented on this branch. It is for a
reviewer who knows OmniGibson and the multi-environment goal, but needs to understand
this PR's implementation, lifecycle, and design constraints.

```text
ParticleViewAPI
    unified cross-scene particle positions and identity
                         |
                         v
TensorizedObjectSystemState
    shared (scene, object, particle-system) tensor lifecycle
                         |
              +----------+-----------+
              |                      |
              v                      v
     ContainedParticles       ContactParticles
        Contains/Filled             Covered
```

The PR also changes simulator graph orchestration, particle-system invalidation, and
rigid-body staging. Future work (include `ParticleModifier` and other smaller fixes) is in
`PARTICLE_VECTORIZATION_FUTURE_WORK.md`.

## 2. Code-change map

| Area | Implemented change |
|---|---|
| `particle_view_utils.py` | Added `ParticleViewAPI`: one flat cross-scene reader with three family-specific inputs. |
| `simulator.py` | Added topology initialization, host preparation, captured particle updates, graph replay, and lazy cache refresh ordering. |
| `tensorized_object_system_state.py` | Added the `(scene, object, system)` tensorized-state front-end. |
| `contains.py` | Reimplemented `ContainedParticles` as a batched GPU count; rewired `Contains` and `Filled`. |
| `contact_particles.py` | Reimplemented physical-particle contact as a batched GPU count with lazy detailed indices. |
| `covered.py` and consumers | Read the cheap tensor count or lazy particle-index result explicitly. |
| `tensorized_state.py` | Made pinned `VALUES_CPU` the tensorized value cache and added graph/cache dirty lifecycle. |
| particle systems | Added metadata invalidation and isolated macro-physical PhysX-view refresh. |
| `usd_utils.py` | Split rigid pose acquisition into host staging and captured H2D/pose-matrix work. |
| `physx_utils.py` | Corrected point-instancer prototype-index wrapping for micro-instancer lifecycle operations. |
| `scene_base.py` | Treat runtime particle-system activation as a topology change. |
| transition rules / particle modifiers | Consume the explicit lazy particle-index result where particle identities are required. |

## 3. Topology construction: two ordered passes

`Simulator.update_handles()` is the topology boundary. Particle states require two
passes because tensorized state tables reference indices and buffers created by the
unified ViewAPIs.

```text
Simulator.update_handles()
    |
    +-- flush USD changes into PhysX
    +-- recreate simulator-wide physics_sim_view
    +-- refresh object handles
    +-- refresh each MacroPhysicalParticleSystem
    |       particles_sim_view -> particles_view
    |
    +-- PASS 1: unified ViewAPIs
    |       RigidContactAPI.initialize_view()
    |       RigidBodyViewAPI.initialize_view()
    |       ArticulatedObjectViewAPI.initialize_view()
    |       ControllableObjectViewAPI.initialize_view()
    |       ParticleViewAPI.initialize_view()
    |
    +-- PASS 2: dependent tensorized states
            each registered TensorizedState.initialize_view()
```

Pass 1 must precede Pass 2:

- `ContainedParticles.initialize_view()` maps parent links through
  `RigidBodyViewAPI` and particle entries through `ParticleViewAPI`.
- `ContactParticles.initialize_view()` uses both plus `AABB`.
- Tensorized states allocate `VALUES`, pinned `VALUES_CPU`, and Warp wrappers
  whose addresses are captured by the per-step graph.

If `update_handles()` occurs while PhysX is stepping, the physics/ViewAPI pass runs
immediately and tensorized-state initialization is deferred. Those state views read
Fabric world geometry and cannot safely build mid-step.

### Topology triggers

```text
Simulator.play()
    -> update_handles()

objects initialized as a batch in _non_physics_step()
    -> update_handles() once after the batch

Scene.get_system() activates a system while playing
    -> update_handles()
    -> new ParticleViewAPI entry and SYS_IDXS column

generic object/prim topology mutation
    -> update_handles()

macro-physical add/remove in an already registered system
    -> local dedicated particle-view refresh
    -> no global topology build
```

System activation changes the registry, tensorized `SYS_IDXS`, and captured shapes,
so it is real topology. During object initialization, `Scene.get_system()` defers
the refresh until the batch ends to avoid exposing partially initialized states.

## 4. ParticleViewAPI topology build and flat contract

An entry is one `(scene_idx, system_name)` pair. The same system in two scenes
creates two entries.

```text
ParticleViewAPI.initialize_view()
    |
    +-- clear old registry/views/buffers
    +-- scan initialized scene.active_systems
    +-- classify micro-physical / macro-physical / macro-visual
    +-- build ordered _entries and _family_keys
    +-- allocate registry-sized _entry_scene and _entry_start
    +-- allocate stable GPU count scalars
    |       PARTICLE_COUNT
    |       _macro_physical_count
    |       _macro_visual_count
    +-- _rebuild_layout()
    +-- _rebuild_micro_physical_metadata()
    +-- _rebuild_macro_physical_metadata()
    +-- _rebuild_macro_visual_metadata()
    +-- clear tracked metadata-dirty flags
    +-- mark graph_dirty
```

Initialization returns with complete entry ranges and family metadata; no later read
finishes initialization lazily.

```text
PARTICLE_POSITIONS
 [ entry 0 ][ entry 1 ][ entry 2 ][ unused capacity ]
   ^start[0]  ^start[1]  ^start[2]

PARTICLE_SCENE_INDEX         scene for every valid row
PARTICLE_ENTRY_INDEX         (scene, system) entry for every valid row
VISUAL_PARTICLE_ORIENTATION  orientation for macro-visual rows
PARTICLE_COUNT               live valid rows, stored as a GPU scalar
```

`_entry_ranges[key] = (start, count)` exposes each system's slice. Slice order must
match the system's own particle order; lazy contact masks can therefore return
system-local indices without another identity table.

The flat arrays reserve at least 1024 rows once a registry exists and grow
geometrically. They do not shrink inside an initialized topology.

## 5. Simulator per-step call flow

```text
Simulator.step()
    |
    +-- _sim_context.step(...)                    PHYSICS
    |
    +-- _non_physics_step()
            |
            +-- initialize pending objects
            |       -> update_handles(), if needed
            +-- each active system.update()
            +-- _refresh_state_caches(dt)
            |       +-- RigidBodyViewAPI.read_from_physx()
            |       +-- ArticulatedObjectViewAPI.read_from_physx()
            |       +-- ParticleViewAPI.prepare_step_host()
            |       +-- synchronize preparation work
            |       +-- _capture_warp_graph(dt)
            +-- per-object UpdateStateMixin.update()
            +-- visual updates
            +-- transition rules
```

`step_physics()` uses the same refresh with `dt=0`. Direct pose/joint mutations
and lazy tensorized reads can also call `_refresh_state_caches(dt=0)`, centralizing
the ordering.

### Capture and replay

```text
_capture_warp_graph(dt)
    |
    +-- each TensorizedState.pre_update(dt)       OUTSIDE GRAPH
    |
    +-- if graph_dirty:
    |       wp.ScopedCapture:
    |           RigidBodyViewAPI.update()
    |           ParticleViewAPI.update_positions_gpu()
    |           RigidContactAPI.update()
    |           each TensorizedState.global_update()
    |       store graph; graph_dirty = False
    |
    +-- wp.capture_launch(stored graph)           GRAPH REPLAY
    +-- synchronize
    +-- each TensorizedState.post_update()        OUTSIDE GRAPH
```

When no tensorized state has values, ViewAPI updates still run directly so ad-hoc
consumers receive fresh data.

Required captured ordering:

```text
RigidBodyViewAPI.update()
    pinned-host H2D + current POSE_MATRICES
                    |
                    v
ParticleViewAPI.update_positions_gpu()
    macro-physical H2D/scatter
    macro-visual parent-pose transform
                    |
                    v
ContainedParticles / ContactParticles
    consume current positions and link poses
```

## 6. What is inside the graph

| Work | Inside? | Reason |
|---|---:|---|
| `RigidBodyViewAPI.read_from_physx()` | No | PhysX host API. |
| `ParticleViewAPI.prepare_step_host()` | No | Metadata plus PhysX/Fabric/system getters. |
| micro `SelectPrims`, `GetPaths`, scatter | No | Fabric wrapper and row order are transient. |
| macro-physical `get_transforms()` | No | PhysX host read into pinned staging. |
| cloth/untracked visual getters | No | Per-system fallback backend calls. |
| `TensorizedState.pre_update()` | No | CPU snapshot/bookkeeping. |
| `RigidBodyViewAPI.update()` | Yes | Stable H2D and pose-matrix kernel. |
| macro-physical H2D/scatter | Yes | Fixed-capacity buffers and launch. |
| macro-visual transform | Yes | Stable metadata and live `POSE_MATRICES`. |
| containment/contact kernels | Yes | Fixed-capacity input and topology tables. |
| `VALUES -> VALUES_CPU` Warp copy | Yes | Stable pinned destination. |
| `TensorizedState.post_update()` | No | CPU comparison and callbacks. |

The boundary is not CPU versus CUDA: the micro scatter uses CUDA but remains outside
capture because its Fabric source cannot be safely replayed.

## 7. Three family data paths

```text
micro physical
    Fabric point-instancer positions
        -> path-to-entry resolution -> flat output

macro physical
    cross-scene PhysX transforms
        -> pinned CPU -> GPU -> offset/scatter -> flat output

macro visual
    parent POSE_MATRICES + scale + local matrix
        -> world-transform kernel
        -> flat position + visual orientation
```

### 7.1 Micro-physical

Examples: water, milk, sand, rice.

`_rebuild_micro_physical_metadata()` maps point-instancer paths to entries and
records the maximum current instancer width. Each refresh performs one Fabric
`SelectPrims`. Because `GetPaths()` row order is not stable, a small host loop
resolves one destination per instancer, followed by one 2-D Warp scatter:

```text
thread = (instancer row, particle row)
destination = entry_start[path_to_entry[path]] + particle row
```

This is O(active instancers), not O(particles), in Python. Selection, path mapping,
and scatter stay outside capture. Destination buffers grow on demand, but that
growth does not dirty the graph because the buffers are not captured.

### 7.2 Macro-physical

Examples: diced/cubed food. Each particle is a rigid body.

Two ownership levels are intentional:

```text
each MacroPhysicalParticleSystem
    particles_sim_view
        -> particles_view
            system-local transform/velocity getters and setters

ParticleViewAPI
    _macro_physical_sim_view
        -> _macro_physical_view
            one read across all macro systems/scenes
```

The local view keeps system APIs independent of ParticleViewAPI lifecycle and
cross-system row mapping. The aggregate view removes per-system reads from the
per-step position path.

Metadata maps each aggregate row to its ParticleViewAPI entry, system-local row, and
rigid-particle center offset.

```text
prepare_step_host()
    _macro_physical_view.get_transforms()
        -> pinned _macro_physical_transforms_host valid prefix

captured update_positions_gpu()
    pinned host -> GPU transforms
        -> scatter kernel
        -> position + quat_rotate(orientation, offset)
        -> entry_start[entry] + local_index
```

Metadata/staging reserve at least 64 rows and grow geometrically. Capture uses the
capacity-sized buffers and launch. `_macro_physical_count` contains the live rows;
threads beyond it return.

#### Membership changes

A rigid-body view snapshots matching bodies, so membership changes must recreate the
view. They do not need to invalidate unrelated simulator views.

```text
MacroPhysicalParticleSystem.add/remove
    +-- mark_particle_metadata_dirty()
    +-- flush USD into PhysX
    +-- recreate particles_sim_view -> particles_view
    |
    +-- next prepare_step_host()
            recreate aggregate _macro_physical_sim_view -> _macro_physical_view
            rewrite metadata valid prefix
            preserve captured buffers if capacity is sufficient
```

Removal calls `particle.remove()` instead of `Simulator.remove_prim()`, whose
contract includes global `update_handles()`. New system activation still uses the
full topology path because it changes the registry.

### 7.3 Macro-visual

Examples: stain/dust attached to object links.

`_rebuild_macro_visual_metadata()` records parent-link index, accumulated scale,
static local particle matrix, destination entry, and system-local row. In graph:

```text
world = parent_pose_matrix * scale_matrix * particle_local_matrix
```

The kernel writes translation and a scale/shear-stripped orientation; containment
uses the orientation for its visual-particle sample offset.

Visual metadata reserves at least 64 rows and grows geometrically. A stable
`_macro_visual_count` gates the capacity-sized launch, so in-capacity attachment or
count changes rewrite the valid prefix without changing capture.

Cloth parents and links absent from `RigidBodyViewAPI` cannot use this kernel. Their
entire entries go into `_macro_visual_fallback_keys` and use system getters in the
host preparation phase.

## 8. Dirty state and graph recapture

Three signals have separate meanings:

| Signal | Meaning | Consumer |
|---|---|---|
| `system.particle_metadata_dirty` | Identity/layout/static family metadata may be stale. | `prepare_step_host()` |
| `TensorizedState.caches_dirty` | Pinned `VALUES_CPU` may be stale. | Next tensorized read or step |
| `TensorizedState.graph_dirty` | Captured pointers/shapes/capacities changed. | Graph manager |

`mark_particle_metadata_dirty()` sets the system flag and marks tensorized values
stale; it does not directly set graph dirty.

`_refresh_dirty_metadata()` scans cached family lists, returns immediately if none
are dirty, rebuilds layout only when counts changed, rebuilds only dirty families,
and clears flags after successful refresh. Full logical family tables may be rebuilt
while reusing capacity-backed arrays.

### Exact graph-dirty rules

| Event | Dirty? | Reason |
|---|---:|---|
| `ParticleViewAPI.initialize_view()` | Yes | Registry arrays/mappings replaced. |
| Any `TensorizedState.initialize_view()` | Yes | State tensors and wrappers reallocated. |
| Flat output exceeds capacity | Yes | Captured pointers and consumer launch capacity change. |
| Macro-physical metadata exceeds capacity | Yes | Captured staging/metadata and launch change. |
| Macro-visual metadata exceeds capacity | Yes | Captured metadata and launch change. |
| New active system/object topology | Yes | Registry/state dimensions change. |
| Count/layout change within capacity | No | Starts, labels, and counts update in place. |
| Macro-physical add/remove within capacity | No | Backend views change; graph buffers do not. |
| Macro-visual rebuild within capacity | No | Existing metadata prefix is rewritten. |
| Micro destination-buffer growth | No | Micro work is outside graph. |
| Physical pose/velocity change | No metadata recapture | Live values are read every refresh. |

Invariant:

> A logical particle count may change without recapture while every pointer and
> launch capacity referenced by the graph remains stable.

```text
capacity = 64; live count = 3
captured launch = 64 threads
threads 0..2 work; threads 3..63 return using the live GPU count
```

`_entry_start` uses the same pattern: its pointer is fixed for the registry; count
changes stage new offsets through pinned host memory into the existing GPU array.

## 9. Tensorized object-system states and cache

`TensorizedObjectSystemState` introduces:

```text
VALUES[scene_idx, object_idx, system_idx, ...]
```

Its topology build creates `OBJ_IDXS`, per-scene `IDX_OBJS`, `SYS_IDXS`, CUDA
`VALUES`, pinned `VALUES_CPU`, previous values, and Warp wrappers.

`TensorizedState.global_update()` launches the subclass kernel and performs a
graph-safe Warp copy to `VALUES_CPU`. Python getters read the pinned mirror without
a per-call GPU synchronization. If `caches_dirty` is true, the first getter invokes
one complete `_refresh_state_caches()`; the re-entrance guard prevents recursive
refreshes. The legacy per-object cache is not used for these values.

## 10. ContainedParticles

`ContainedParticles` stores an int32 count per
`(scene, container, particle system)`.

Topology builds a flat fillable-mesh face table, mesh-to-parent-link mappings,
per-object mesh ranges, entry-to-system mapping, and a visual flag.

In graph it first derives current world-to-mesh transforms from
`POSE_MATRICES`, then launches one thread per
`(particle capacity row, container)`. Threads gate on live count, scene, and entry;
the halfspace test atomically increments `VALUES[scene, object, system]`.

`ContainedParticlesData.n_in_volume` is the cheap tensor result. `.positions` and
`.in_volume` retain the detailed per-object calculation and materialize lazily for
setters/transition rules. Materialization re-derives the count from that mask.
`Contains` and `Filled` interpret `.n_in_volume` directly.

## 11. ContactParticles

`ContactParticles` stores an int32 count per
`(scene, object, physical system)`; visual entries are skipped.

It reuses `AABB.VALUES_WP`, `RigidBodyViewAPI.LINK_MESH_IDS`,
`POSE_MATRICES`, per-object rigid-link ranges, and per-entry contact radii. The
captured kernel launches over `(particle capacity, object count)`. After scene and
AABB rejection, a signed mesh query reports contact if a particle is inside the solid
at any depth or within contact radius of its surface, matching the previous PhysX
`overlap_sphere` behavior.

`ContactParticlesData.count` reads the tensor. `.particle_indices` lazily runs the
same predicate over one system slice and converts its mask into system-local indices.
`Covered` needs only count; removers and transition rules request identities.

## 12. End-to-end dependency flow

```text
PhysX / Fabric / attachment metadata
                    |
                    v
ParticleViewAPI flat snapshot
                    |
          +---------+----------+
          |                    |
          v                    v
ContainedParticles       ContactParticles
halfspace kernel         AABB + signed mesh
          |                    |
          v                    v
VALUES[scene,obj,sys]    VALUES[scene,obj,sys]
          |                    |
     +----+----+               v
     |         |             Covered
     v         v
  Contains   Filled
```

All consumers see one particle and rigid-pose snapshot per refresh. Simulator order
prevents macro-visual particles or state kernels from reading previous-frame parent
poses.

## 13. Review invariants

1. `initialize_view()` is ParticleViewAPI's only topology entry and completes all
   metadata.
2. Unified ViewAPIs build before dependent tensorized states.
3. Transient backend reads finish before capture/replay.
4. Rigid pose matrices update before visual-particle and state kernels.
5. Entry slices match owning-system particle order.
6. Captured arrays are grow-only; live count is data, not launch shape.
7. Metadata dirty does not imply graph dirty.
8. System activation is topology; in-registry count change is metadata/layout.
9. Macro membership refreshes dedicated PhysX views, not global handles.
10. `VALUES_CPU` is the tensorized-state value cache.

## 14. Validation and review index

Coverage includes cross-scene reader parity for all families, same-refresh parent and
rigid-particle motion, in-capacity macro add/remove without global handle rebuild or
graph dirtiness, family-scoped invalidation, micro-instancer lifecycle/serialization,
containment parity/scene isolation, contact count and exact-index parity, and legacy
`Contains`/`Filled`/`Covered` behavior.

The 12 tests in `OmniGibson/tests/test_particle_view_api.py` pass in the headless `behavior`
environment.

Core lifecycle:

- `OmniGibson/omnigibson/simulator.py`
- `OmniGibson/omnigibson/utils/particle_view_utils.py`
- `OmniGibson/omnigibson/object_states/tensorized_state.py`
- `OmniGibson/omnigibson/object_states/tensorized_object_system_state.py`

Consumers:

- `OmniGibson/omnigibson/object_states/contains.py`
- `OmniGibson/omnigibson/object_states/contact_particles.py`
- `OmniGibson/omnigibson/object_states/covered.py`
- `OmniGibson/omnigibson/object_states/filled.py`

Invalidation/topology:

- `OmniGibson/omnigibson/systems/system_base.py`
- `OmniGibson/omnigibson/systems/macro_particle_system.py`
- `OmniGibson/omnigibson/systems/micro_particle_system.py`
- `OmniGibson/omnigibson/scenes/scene_base.py`
- `OmniGibson/omnigibson/utils/usd_utils.py`
- `OmniGibson/omnigibson/utils/physx_utils.py`

Detailed-result consumers:

- `OmniGibson/omnigibson/transition_rules.py`
- `OmniGibson/omnigibson/object_states/particle_modifier.py`

Tests:

- `OmniGibson/tests/test_particle_view_api.py`
- `OmniGibson/tests/test_particle_states_vec.py`
