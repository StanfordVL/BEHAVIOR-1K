# Newton Migration

This document is the canonical engineering record for the OmniGibson migration
from Isaac Sim / PhysX to Newton. It describes the current implementation,
intentional compatibility decisions, known limitations, temporary workarounds,
validation coverage, and the planned path toward broader OmniGibson feature
support.

See [Newton Solver and Coupling Strategy](newton_solver_strategy.md) for the
multiphysics solver analysis, coupling options, vector-environment implications,
and validation plan.

See [Newton Renderer Architecture Handoff](newton_renderer_architecture.md) for
the replaceable camera-renderer interface and implementation acceptance criteria.

See [Newton GPU Pipeline Architecture](newton_gpu_pipeline_architecture.md) for
the vectorized execution model, reproducibility policy, and migration order.

Last updated: July 16, 2026.

## Status

The current implementation provides a Newton-native execution path for importing and
simulating BEHAVIOR scenes, objects, and robots from USD. The active solver is
MuJoCo through Newton's MuJoCo-Warp integration, and the active renderer is
Newton's OpenGL viewer.

The following examples have been used as migration acceptance tests without
example-specific backend logic:

- `omnigibson/examples/objects/visualize_object.py`
- `omnigibson/examples/scenes/scene_selector.py`
- `omnigibson/examples/robots/robot_control_example.py`

This is not yet feature parity with the original Isaac Sim implementation.
Scene import, rigid-body simulation, articulated objects, basic object APIs, and
robot control are operational. Object states, systems, sensors, tasks, dynamic
scene mutation, serialization, and several other higher-level features remain
unsupported.

## Migration Principles

The migration follows these rules:

1. Keep core BEHAVIOR/OmniGibson public APIs unchanged: signatures, return
   types (torch tensors), and configuration shapes stay stable while the
   implementation underneath changes. `omnigibson.prims` is the explicit
   exception. Deviations require a decision-log entry.
2. Keep high-level environment, scene, object, robot, and controller code
   backend-agnostic where practical.
3. Do not reproduce Isaac Sim's prim-centered runtime architecture in Newton.
   Runtime identity is represented by bodies, joints, shapes, entities, and
   state/control buffers.
4. Use USD as the import format for both dataset objects and robots.
5. Keep source dataset USDs immutable. Any Newton-specific repair is applied to
   a temporary prepared copy.
6. Prefer fixes at the shared OmniGibson API boundary over modifications to
   individual examples.
7. Pin Newton to an exact commit and validate upgrades before moving the pin.

## Dependency Baseline

The Newton dependency is pinned in `OmniGibson/setup.py`.

| Component | Current version |
| --- | --- |
| Newton | `1.5.0.dev0` |
| Newton commit | `8447545335aa264ef79bc7cd386bcdbb4f26eec3` |
| Warp | `1.16.0.dev20260716` (transitive, not pinned) |
| MuJoCo | `3.10.0` (transitive, not pinned) |
| MuJoCo-Warp | `3.10.0.2` (transitive, not pinned) |
| Python | `3.12` in the `newton-b1k` conda environment |

Only the Newton commit is enforced by the pin. Warp, MuJoCo, and MuJoCo-Warp
are transitive requirements of that commit and can drift between installs; the
versions above are what a fresh install resolved on July 16, 2026. Record the
resolved versions whenever the environment is rebuilt.

Newton development commits may change public APIs, private APIs, USD import
behavior, solver defaults, and transitive Warp / MuJoCo-Warp requirements. A
newer commit must not replace the pin without completing the upgrade procedure
and validation described below.

## Environment Setup

The Newton runtime uses its own conda environment, `newton-b1k`, separate from
the Isaac Sim `behavior` environment.

`pip install -e OmniGibson[newton]` does not currently resolve on Python 3.12:
OmniGibson's base requirements include `pymeshlab~=2022.2`, which has no
Python 3.12 wheels, and `bddl~=3.7.0`, which is provided by the monorepo's
`bddl3/` package rather than PyPI. Until the base requirements are updated,
install OmniGibson with `--no-deps` and install the Newton-path runtime
dependencies explicitly:

```bash
# 1. Create the environment.
conda create -y -n newton-b1k python=3.12

# 2. Install the pinned Newton build. The commit must match OmniGibson/setup.py.
#    The NVIDIA index hosts the Warp development builds Newton requires.
conda run -n newton-b1k pip install \
    "newton[examples] @ git+https://github.com/newton-physics/newton.git@8447545335aa264ef79bc7cd386bcdbb4f26eec3" \
    --extra-index-url https://pypi.nvidia.com/

# 3. Install the runtime dependencies the Newton code path imports.
conda run -n newton-b1k pip install \
    torch addict cryptography gymnasium h5py huggingface_hub imageio ipython \
    matplotlib packaging pillow pyyaml scipy termcolor tqdm

# 4. Install the monorepo packages as editable, skipping legacy dependencies.
cd /path/to/BEHAVIOR-1K
conda run -n newton-b1k pip install -e bddl3
conda run -n newton-b1k pip install -e OmniGibson --no-deps
```

This procedure was validated from scratch on July 16, 2026. Because OmniGibson
is installed as an editable package, `PYTHONPATH` does not need to be set.

## Architecture

### Simulator

`omnigibson.simulator` is now a package:

- `simulator/simulator.py` defines the legacy-compatible abstract simulator
  surface.
- `simulator/newton.py` implements the active Newton simulator.
- `simulator/physx.py` retains the previous PhysX simulator implementation as a
  reference. It is not selected by the current public launch path.

`omnigibson.Environment` creates the active simulator from the existing
OmniGibson configuration shape. `og.sim` points to that simulator after the
environment is built.

The Newton simulation loop owns:

- a Newton `Model`
- two Newton `State` buffers
- a Newton `Control` buffer
- a contact buffer
- a Newton solver
- an optional Newton viewer

The default simulation rate is 50 environment frames per second with eight
physics substeps per frame.

### Simulator API Support

Classification of the legacy `Simulator` surface (95 public methods and
properties in `simulator/physx.py`) against the Newton implementation,
audited July 16, 2026. Every legacy API resolves on the Newton simulator —
either to an implementation, a documented no-op, or an explicit
`UnsupportedSimulatorFeature`.

**Implemented on Newton.** Environment lifecycle
(`from_environment_configs`, `build_environment`, `apply_environment_action`,
`step`, `step_physics`, `render`, `close`); state (`dump_state`,
`load_state`, `serialize`, `deserialize`); timing (`get_sim_step_dt`,
`get_physics_dt`, `get_rendering_dt`, `current_time`,
`current_time_step_index`, `n_physics_timesteps_per_render`,
`initial_physics_dt`, `initial_rendering_dt`); viewer (`attach_viewer`,
`viewer_camera`, `enable_viewer_camera_teleoperation`, `viewer_visibility`,
`viewer_width`, `viewer_height`); registry access (`entities`, `objects`,
`robots`, `scenes`, `floor_plane`, `skybox`); and `gravity`, `device`,
`is_playing`, `is_stopped`, `is_paused`.

**No-ops by design.** `play` / `pause` / `stop` (Newton is stepped
explicitly; `is_playing` is always True), `sync_physx_to_fabric` and
`update_handles` (PhysX/Fabric concepts without a Newton equivalent),
`render_on_step` / `slowed` / `editing_usd` (pass-through context managers),
`camera_mover` (returns None).

**No-ops that must become real implementations.** The callback registries
(`add_/remove_callback_on_play/stop/add_obj/remove_obj`,
`*_callback_on_system_init/clear`, `get_callbacks_on_system_*`) accept
registrations but never fire. Pre-build spec mutation must dispatch the
object callbacks, and scene systems (Phase 5) must dispatch the system
callbacks; until then, code relying on callbacks silently misbehaves rather
than failing loudly. `adding_objects` / `removing_objects` context managers
scope pre-build spec mutation only; post-build mutation raises.

**Raises `UnsupportedSimulatorFeature`.** USD/prim access (`stage`,
`stage_id`, `world_prim`, `get_obj_at_prim_path`, `remove_prim`,
`import_scene`); PhysX interfaces (`pi`, `psi`, `psqi`, `physics_sim_view`,
`get_physics_context`); runtime mutation (`batch_add_objects`,
`batch_remove_objects`, `add_ground_plane`, `add_skybox` — ground planes and
lights are scene-spec driven at build time); configuration
(`set_simulation_dt`, `set_lighting_mode`); and scene persistence (`save`,
`restore`), pending scene-level JSON support.

### Runtime Entities

Newton does not use OmniGibson prim wrappers as its runtime abstraction. USD
paths are used during import and for diagnostic labels, but simulation access
is represented by:

- `SimEntity`
- `SimBody`
- `SimJoint`
- `SimShape`
- `EntityRegistry`

The Newton implementations expose body poses and velocities, joint positions
and velocities, object bounds, object pose updates, joint updates, and robot
control through Newton model/state/control arrays.

The Newton path must not import `omnigibson.prims`. Legacy prim files may remain
in the repository while other code is migrated, but they are not part of the
Newton runtime design.

### Scene Loading

The existing scene configuration remains the user-facing entry point.
`InteractiveTraversableScene` scene metadata is converted into object, robot,
and light declarations before the Newton model is finalized.

The runtime build flow is:

1. Resolve scene, object, and robot assets.
2. Prepare temporary DatasetObject USD copies.
3. Import robot physics from USD.
4. Import object collision geometry, rigid bodies, and joints from USD.
5. Merge object builders into one scene builder.
6. Import render-only visual meshes in a second pass.
7. Validate and repair mass/inertia arrays required by MuJoCo.
8. Finalize the Newton model.
9. Create runtime entities from imported label ranges.
10. Create state, control, contact, solver, and viewer objects.

The model is immutable after finalization. Dynamic object addition and removal
therefore require an explicit rebuild design and are not currently supported.

### Objects

The Newton path supports the existing constructor shape for:

- `DatasetObject`
- `USDObject`
- `PrimitiveObject`
- `LightObject`

Supported behavior includes:

- initial position and orientation
- scale
- `fixed_base`
- `visual_only`
- object registry lookup by common keys
- body/link/joint access
- pose and AABB queries
- pose updates
- joint position and velocity queries and updates
- `keep_still`

`LightObject` declarations currently affect the OpenGL viewer's global light
settings. They are not full USD or Omniverse light objects.

### Robots and Controllers

Robots are imported through USD. Robot defaults and controller-relevant metadata
are read from existing robot YAML files where available.

The Newton robot adapter currently recognizes these controller names:

- `JointController`
- `NullJointController`
- `DifferentialDriveController`
- `HolonomicBaseJointController`
- `MultiFingerGripperController`
- `InverseKinematicsController`
- `OperationalSpaceController`

This is an API compatibility layer, not a port of the original controller
implementations. The original controllers depend on Isaac articulation views.
Newton commands are applied directly to Newton / MuJoCo target buffers.

The current IK and OSC compatibility paths use a finite-difference positional
Jacobian and do not reproduce the full original controllers. Orientation
commands, null-space behavior, impedance behavior, command smoothing, and exact
gain semantics require further work.

The legacy fixed-base default is preserved: robots that are not eligible for a
floating non-holonomic locomotion base are fixed by default.

### Rendering

The current renderer is `newton.viewer.ViewerGL`.

Visuals are imported separately from collision geometry:

- collision meshes remain in the physics model but are hidden from rendering
- USD visual meshes are added as render-only shapes
- textures and material-bound mesh subsets are loaded
- textured meshes use a neutral shape color to avoid tinting their albedo
- hidden functional metalinks are excluded

Metalinks such as `fillable`, particle source/sink, particle
applier/remover, and slicer geometry are hidden. Toggle-button metalinks remain
visible.

Newton's development branch also provides an experimental RTX viewer, but
OmniGibson does not yet expose renderer selection or integrate `ViewerRTX`.
Sensors and Omniverse RTX rendering are separate future tasks.

## Physics Configuration

The default solver is Newton's `SolverMuJoCo` using MuJoCo-Warp contacts.
Configuration lives under `newton.simulation` and supports both flat fields and
the Isaac-Lab-style nested `solver_cfg` and `default_shape_cfg` forms.

Example:

```python
cfg = {
    "newton": {
        "simulation": {
            "num_substeps": 8,
            "solver_cfg": {
                "use_mujoco_contacts": True,
                "iterations": 150,
                "ls_iterations": 80,
                "ccd_iterations": 120,
                "cone": "elliptic",
                "impratio": 5.0,
            },
            "default_shape_cfg": {
                "ke": 100.0,
                "kd": 50.0,
                "kf": 100.0,
                "mu": 0.9,
            },
        }
    }
}
```

These defaults are intentionally conservative for dense BEHAVIOR scenes. They
improve stability but are expensive and do not eliminate all visible
penetration.

XPBD can be selected for diagnostics, but MuJoCo is the supported default for
the current migration.

`newton.environment.auto_build` (default `true`) controls whether
`og.Environment` builds the Newton model immediately on construction. Set it to
`false` to defer the build until the environment is first used.

## Known Workarounds and Technical Debt

Each workaround below must remain documented until it is either removed or
replaced by a tested upstream solution.

### W1: Single-threaded OpenUSD work

`PXR_WORK_THREAD_LIMIT=1` is set before Newton or PXR imports.

Reason: collider-dense BEHAVIOR USD imports have produced native failures in
OpenUSD parallel physics traversal.

Cost: slower USD import.

Removal condition: full `Rs_int`, `house_single_floor`, and a collider-dense
multi-object stress test must repeatedly load with the default thread count
without a native crash or excessive memory growth on the pinned Newton/OpenUSD
stack.

### W2: Robot-first physics import

Robot physics is imported before scene objects.

Reason: importing large robot articulations after many object resources had
already accumulated produced importer instability, especially for R1Pro.

Cost: import order is constrained.

Removal condition: robot-first and object-first imports must produce equivalent
models and pass repeated large-scene load tests.

### W3: Two-pass physics and visual import

All body, joint, and collision imports finish before visual-only meshes are
added.

Reason: importing collision and visual resources together across a full
BEHAVIOR scene previously caused native parser crashes.

Cost: custom visual import code must resolve materials, mesh subsets,
transforms, and body bindings separately from Newton's USD importer.

Removal condition: Newton's native `add_usd(..., load_visual_shapes=True)` path
must load full scenes repeatedly, preserve textures and visibility, and avoid
duplicate collision rendering.

### W4: Temporary prepared DatasetObject USDs

Encrypted assets are decrypted into temporary directories. Unencrypted assets
are copied before preprocessing. Source assets are never changed.

The prepared copy may receive:

- missing mass properties
- hidden metalink visibility/purpose

The preprocessing helper supports applying root scale, but the current scene
path scales Newton builder data after import so body, joint, shape, and inertia
arrays can be updated together.

Prepared directories are intentionally retained for the process lifetime.

Reason: Newton may keep native references to USD-backed mesh resources after
model finalization. Early cleanup previously caused native memory corruption in
large scenes.

Cost: temporary disk usage grows during long-running processes.

Removal condition: imported mesh and material ownership must be confirmed
independent of the USD stage and repeated build/close cycles must pass under
memory checking.

### W5: Approximate missing mass and inertia

Missing mass properties are authored on prepared USD copies using a coarse
bounding-box inertia approximation. Newton builder mass, inertia, and inverse
arrays are refreshed after runtime scaling.

Reason: some BEHAVIOR rigid bodies do not provide mass properties accepted by
Newton/MuJoCo. Native mass-property computation has failed for some assets, and
near-zero inertia causes severe instability.

Cost: physical parameters are approximate and may differ from PhysX.

Removal condition: assets receive validated authored mass properties or Newton
provides a robust importer path for all affected assets.

### W6: Manual scaling repair

Object scale is propagated through body positions, centers of mass, joint
frames, shape geometry, and mass/inertia bookkeeping before finalization.

Reason: scaling only visual or collision shape dimensions misaligns articulated
links and can leave stale inverse mass/inertia arrays.

Cost: this is complex and sensitive to Newton builder representation changes.

Removal condition: upstream USD scaling must preserve articulation placement,
collision geometry, mass properties, and runtime stability for BEHAVIOR assets.

### W7: Fixed-base anchor body and fixed joint

Fixed-base articulated objects receive a zero-mass anchor body fixed to world.
Imported movable root joints are reparented to that anchor.

Reason: a fixed base must remain fixed while drawers and doors remain dynamic.
Marking only the base body kinematic is not compatible with MuJoCo-Warp's child
joint mapping, while fixing every body makes the articulation non-interactive.

Cost: the runtime model contains a synthetic body and joint not authored in the
source USD.

Removal condition: Newton's USD import and MuJoCo conversion must directly
preserve the intended world-to-base fixed joint and child articulation tree for
all fixed-base BEHAVIOR objects.

### W8: Explicit self-collision filtering

Intra-object and intra-robot collision pairs are filtered explicitly.

Reason: fixed-base or merged shapes can be represented as world-body shapes,
allowing MuJoCo to detect contacts between an object's own frame, doors, and
drawers even when self-collision was disabled during import.

Cost: collision-filter pair counts and build time increase.

Removal condition: imported and merged builders must preserve equivalent
articulation self-collision semantics without explicit pair generation.

### W9: Mass/inertia floors for moving adapter bodies

Moving joint children with invalid zero mass or inertia receive small positive
floors before MuJoCo model compilation.

Reason: MuJoCo rejects moving bodies at or below `mjMINVAL`. Some robot USDs
contain massless adapter links that PhysX accepted.

Cost: these links gain artificial mass and inertia.

Removal condition: robot assets are corrected or the importer can safely
collapse/represent these adapter links without changing kinematics.

### W10: Added joint damping, armature, and friction

Unactuated object joints receive passive velocity damping. Scaled movable
joints receive small armature and friction floors.

Reason: dense resting contacts and zero-gain imported joints can inject energy
or drift after objects are awakened.

Cost: articulated objects may feel more damped than in the original simulator.

Removal condition: validated authored dynamics and stable contact behavior make
the added floors unnecessary.

### W11: Simplified controller compatibility

Controller names and command dimensions are preserved, but control is
implemented directly against Newton buffers. IK/OSC use a finite-difference
Jacobian computed on a scratch FK state (6 rows including orientation, damped
least squares); earlier revisions perturbed the live simulation state, which
corrupted velocities and model defaults. Gripper `JointController` commands
use position deltas because pure velocity targets barely moved Panda fingers.
Controller gain and mode changes after build push
`ModelFlags.JOINT_DOF_PROPERTIES` to the solver; actuator position/velocity
modes remain baked at build time.

Cost: controller response, stiffness, speed, and semantics differ from the
original implementations.

Removal condition: backend-agnostic controller math is separated from
Isaac-specific articulation access and validated against both expected
kinematics and physical response. `tests/test_newton_controllers.py` encodes
the acceptance criteria as quantitative tracking tests. As of July 22, 2026
the six Fetch tests pass (hold, joint tracking, closed-loop Cartesian IK to
1 cm, base drive, gripper limits), and per-family checks pass for fetch,
r1pro, locobot, ur5e, and stretch; tiago is `xfail` (W15).

Multi-family work added: joint groups split into `arm_left`/`arm_right` and
`gripper_left`/`gripper_right` for bilateral robots (detected from arm-joint
side prefixes, so single-arm grippers with left/right fingers are not
mis-split); gripper-mechanism tokens (`knuckle`, `finger`, `claw`) classify
into the gripper group; a `HolonomicBaseJointController` maps `(vx, vy, vyaw)`
onto virtual footprint joints; `get_position_orientation` reports `base_link`
rather than a non-moving virtual footprint link; and stiff position servos on
light gripper-linkage joints receive rotor armature (`5*ke*dt^2`) to stay
numerically stable. Cartesian deltas integrate on the previous target rather
than current positions, since re-anchoring to lagging joints let low-inertia
wrists ratchet away. Base drive speed calibration and orientation/OSC
tracking coverage remain.

### W13: Visual meshes added mass; robot inertia realigned to authored mass

Resolved July 22, 2026, in two parts. First, the render-only visual import
added every visual mesh at the ShapeConfig default density (1000), silently
adding each mesh's volume in kilograms to its body — Fetch imported at 403 kg
against 113 kg authored, saturating its torso actuator (`qfrc_actuator`
railed at the 450 N effort limit and oscillated instead of holding) and
pinning the base. This affected every object in every scene, not only
robots. Visual shapes now import with density 0.

Second, robot USDs author `MassAPI` mass and center of mass but not diagonal
inertia, so the importer keeps the authored mass while computing inertia from
collision geometry at default density; the builder's mass/inertia consistency
repair then raised the mass back to match the oversized inertia. Robot import
now rebuilds each body's inertia as a solid box at the authored mass (the
same approximation used for scaled scene objects) and applies the authored
center of mass.

Residual cost: robot inertia tensors are box approximations.

Removal condition: robot assets author full mass, inertia, and center of
mass, and the importer consumes them directly.

### W14: Chassis caster pads get low-friction priority contacts

Robot USDs bake caster pads into the base link's collision mesh instead of
modeling caster links; the pads rested on the ground at default friction and
static friction pinned mobile bases in place regardless of wheel commands.
Wheeled robots now import with near-zero friction on the wheel-parent chassis
shapes, and those geoms get elevated MuJoCo `geom_priority` so the low
friction wins the contact-parameter mix against the ground plane (MuJoCo
otherwise takes the larger of the two geom frictions).

Cost: chassis side contacts (e.g. bumping walls) also become low-friction,
and the priority elevation writes MuJoCo model arrays after solver
construction.

Removal condition: robot assets model casters as explicit rolling elements or
author per-shape friction materials that the importer consumes.

### W12: Private Newton API usage

The current implementation imports symbols from `newton._src`, including shape
flags, inertia validation, and USD helpers.

Reason: required functionality was not available through the public API when
the migration code was written.

Cost: private APIs can change without deprecation, including between pinned
development commits.

Removal condition: replace each private import with a public Newton API or
contribute the required public API upstream.

Resolved for controller targets: `newton/entities.py` now writes through a
`_control_target_array` helper that prefers the current `joint_target_q` /
`joint_target_qd` names and falls back to the deprecated
`joint_target_pos` / `joint_target_vel` aliases only if the new names are
absent.

### W15: Massless footprint virtual links diverge to NaN (tiago)

Holonomic-base robots mount `base_link` on a chain of virtual footprint
joints (x/y/z/rx/ry/rz) whose intermediate links carry no authored mass.
r1pro imports and simulates stably, but tiago diverges to NaN at import
step 0 in exactly these footprint bodies, so tiago is the one family whose
controller acceptance test is currently `xfail`.

Likely cause: the mass/inertia floor for moving adapter bodies (W9) does not
cover every tiago footprint link, leaving a near-zero-inertia body that
MuJoCo divides by. The difference from r1pro (same mechanism, stable) is not
yet isolated.

Removal condition: tiago builds and holds finite body states for 300 frames,
and `test_family_tiago` passes without `xfail`.

## Known Limitations

### Unsupported subsystems

The following are explicitly unsupported:

- object states
- transition rules
- particle and material systems
- sensors and observation modalities
- policy/data wrappers
- action primitives
- task and reward execution
- policy training integration

### Unsupported or partial core APIs

- `env_base.py`, `scene_base.py`, and the object classes currently contain
  narrower Newton implementations than the original OmniGibson classes. This is
  transitional migration debt, not the desired final architecture. Restore
  backend-agnostic behavior and method surfaces incrementally from the retained
  PhysX reference without reintroducing prim wrappers.
- Dynamic object insertion, removal, and registration after the model is
  built are unsupported by design (see the decision log). Planned support is
  pre-build spec mutation plus parked object pools for runtime variability.
- Scene systems and system registries are unsupported.
- Traversability sampling and shortest-path planning are unsupported.
- Scene JSON save/restore is unsupported. Simulator-level `dump_state`,
  `load_state`, `serialize`, and `deserialize` are implemented over the Newton
  state and control arrays; `Environment.reset()` restores the build-time
  state and accepts a seed. Restores are exact, but replayed trajectories
  match only to MuJoCo-Warp solver noise (GPU atomics), roughly `2e-5` over
  tens of frames.
- Moving or rotating the scene root is unsupported.
- Environment reset options are unsupported.
- Newton body/entity pose APIs currently support world frame only.
- `play`, `pause`, and `stop` are compatibility no-ops; Newton simulation is
  stepped explicitly.
- Several legacy simulator callbacks and Fabric/PhysX accessors are no-ops or
  raise `UnsupportedSimulatorFeature`.
- Environment observations are proprioception-only; reward and termination
  values remain placeholders (0.0 / False) until task execution is restored.
- Vectorized environments are not implemented as Newton worlds.
- Light support is approximate and viewer-specific.

### Physics and performance

- Dense scenes remain slow with the conservative MuJoCo settings.
- Some visible penetration remains under heavy contact.
- There is no validated sleeping strategy equivalent to the previous PhysX
  scene behavior.
- Controller gains and speeds need tuning per robot family.
- Object mass and inertia can be approximate.
- Self-collision is disabled by default to preserve prior OmniGibson behavior.
- Articulated object scaling and link alignment remain a high-risk regression
  area. The `Rs_int` cabinets and drawers must be visually checked after changes
  to USD import, joint frames, or scale propagation.
- Contact stability has been validated on `Rs_int`, not across every BEHAVIOR
  scene and asset.

### Rendering

- Only the OpenGL viewer is integrated.
- Rendering and physics use separate imported shape sets.
- USD material validation warnings may appear for assets that author material
  bindings without applying `MaterialBindingAPI`.
- RTX/path-traced rendering and camera sensors are not integrated.
- Collision/visual toggling is not exposed as an OmniGibson renderer option.

## Validation Record

The Newton development pin was validated on an NVIDIA RTX PRO 6000 Blackwell
workstation with CUDA 12.9 and driver CUDA compatibility 13.0.

Completed checks:

| Test | Result |
| --- | --- |
| Import `omnigibson` without Isaac Sim | Pass |
| Random `visualize_object` execution, 100 steps | Pass |
| Full `Rs_int` `scene_selector`, 100 steps | Pass |
| Fetch `robot_control_example`, 100 random-control steps | Pass |
| Full `Rs_int`, 300 normal frames (historical, Newton 1.2 baseline) | Pass |
| Full `Rs_int`, 300 disturbed frames on the current dev pin | Pass |
| Ruff and Ruff format for dependency pin | Pass |

The disturbance test applied a 20 N lateral and 5 N vertical force to
`straight_chair_amgwaw_0` and a 3 N m joint disturbance to
`bottom_cabinet_dajebq_0` for the first 30 frames. Body poses, body velocities,
joint positions, and joint velocities remained finite for 300 frames, and no
body exceeded the escape bound.

### Fresh environment reinstall, July 16, 2026

The `newton-b1k` environment was rebuilt from scratch on an NVIDIA GeForce RTX
5090 (driver 580.159.03) using the Environment Setup procedure above, and the
example suite was re-run headless (`OMNIGIBSON_HEADLESS=1`, `main()` called with
`random_selection=True, headless=True, short_exec=True`, 100 steps each):

| Test | Result |
| --- | --- |
| Import `omnigibson` without Isaac Sim | Pass |
| Random `visualize_object` (pool_stick/werrrt), 100 steps | Pass |
| Full `Rs_int` `scene_selector`, 100 steps | Pass |
| Full `restaurant_cafeteria` `scene_selector`, 100 steps | Pass |
| Fetch `robot_control_example` quickstart, 100 random-control steps | Pass |
| Full `Rs_int`, 600 undisturbed frames, positions/velocities tracked | Pass |

In the 600-frame run all body states stayed finite, no body left the scene
bounds or fell below the floor, and peak linear velocity was 0.51 m/s during a
settling transient near frame 300 that damped back out by frame 400.

The install resolved Warp `1.16.0.dev20260716` (one day newer than the previous
record), confirming that transitive versions drift under the Newton commit pin.

This validation is a migration smoke suite, not comprehensive regression
coverage. It should become automated before broad feature work resumes.

## Running the Current Examples

From the BEHAVIOR-1K worktree:

```bash
export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K/datasets
```

Visualize an object:

```bash
conda run --no-capture-output -n newton-b1k \
python OmniGibson/omnigibson/examples/objects/visualize_object.py
```

Load a scene:

```bash
conda run --no-capture-output -n newton-b1k \
python OmniGibson/omnigibson/examples/scenes/scene_selector.py
```

Run robot control:

```bash
conda run --no-capture-output -n newton-b1k \
python OmniGibson/omnigibson/examples/robots/robot_control_example.py --quickstart
```

Set `OMNIGIBSON_HEADLESS=1` for non-interactive validation. For headless smoke
tests, call each example's `main()` with `random_selection=True, headless=True,
short_exec=True` so no interactive prompt blocks execution; each run steps the
simulation 100 times. Note that `og.shutdown()` exits the process, so a clean
exit code 0 is the success signal.

Run the automated smoke suite (GPU required; about two minutes with a warm
Warp kernel cache, substantially longer on first run while kernels compile)
from `OmniGibson/`:

```bash
conda run --no-capture-output -n newton-b1k \
python -m pytest tests/test_newton_smoke.py -v
```

## Upgrading Newton

Use this process for every Newton upgrade:

1. Select a specific upstream commit. Do not depend on a moving branch.
2. Review Newton's changelog between the old and new commits, especially USD
   import, builder arrays, solver construction, contact semantics, and viewers.
3. Install the selected commit only into the `newton-b1k` conda environment:

   ```bash
   conda run --no-capture-output -n newton-b1k \
   python -m pip install --upgrade \
   "newton[examples] @ git+https://github.com/newton-physics/newton.git@<commit>" \
   --extra-index-url https://pypi.nvidia.com/
   ```

4. Update the exact commit in `OmniGibson/setup.py`.
5. Confirm the installed Newton package `direct_url.json` records that commit.
6. Audit all `newton._src` imports first because private APIs have no
   compatibility guarantee.
7. Run import and model-build smoke tests before changing OmniGibson code.
8. Run the three example tests and the 300-frame disturbance test.
9. Compare object placement, articulation alignment, visual materials,
   controller response, penetration, and frame rate.
10. Update this document with new behavior, removed workarounds, or new
    limitations before committing the new pin.

If validation fails, reinstall the previous pinned commit in the `newton-b1k`
environment. The dependency pin and git history provide the rollback reference.

## Roadmap

The order below prioritizes restoring OmniGibson behavior before optimization.

### Phase 1: Migration regression coverage

Implemented in `OmniGibson/tests/test_newton_smoke.py` (each scenario runs in
its own subprocess because `og.shutdown()` exits the process and the model is
immutable after finalization):

- Automated smoke tests for import, scene load, robot load, entity registry
  lookups, stepping, and controller command validation. Done.
- A repeatable 300-frame stability/disturbance test on full `Rs_int` (100
  settling frames, then a chair drop and a cabinet joint set to mid-range,
  then 200 monitored frames with finite/bounds assertions). Done.
- Tests fail if `isaacsim`, `omni`, `carb`, or `omnigibson.prims` appears in
  `sys.modules` on the Newton path. Done.
- Repeated build/close cycles in one process for native memory safety. Done
  (two cycles; extend before using it as the W4 removal test).

Remaining in this phase:

- Visual coverage is count-level only (`shape_count`); add assertions on
  visual shape structures and textures.
- CI: deferred while this is a development branch. The suite runs locally in
  the `newton-b1k` environment; a CI job needs a Python 3.12 environment built
  from the setup.py pin on a dataset-enabled GPU runner.

### Phase 2: Core API restoration

- Restore environment observation/action spaces and reset semantics. Done:
  `Environment.reset()` restores the build-time snapshot and seeds RNGs
  (`state_reset_determinism` scenario); action/observation spaces mirror the
  legacy shapes — `gym.spaces.Dict` keyed by robot name with per-robot Box
  action spaces, `flatten_action_space` / `flatten_obs_space` env-config flags,
  dict- or flat-tensor actions, and torch tensors at the API boundary
  (`gym_api_conformance` scenario). The only observation modality is
  `proprio` (joint positions/velocities, base pose and velocities);
  unsupported modalities such as `rgb` are skipped with a warning until
  sensors arrive in Phase 6.
- Implement simulator and scene state dump/load/serialize/deserialize. The
  simulator level is done (state/control array snapshots; solver persistent
  buffers cleared on load; controller target caches invalidated). Scene-level
  JSON save/restore remains.
- Dynamic object insertion/removal: resolved by decision rather than code.
  There is no runtime model mutation; see the decision log entry on pre-build
  spec mutation and parked object pools. Pre-build `add_object` support and
  the parked-pool mechanism land with their consumers (scene semantics and
  task sampling in Phase 4, transition rules in Phase 5).
- Consolidate overlapping simulator base interfaces. Done: the legacy
  `Simulator` surface is fully classified in the Simulator API Support
  section; `gravity`, `current_time`, `current_time_step_index`, and
  `n_physics_timesteps_per_render` were added to close the audit gaps.
- Replace compatibility no-ops with implementations or explicit unsupported
  errors. Documented per API in Simulator API Support; the callback
  registries remain silent no-ops until the dynamic-mutation and systems
  designs land.

### Phase 3: Robot control

- Separate controller math from Isaac articulation access.
- Replace finite-difference IK/OSC compatibility paths with shared,
  backend-agnostic implementations.
- Validate all registered robot families and fixed-base defaults.
- Tune gains, stiffness, speed, limits, and gripper behavior.
- Add mobile and holonomic base coverage.

### Phase 4: Scene semantics

- Restore traversability maps, floor sampling, and path planning.
- Restore object registry keys used by tasks and BDDL.
- Implement scene systems and lifecycle management.
- Validate all BEHAVIOR scenes for import, placement, contacts, and
  articulation alignment.

### Phase 5: Object states and tasks

- Rebuild object states on entity, shape, contact, and sensor APIs rather than
  prim wrappers. When restoring object-state deserialization, reapply the
  `_recorded_non_kin_state_names` filter from PR #2260 (main commit
  `77a05f3a5`); that fix landed in the legacy `usd_object.py` after the Newton
  rewrite and was dropped in the v3.9.0 rebase.
- Restore transition rules and BDDL predicates.
- Restore task initialization, reward, termination, and sampling.
- Restore particles and material systems after rigid-body semantics are stable.

### Phase 6: Rendering and sensors

- Add renderer selection with OpenGL as the lightweight default.
- Evaluate and integrate Newton `ViewerRTX`.
- Implement camera, depth, segmentation, scan, and contact sensor interfaces.
- Define consistent visual/collision/debug visibility controls across renderers.

### Phase 7: Performance and scale

- Profile USD preparation, model building, contact generation, solver time, and
  rendering separately.
- Revisit substeps and MuJoCo solver iteration defaults after broad regression
  coverage exists.
- Investigate sleeping or active-set reduction.
- Add multi-world/vectorized execution without changing single-environment APIs.

## Decision Log

### Newton-native runtime instead of prim wrappers

Newton's model and state arrays are the runtime source of truth. Recreating
Isaac prim wrappers would preserve backend-specific architecture instead of
preserving useful OmniGibson behavior.

### USD-only asset import

USD remains the common authored asset format for scenes, objects, and robots.
This preserves the existing asset pipeline and avoids maintaining separate robot
descriptions for Newton.

### MuJoCo as the default solver

MuJoCo-Warp is Newton's primary robotics backend and is the current target for
articulations and robot control. Newton-generated contacts remain available as
a diagnostic fallback but produced unacceptable penetration in tested scenes.

### Preserve high-level APIs, not Isaac lifecycle semantics

Environment, scene, object, robot, and controller entry points should remain
recognizable. Isaac-specific concepts such as stage mutation requirements,
Fabric synchronization, prim handles, and application play/stop state should
not be reproduced unless they provide required user-facing behavior.

### Torch-facing APIs, unlike Isaac Lab's Warp-array switch

Isaac Lab 3.0's Newton migration (the closest comparable effort) kept its
asset-class APIs stable through abstract base classes and a backend factory,
but deliberately broke one contract: data properties now return Warp arrays
instead of PyTorch tensors, trading API compatibility for CUDA-graph
performance. BEHAVIOR keeps torch tensors at the API boundary per migration
principle 1; Warp-native fast paths may be added internally during Phase 7
without changing public signatures. Other Isaac Lab patterns worth tracking:
pluggable visualizers (Newton OpenGL / Rerun / Viser) for Phase 6, the
`NewtonSceneAPI`/`MjcSceneAPI` USD schemas for authoring solver parameters and
mass fixes in USD instead of runtime repairs (a removal path for W4/W5), and
their published Newton asset constraints (no reversed joints, no closed
chains, no near-zero mass/inertia, strict USD composition), which match our
W5/W7/W9 experience and motivate a pre-import asset audit tool.

### No runtime scene mutation: pre-build spec mutation plus parked pools

The Newton/MuJoCo model is immutable after finalization, so objects cannot be
imported or removed mid-session the way the PhysX runtime mutated the stage.
Isaac Lab has the same posture even on PhysX: its InteractiveScene fixes the
asset set before simulation starts, the team recommends against replacing
objects during simulation, and the sanctioned pattern for variable object
sets is a pre-spawned pool whose unused members are parked far outside the
workspace. BEHAVIOR adopts the same two-part design instead of a
rebuild-based emulation:

1. Pre-build spec mutation. Scene object specs are declarative until the
   model is finalized, so `add_object`/removal before build is cheap spec
   editing. This maps the legacy stopped-add-play lifecycle (used by the
   legacy test fixtures) onto spec-then-build, with object add/remove
   callbacks dispatched at that stage.
2. Parked object pools for runtime variability. Everything an activity might
   need — transition-rule outputs (statically enumerable from BDDL rule
   definitions), task-sampling candidates — is added to the model at build
   time and parked outside the workspace: collision-filtered, kept still,
   candidates for MuJoCo-Warp sleeping. Spawning is teleport-and-activate;
   despawning is teleport-and-park. Rigid objects only; particle-based
   outputs are part of the systems design in Phase 5.

Post-build `add_object`/removal raises explicitly. Scene composition changes
require recreating the environment. Large always-colliding pools are a known
performance cost; collision filtering and sleeping bound it.

### Keep PhysX as a reference implementation for now

The previous simulator implementation remains in `simulator/physx.py` to aid API
comparison and incremental restoration. The active runtime is Newton, and new
features should not introduce Newton/PhysX branching throughout high-level
code.

## Maintenance Rules

- Update this document whenever a workaround is added, removed, or materially
  changed.
- Every workaround comment should explain the observed failure, the chosen
  mitigation, and its removal condition here.
- Do not modify source dataset USDs during import.
- Do not add example-specific backend patches.
- Do not add new Newton dependencies on `omnigibson.prims`.
- Do not upgrade a floating Newton branch; pin a full commit SHA.
- Prefer public Newton APIs. Track every unavoidable `newton._src` dependency.
- Add validation proportional to any change in import, scaling, mass/inertia,
  fixed-base handling, contacts, or controller behavior.
