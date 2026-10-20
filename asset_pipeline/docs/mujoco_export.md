# MuJoCo export

The MuJoCo export turns the BEHAVIOR-1K dataset into plain MJCF that MuJoCo
loads directly. MJCF can't express everything BEHAVIOR needs, so the rest goes
into `b1k_metadata` custom texts and JSON sidecars.

Scenes scale objects with **frame scale**:

```xml
<frame scale="sx sy sz" scaling="geometry" massscale="1">
```

This needs a MuJoCo with anisotropic frame scaling.

| Stage | Script | Output |
| --- | --- | --- |
| Objects | `export_objects_mujoco.py` | `objects_mujoco.zip` |
| Base scenes | `export_scenes_mujoco.py` | `scenes_mujoco.zip` |
| Joint order (OmniGibson) | `dump_joint_order_og.py` (or `dump_joint_order_physx.py`) | `metadata/mujoco_joint_order.json` |
| Validation | `validate_mujoco_export.py` | report |

## Naming

| Thing | Name |
| --- | --- |
| Object instance N | OmniGibson's name, e.g. `bottom_cabinet_qacthv_0` (the scene URDF's `bottom_cabinet-qacthv-0` with `-` replaced by `_`) |
| Root body | `N_root` in a part or for task-added objects; `{part}__N_root` when part `{part}` is attached into a composed scene. Static if the object is fixed; loose objects have `<freejoint name="…_freejoint"/>` |
| Object contents | `N_base_link`, `N_link_1`, `N_j_link_1`, … in a part (`{part}__N_…` in a composed scene) |
| Object metadata | custom text `b1k_metadata` in an object archive; `N_b1k_metadata` in a part (`{part}__N_b1k_metadata` in a composed scene) |
| Catalog model asset | `{category}-{model}` |
| Inlined structure assets | `{part}__{category}-{model}-{original}` |
| Cameras | `camera_{name}` in a part (`{part}__camera_{name}` in a composed scene), with `rooms::camera_{name}` texts for room cameras |
| Rooms of an object | custom text `rooms::N` in a part (`{part}__rooms::N` in a composed scene) |

Structure objects (walls, floors, ceilings, lawn, driveway, fence, roof and
background) and visual-only objects are inlined into scenes rather than
attached. They still get exactly the names that attaching them would give them.

## Object archives

An object archive lives at `objects/{category}/{model}/mjz/{model}.mjz` and
contains:

- `model.xml`: a single `base_link` body tree, with no free joint and every joint named;
- `visual/`, `collision/` and `volumes/`: OBJ meshes;
- `textures/`: PBR texture layers;
- `visual/*.glb`: per-link GLBs.

Geom groups:

| Group | Contents |
| --- | --- |
| 2 | Visual (`contype=conaffinity=0`, massless) |
| 3 | Collision (with `<compiler inertiagrouprange="3 3"/>`) |
| 4 | Meta volumes and meta sites |

Masses are the category mass, split over the links by collision volume, as
OmniGibson's `FORCE_CATEGORY_MASS` does.

Contacts follow OmniGibson's collision filtering:

- **Self-collisions are off.** `<contact><exclude>` covers every pair of links
  that have collision geoms.
- **Structure collision geoms** use `contype=2 conaffinity=1`.
- **The moving parts of sliding and garage doors** use `contype=4 conaffinity=1`,
  so they can slide into walls. Everything else uses `1/1`.

**Anisotropic scaling.** Uniform frame scales work for every object. An
anisotropic one needs both of these:

- hinge and slide joints, and box, ellipsoid and plane geoms, are aligned with
  the base link's axes;
- no other primitive geoms are used.

The object exporter records whether an object meets them in
`b1k_metadata.anisotropic_scaling` and lists the ones that don't as warnings.
The scene exporter asserts it for every object it scales anisotropically.

**Joint fixes.** `find_mujoco_joint_fixes.py` finds every object that a scene or
task instance scales anisotropically and that has unaligned joints. It writes
the list to `metadata/mujoco_joint_fixes.json`, which the object exporter
applies:

| Off-axis angle | Fix | `action` |
| --- | --- | --- |
| Within 10° | The joint's axis is rotated onto the nearest base-link axis. | `snap` |
| More than 10° | The joint is made fixed at its zero position. These are cosmetic parts mounted at an angle, such as toilet flush levers. | `fix` |

`b1k_metadata.joint_fixes` records what the exporter did to each object.

### `b1k_metadata`

The archive embeds a compact JSON object as `<custom><text name="b1k_metadata" data="…"/></custom>`.

> [!NOTE]
> Lengths in `b1k_metadata` are **unscaled**. In a scene, multiply them by the
> instance's scale, or read the compiled model, which MuJoCo has already scaled.

| Key | Content |
| --- | --- |
| `format` | `"b1k_metadata/1"` |
| `dataset_version` | the asset pipeline's `VERSION` |
| `category`, `model` | |
| `synset`, `abilities` | from bddl's object taxonomy (abilities with their parameters) |
| `structure` | whether the category is a structure category |
| `bbox_size`, `base_link_offset` | from `misc/metadata.json` |
| `mass` | the total (category) mass |
| `joints` | movable joint names (unprefixed) |
| `anisotropic_scaling` | `{ok, problems}`: whether any anisotropic frame scale compiles |
| `openable` | the raw `openable_joint_ids` and `openable_both_sides`, plus `joints: [{joint, direction}]` and `both_sides` as OmniGibson's Open state sees them: the annotated joints with direction 1, or every joint if none is annotated |
| `meta_links` | `{type: {id: [{link, body, element, name, shape, pos, quat, size, mesh, dir}]}}`: every meta element written (fillable/openfillable volume geoms and their box sites, toggle buttons, heat sources, attachment points, lights, …), in the frame of `body` |
| `link_tags`, `object_parts` | raw |
| `groups` | `{collision: 3, visual: 2, meta: 4}` |

## Base scenes

Each scene is split into **parts**, following OmniGibson's loading rule
(`InteractiveTraversableScene._should_load_object`):

- `shell`: walls, ceilings, fences, roofs, background, doors and sliding doors,
  which load whatever the rooms are. It also holds the global cameras.
- One part per distinct room set of the other objects. Most parts are a single
  room, such as `kitchen_0`, but objects that span rooms get their own part,
  such as `kitchen_0-living_room_0`. Each part holds its rooms' cameras.
- `no_room`: the other objects without rooms. OmniGibson only loads these when
  it loads every room, so only full scenes attach this part.

A part is a scene MJCF of its own, at `scenes/{scene}/mjcf/parts/{part}/model.xml`:

- catalog objects are attached from their archives, by relative path;
- structure is inlined, with its files under `structure/{category}_{model}/`
  and its textures capped at 1024 px;
- it has no skybox.

Parts compose with attach prefix `{part}__`, so each part's elements are
namespaced by its part name in any composed scene:

```xml
<mujoco model="house_single_floor_best">
  <compiler angle="radian" autolimits="true" inertiagrouprange="3 3"/>
  <asset>
    <texture name="skybox" type="skybox" file="../../../skybox.png" gridsize="3 4" gridlayout=".U..LFRB.D.."/>
    <model name="part-shell" file="parts/shell/model.xml"/>
    <model name="part-kitchen_0" file="parts/kitchen_0/model.xml"/>
  </asset>
  <worldbody>
    <attach model="part-shell" prefix="shell__"/>
    <attach model="part-kitchen_0" prefix="kitchen_0__"/>
  </worldbody>
</mujoco>
```

`{scene}_best.xml` attaches every part; the `_with_clutter` variant isn't
exported. Poses, scales and fixedness come from the scene's settled
`{scene}_stable.json` in the task-instance archive when it has one, and from
the scene's `_best` URDF otherwise. Cameras come from the URDF.

`{scene}_parts.json` describes the scene:

```json
{
  "format": "b1k_mujoco_scene/1",
  "scene": "house_single_floor",
  "pose_source": "stable.json",
  "scenes": {"best": "house_single_floor_best.xml"},
  "parts": {"kitchen_0": {"file": "parts/kitchen_0/model.xml", "rooms": ["kitchen_0"],
                          "objects": ["…"], "inlined": ["floors_…_0"], "cameras": ["camera_…"]}},
  "objects": {"bottom_cabinet_qacthv_0": {"category": "bottom_cabinet", "model": "qacthv", "scale": [1, 1, 1],
              "fixed": true, "visual_only": false, "rooms": ["kitchen_0"], "part": "kitchen_0",
              "inlined": false, "archive": "objects/bottom_cabinet/qacthv/mjz/qacthv.mjz"}},
  "issues": []
}
```

To load a scene restricted to some rooms, attach `shell` plus every part whose
`rooms` intersect them.

## Task scenes

`export_tasks_mujoco.py` exports each activity of the 2026 challenge task
instances, which have a template, 300 train instances and 20 public test
instances. It writes these files next to the scene's parts:

- `{scene}_task_{activity}_partial.xml`: the rooms the official evaluator
  loads. These are the `inroom` room types of the BDDL's base scope, plus one
  level of `ROOM_DEPENDENCIES`, plus `TASK_SPECIFIC_EXTRA_ROOMS`. The scene
  attaches `shell` and every part whose rooms intersect them.
- `{scene}_task_{activity}_full.xml`: every part.
- `{scene}_task_{activity}.json`: the sidecar.

Each task scene contains, in this order:

1. its parts, attached with prefix `{part}__`;
2. the objects that the template adds to the settled scene, as `N_root`;
3. the robot frames `robot_template`, `robot_train_0007`, `robot_test_0301`, …
   These are yaw-only, worldbody `<frame>`s placed after every body. A few are
   shifted so that Huba fits (see [Robot frames](#robot-frames)).
4. keyframes named `template`, `train_0007`, `test_0301`, … Each holds the
   scene's full `qpos`.

`qpos0` is the settled scene, with task objects at their template poses. The
XML holds no task semantics.

To start an instance in plain MuJoCo:

```python
model = mujoco.MjModel.from_xml_path("house_double_floor_lower_task_turning_on_radio_partial.xml")
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("test_0301").id)
# Attach or place the robot at the frame robot_test_0301 (in an MjSpec, before compiling).
```

The sidecar (`format: b1k_mujoco_task/1`) holds:

| Key | Content |
| --- | --- |
| `activity`, `activity_definition`, `task_index`, `scene` | |
| `instruction`, `human_length_steps`, `eval_timeout_seconds` | from the challenge's `tasks.jsonl` and `metadata/task.jsonl`; the timeout is 1.5 × length / 30 Hz |
| `modes` | `{partial, full}`: `xml`, `rooms`, `parts`, `added_objects`, `nq` and `qpos_layout` (`[joint, type, qposadr, size]`) |
| `default_mode` | `"partial"` |
| `rooms` | the evaluator's room `types` and their `instances` in the scene |
| `bddl` | `problem_sha256`, `wildcards` (room instances, scene layout and the expanded instances' objects, as OmniGibson's BehaviorTask compiles them), and the compiled `init` and `goal` |
| `scope` | scope instance → `{kind: object, object, synset}`, `{kind: agent}`, `{kind: system, system}` or `{kind: future}` |
| `objects` | for scope and task-added objects: category, model, scale, fixedness, rooms, part, archive, `prefix`, `root`, `freejoint` and `joints` |
| `initial_nonkin`, `nonkin_varies` | the template's non-kinematic states of scope objects, and which states some instance changes |
| `instances` | per instance: `id`, `split`, `keyframe`, `robot_frame`, `source` (the instance file), `robot_pose` (BEHAVIOR's pose: key, position, orientation, tilt), `robot_frame_pose` (the XML frame's `position`, `quat_wxyz` and its `shift`, or `null`), `nonkin` (differences only), `systems` (raw particle data) and `issues` |
| `issues`, `warnings` | for example, that systems aren't simulated, or that the joint order of a model was assumed |

### Robot frames

Each robot frame starts from BEHAVIOR's R1Pro base pose for that instance,
reduced to its yaw. The evaluated robot, though, is Huba (Automaton), whose
footprint and home pose differ from R1Pro's: at some initial states, Huba
standing at R1Pro's pose would penetrate furniture or the objects that the
instance spawned next to the robot.

`metadata/mujoco_robot_frame_offsets.json` lists, for those initial states
only, the smallest shift that makes Huba fit (1,500 of the 21,120 initial
states of the 66 activities that google3's mjdb tasks use; median 8 cm):

```json
{"activities": {"turning_on_radio": {"scene": "house_double_floor_lower",
  "frames": {"train_0081": [0.08, 0.0, 0.0], "train_0100": [0.02, -0.02, 0.0]}}}}
```

A shift `[dx, dy, dyaw_deg]` is in the frame's own coordinates: `dx` forward,
`dy` left, and a turn about the frame's z. The exporter applies it when it
writes the frame, and records the result in the sidecar's
`robot_frame_pose` (`robot_pose` stays BEHAVIOR's pose); the validator checks
that the two agree. Every other frame is BEHAVIOR's pose, unshifted.

google3's `check_robot_fit --search` (in
`prototypes/gdm/robotics/simulation/external_benchmark_conversions/behavior_1k`)
writes the file. It stands Huba in its home pose at each initial state's frame,
in the partial-rooms task scene at that keyframe, and collides only the pairs
that MuJoCo would. An initial state needs a shift if Huba penetrates a
non-ground geom by more than 1 mm, or the ground (floors, lawn, driveway,
carpet) by more than 5 mm. A shifted pose fits if Huba keeps 2 cm from every
non-ground geom, its base's footprint stands on the ground of the loaded rooms,
and the straight path from the original frame doesn't cross a wall. The search
tries, in order, and keeps the first pose that fits:

1. translations on a 2 cm grid out to 0.5 m, nearest first, keeping the yaw;
   then the same with the frame turned by 5°, 10° and 15° either way;
2. the same with only 1 cm, then 5 mm, of clearance;
3. as a last resort, steps 1 and 2 again with translations out to 0.8 m and
   turns of up to 90°.

The search's report lists the clearance of every shift and which needed the
last resort. Shifts are never applied to frames that already fit.

**Instance joint values.** Instances store `joint_pos` in OmniGibson's joint
order. `metadata/mujoco_joint_order.json`, written by `dump_joint_order_og.py`,
gives that order. Without it, the URDF's order is assumed, and the sidecar
says so. Joints that `mujoco_joint_fixes.json` made fixed are skipped.

Where Isaac Sim's renderer can't start (Isaac Sim 5.1 crashes in
`librtx.scenedb` on R590+ drivers), `dump_joint_order_physx.py` writes the same
file from PhysX alone: it reads the joint order from `omni.physics.tensors`, as
OmniGibson does, in a Kit app without rendering. For the 39 models that the
2026 instances articulate, the order matches the URDF's.

## Validation

`validate_mujoco_export.py` needs only MuJoCo and NumPy. For each scene it:

- compiles every part and every composed scene;
- checks root bodies, free joints and metadata texts;
- rejects `scale` attributes on `<attach>`;
- reports contacts at `qpos0` and drift over 100 steps of 2 ms;
- reports model sizes.

With `--check_objects`, it also compiles each referenced object archive under
an anisotropic frame scale.
