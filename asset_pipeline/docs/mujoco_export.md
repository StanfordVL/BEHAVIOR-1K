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
| Joint order (OmniGibson) | `dump_joint_order_og.py` | `metadata/mujoco_joint_order.json` |
| Validation | `validate_mujoco_export.py` | report |

## Naming

| Thing | Name |
| --- | --- |
| Object instance N | OmniGibson's name, e.g. `bottom_cabinet_qacthv_0` (the scene URDF's `bottom_cabinet-qacthv-0` with `-` replaced by `_`) |
| Root body | `N_root`, at the object's base link pose. Static if the object is fixed; loose objects have `<freejoint name="N_freejoint"/>` |
| Object contents | `N_base_link`, `N_link_1`, `N_j_link_1`, … (attach prefix `N_`) |
| Object metadata | custom text `b1k_metadata` in an object archive; `N_b1k_metadata` in scenes |
| Catalog model asset | `{category}-{model}` |
| Inlined structure assets | `{part}__{category}-{model}-{original}` |
| Cameras | `camera_{name}`, with `rooms::camera_{name}` texts for room cameras |
| Rooms of an object | custom text `rooms::N` (comma-separated) |

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
| 0 | Collision |
| 2 | Visual (`contype=conaffinity=0`, massless) |
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
| `groups` | `{collision: 0, visual: 2, meta: 4}` |

## Base scenes

Each scene is split into **parts**, following OmniGibson's loading rule
(`InteractiveTraversableScene._should_load_object`):

- `shell`: walls, ceilings, fences, roofs, background, doors and sliding doors,
  which load whatever the rooms are. It also holds the global cameras.
- One part per distinct room set of the other objects. Most parts are a single
  room, such as `kitchen_0`, but objects that span rooms get their own part,
  such as `kitchen_0-living_room_0`. Each part holds its rooms' cameras.
- `{part}.clutter`: the objects that `_with_clutter` adds.

A part is a scene MJCF of its own, at `scenes/{scene}/mjcf/parts/{part}/model.xml`:

- catalog objects are attached from their archives, by relative path;
- structure is inlined, with its files under `structure/{category}_{model}/`
  and its textures capped at 1024 px;
- it has no skybox.

Parts compose with an empty attach prefix, so objects keep their names in any
composed scene:

```xml
<mujoco model="house_single_floor_best">
  <compiler angle="radian" autolimits="true"/>
  <asset>
    <texture name="skybox" type="skybox" file="../../../skybox.png" gridsize="3 4" gridlayout=".U..LFRB.D.."/>
    <model name="part-shell" file="parts/shell/model.xml"/>
    <model name="part-kitchen_0" file="parts/kitchen_0/model.xml"/>
  </asset>
  <worldbody>
    <attach model="part-shell" prefix=""/>
    <attach model="part-kitchen_0" prefix=""/>
  </worldbody>
</mujoco>
```

`{scene}_best.xml` attaches every part, and `{scene}_with_clutter.xml` adds the
clutter parts on top. Poses, scales and fixedness come from the scene's settled
`{scene}_stable.json` in the task-instance archive when it has one, and from
the scene URDF otherwise.

`{scene}_parts.json` describes the scene:

```json
{
  "format": "b1k_mujoco_scene/1",
  "scene": "house_single_floor",
  "pose_source": "stable.json",
  "scenes": {"best": "house_single_floor_best.xml", "with_clutter": "house_single_floor_with_clutter.xml"},
  "parts": {"kitchen_0": {"file": "parts/kitchen_0/model.xml", "rooms": ["kitchen_0"], "clutter": false,
                          "objects": ["…"], "inlined": ["floors_…_0"], "cameras": ["camera_…"]}},
  "objects": {"bottom_cabinet_qacthv_0": {"category": "bottom_cabinet", "model": "qacthv", "scale": [1, 1, 1],
              "fixed": true, "visual_only": false, "rooms": ["kitchen_0"], "part": "kitchen_0",
              "inlined": false, "archive": "objects/bottom_cabinet/qacthv/mjz/qacthv.mjz", "clutter": false}},
  "issues": []
}
```

To load a scene restricted to some rooms, attach `shell` plus every part whose
`rooms` intersect them.

## Validation

`validate_mujoco_export.py` needs only MuJoCo and NumPy. For each scene it:

- compiles every part and every composed scene;
- checks root bodies, free joints and metadata texts;
- rejects `scale` attributes on `<attach>`;
- reports contacts at `qpos0` and drift over 100 steps of 2 ms;
- reports model sizes.

With `--check_objects`, it also compiles each referenced object archive under
an anisotropic frame scale.
