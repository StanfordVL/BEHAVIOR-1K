"""Conventions shared by the MuJoCo exporters.

The MuJoCo export is plain MJCF: object archives (export_objects_mujoco.py),
base-scene parts and composed scenes (export_scenes_mujoco.py), and task scenes
(export_tasks_mujoco.py). BEHAVIOR-specific data that MJCF can't express lives
in `b1k_metadata` custom texts and in JSON sidecars. See docs/mujoco_export.md.
"""

import json
import pathlib

import numpy as np

import b1k_pipeline.utils

DATASET_VERSION = (b1k_pipeline.utils.PIPELINE_ROOT / "VERSION").read_text().strip()

# Geom and site groups. Visual meshes are in group 2, collision geoms in
# group 3 (with compiler inertiagrouprange="3 3" so inertia comes from the
# collision meshes while keeping them hidden by default in the MuJoCo viewer),
# and meta volumes and sites in group 4.
COLLISION_GROUP = 3
VISUAL_GROUP = 2
META_GROUP = 4
GROUPS = {"collision": COLLISION_GROUP, "visual": VISUAL_GROUP, "meta": META_GROUP}

# OmniGibson's omnigibson.utils.constants. Copied rather than imported because
# the pipeline doesn't depend on OmniGibson.
STRUCTURE_CATEGORIES = frozenset(
    {"floors", "walls", "ceilings", "lawn", "driveway", "fence", "roof", "background"}
)
GROUND_CATEGORIES = frozenset({"floors", "lawn", "driveway", "carpet"})

STRUCTURAL_DOOR_CATEGORIES = frozenset({"sliding_door", "garage_door"})

# Contact bitmasks. MuJoCo collides two geoms if (contype1 & conaffinity2) or
# (contype2 & conaffinity1). Everything uses 1/1 except:
# * structure collision geoms, 2/1: they collide with everything but each other
#   (they're static anyway) and the moving parts of structural doors;
# * moving parts of structural doors, 4/1: they slide into walls, as OmniGibson's
#   "structural_doors" collision group allows.
STRUCTURE_CONTACT = {"contype": "2", "conaffinity": "1"}
STRUCTURAL_DOOR_CONTACT = {"contype": "4", "conaffinity": "1"}

# Objects that InteractiveTraversableScene._should_load_object always loads,
# whatever rooms are requested: structure that isn't ground, and doors.
SHELL_CATEGORIES = (STRUCTURE_CATEGORIES - GROUND_CATEGORIES) | {"door", "sliding_door"}

# Root file of every archive (MuJoCo's own MJZ convention).
ARCHIVE_ROOT_FILE = "model.xml"

# Name of the custom text that carries an object's BEHAVIOR metadata. MuJoCo's
# attach prefixes custom texts like everything else, so an object attached with
# prefix "N_" exposes it as "N_b1k_metadata".
METADATA_TEXT = "b1k_metadata"
METADATA_FORMAT = "b1k_metadata/1"

# Frame-scale attributes. BEHAVIOR keeps an object's mass (its category mass)
# whatever its scale, and scales objects anisotropically, so only geometry and
# inertia follow the scale.
SCALE_ATTRS = {"scaling": "geometry", "massscale": "1"}

# Scales within this tolerance of 1 on every axis are written without a frame.
# Settled scales are float32 bounding-box ratios, so "unit" scales carry noise
# around 1e-6; 1e-4 is a tenth of a millimeter on a one-meter object.
UNIT_SCALE_TOLERANCE = 1e-4

# MuJoCo checks that shapes and joints are aligned with an anisotropic scale's
# axes to within about 2 degrees; stay a little inside that.
ALIGNMENT_TOLERANCE_DEG = 1.9


def fmt(values):
    return " ".join(f"{v:.10g}" for v in np.asarray(values, dtype=np.float64).flatten())


def xyzw_to_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float64)
    return np.array([quat[3], quat[0], quat[1], quat[2]])


def object_name(urdf_link_name):
    """OmniGibson's object name for a scene URDF link (bottom_cabinet-qacthv-0)."""
    return urdf_link_name.replace("-", "_")


def root_body_name(name):
    return f"{name}_root"


def freejoint_name(name):
    return f"{name}_freejoint"


def attach_prefix(name):
    return f"{name}_"


def part_attach_prefix(part):
    return f"{part}__"


def model_asset_name(category, model):
    return f"{category}-{model}"


def object_archive_path(category, model):
    """An object's archive, relative to the dataset root."""
    return f"objects/{category}/{model}/mjz/{model}.mjz"


def is_unit_scale(scale):
    return bool(np.all(np.abs(np.asarray(scale, dtype=np.float64) - 1.0) <= UNIT_SCALE_TOLERANCE))


def is_uniform_scale(scale):
    """Whether a scale is the same on every axis (within the unit tolerance,
    relatively); uniform frame scales work for every object."""
    scale = np.asarray(scale, dtype=np.float64)
    return bool(np.ptp(scale) <= UNIT_SCALE_TOLERANCE * np.max(np.abs(scale)))


def check_scalable(name, metadata, scale):
    """Raise ValueError if `scale` is anisotropic and the object, by its
    b1k_metadata, can't be stretched anisotropically."""
    if scale is None or is_unit_scale(scale) or is_uniform_scale(scale):
        return
    scaling = (metadata or {}).get("anisotropic_scaling", {"ok": True})
    if not scaling["ok"]:
        raise ValueError(
            f"{name} has anisotropic scale {fmt(scale)} but isn't anisotropically scalable: "
            + "; ".join(scaling["problems"])
        )


def metadata_json(metadata):
    return json.dumps(metadata, separators=(",", ":"), sort_keys=True)


def is_axis_aligned(vector, tolerance_deg=ALIGNMENT_TOLERANCE_DEG):
    """Whether a direction is within the tolerance of a coordinate axis."""
    v = np.asarray(vector, dtype=np.float64)
    n = np.linalg.norm(v)
    if n == 0:
        return False
    return np.max(np.abs(v)) / n >= np.cos(np.deg2rad(tolerance_deg))


def is_frame_aligned(rotation, tolerance_deg=ALIGNMENT_TOLERANCE_DEG):
    """Whether a rotation matrix maps each coordinate axis onto one."""
    rotation = np.asarray(rotation, dtype=np.float64)
    return all(is_axis_aligned(rotation[:, i], tolerance_deg) for i in range(3))
