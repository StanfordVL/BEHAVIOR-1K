"""Asset resolution helpers for the Newton backend."""

import contextlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from cryptography.fernet import Fernet


DEFAULT_DATASET_NAME = "behavior-1k-assets"
DEFAULT_ROBOT_DATASET_NAME = "omnigibson-robot-assets"
HIDDEN_METALINK_TYPES = (
    "particlesource",
    "particlesink",
    "fillable",
    "particleremover",
    "particleapplier",
    "slicer",
)
VISIBLE_METALINK_TYPES = ("togglebutton",)


@dataclass(frozen=True)
class DatasetObjectSpec:
    """A BEHAVIOR DatasetObject asset selection."""

    category: str
    model: str
    dataset_name: str = DEFAULT_DATASET_NAME


@dataclass(frozen=True)
class RobotSpec:
    """An OmniGibson robot asset selection."""

    model: str
    asset_format: str = "usd"
    dataset_name: str = DEFAULT_ROBOT_DATASET_NAME


def resolve_data_path(data_path=None):
    """Resolve the OmniGibson data root without importing OmniGibson globals."""
    if data_path is not None:
        candidates = [Path(data_path).expanduser()]
    elif "OMNIGIBSON_DATA_PATH" in os.environ:
        candidates = [Path(os.environ["OMNIGIBSON_DATA_PATH"]).expanduser()]
    else:
        candidates = [
            Path(__file__).resolve().parents[3] / "datasets",
            Path.home() / "Research" / "BEHAVIOR-1K" / "datasets",
        ]

    for candidate in candidates:
        path = candidate.resolve()
        if _is_omnigibson_data_root(path):
            return path
    raise FileNotFoundError(f"OmniGibson data path does not exist. Checked: {', '.join(str(p) for p in candidates)}")


def _is_omnigibson_data_root(path):
    if not path.exists():
        return False
    return (path / DEFAULT_DATASET_NAME).exists() or (path / DEFAULT_ROBOT_DATASET_NAME).exists()


def get_all_object_categories(data_path=None, dataset_name=DEFAULT_DATASET_NAME):
    """Return DatasetObject categories available under the Newton data path."""
    objects_dir = _dataset_objects_dir(data_path, dataset_name)
    return sorted(entry.name for entry in objects_dir.iterdir() if entry.is_dir() and not entry.name.startswith("."))


def get_all_object_category_models(category, data_path=None, dataset_name=DEFAULT_DATASET_NAME):
    """Return DatasetObject model ids available for a category under the Newton data path."""
    category_dir = _dataset_objects_dir(data_path, dataset_name) / category
    if not category_dir.exists():
        raise FileNotFoundError(f"DatasetObject category does not exist: {category_dir}")
    return sorted(entry.name for entry in category_dir.iterdir() if entry.is_dir() and not entry.name.startswith("."))


def resolve_dataset_object_usd(spec, data_path=None):
    """Return the USD or encrypted USD path for a DatasetObject."""
    data_path = resolve_data_path(data_path)
    base = data_path / spec.dataset_name / "objects" / spec.category / spec.model / "usd"
    usd_path = base / f"{spec.model}.usd"
    encrypted_usd_path = base / f"{spec.model}.encrypted.usd"
    if usd_path.exists():
        return usd_path
    if encrypted_usd_path.exists():
        return encrypted_usd_path
    raise FileNotFoundError(f"Could not find DatasetObject USD for {spec.category}/{spec.model} under {base}")


def resolve_robot_asset(spec, data_path=None):
    """Return the robot USD path for an OmniGibson robot asset."""
    if spec.asset_format != "usd":
        raise ValueError("The Newton-first OmniGibson path imports robots through USD only.")

    data_path = resolve_data_path(data_path)
    base = data_path / spec.dataset_name / "models" / spec.model / spec.asset_format
    candidates = [base / f"{spec.model}.usda", base / f"{spec.model}.usd"]
    if base.exists():
        candidates.extend(sorted(base.glob("*.usda")))
        candidates.extend(sorted(base.glob("*.usd")))

    for path in candidates:
        if path.exists():
            return path

    family_usd_path = _resolve_robot_family_usd(spec, data_path)
    if family_usd_path is not None:
        return family_usd_path
    raise FileNotFoundError(f"Could not find robot USD asset under {base}")


def resolve_robot_default_joint_positions(spec, data_path=None):
    """Return authored default joint positions for YAML-backed robot families."""
    data_path = resolve_data_path(data_path)
    family_cfg = _load_robot_family_config(spec, data_path)
    if family_cfg is None:
        return None

    end_effectors = (family_cfg.get("manipulation") or {}).get("end_effectors") or {}
    eef_cfg = end_effectors.get("gripper") or next(iter(end_effectors.values()), None)
    if eef_cfg is None:
        return None
    return eef_cfg.get("default_joint_pos")


def resolve_robot_controller_metadata(spec, data_path=None):
    """Return controller-relevant metadata authored in the robot family YAML."""
    data_path = resolve_data_path(data_path)
    family_cfg = _load_robot_family_config(spec, data_path)
    if family_cfg is None:
        return {}

    metadata = {}
    two_wheel_cfg = family_cfg.get("two_wheel") or {}
    if "wheel_radius" in two_wheel_cfg:
        metadata["wheel_radius"] = float(two_wheel_cfg["wheel_radius"])
    if "wheel_axle_length" in two_wheel_cfg:
        metadata["wheel_axle_length"] = float(two_wheel_cfg["wheel_axle_length"])

    end_effectors = (family_cfg.get("manipulation") or {}).get("end_effectors") or {}
    eef_cfg = end_effectors.get("gripper") or next(iter(end_effectors.values()), None)
    if eef_cfg is not None:
        eef_link_names = eef_cfg.get("eef_link_names") or {}
        metadata["eef_link_names"] = tuple(eef_link_names[key] for key in sorted(eef_link_names))
    return metadata


def resolve_robot_fixed_base_default(spec, data_path=None):
    """Return the legacy OmniGibson default fixed-base decision for a robot.

    Isaac-era Robot forced fixed bases for every robot except non-holonomic
    locomotion robots. Preserve that rule for Newton USD imports so fixed arms
    and holonomic mobile manipulators do not become unintended floating bodies.
    """
    data_path = resolve_data_path(data_path)
    family_cfg = _load_robot_family_config(spec, data_path)
    if family_cfg is None:
        return True

    is_locomotion = any(family_cfg.get(key) is not None for key in ("locomotion", "two_wheel", "holonomic_base"))
    is_holonomic_base = family_cfg.get("holonomic_base") is not None
    can_be_floating = is_locomotion and not is_holonomic_base
    return not can_be_floating


def _resolve_robot_family_usd(spec, data_path):
    family_cfg = _load_robot_family_config(spec, data_path)
    if family_cfg is None:
        return None

    end_effectors = (family_cfg.get("manipulation") or {}).get("end_effectors") or {}
    eef_cfg = end_effectors.get("gripper") or next(iter(end_effectors.values()), None)
    if not eef_cfg or "usd_path" not in eef_cfg:
        return None

    usd_path = data_path / spec.dataset_name / eef_cfg["usd_path"]
    return usd_path if usd_path.exists() else None


def _load_robot_family_config(spec, data_path):
    family_yaml = data_path / spec.dataset_name / "models" / spec.model / f"{spec.model}.yaml"
    if not family_yaml.exists():
        return None

    import yaml

    with family_yaml.open("r") as f:
        return yaml.load(f, Loader=yaml.FullLoader) or {}


def _key_path(data_path):
    return resolve_data_path(data_path) / "omnigibson.key"


def _dataset_objects_dir(data_path=None, dataset_name=DEFAULT_DATASET_NAME):
    objects_dir = resolve_data_path(data_path) / dataset_name / "objects"
    if not objects_dir.exists():
        raise FileNotFoundError(f"DatasetObject directory does not exist: {objects_dir}")
    return objects_dir


def _decrypt_usd(encrypted_usd_path, data_path, temp_dir, scale=None):
    key_path = _key_path(data_path)
    if not key_path.exists():
        raise FileNotFoundError(f"Missing BEHAVIOR asset key: {key_path}")

    source_model_dir = encrypted_usd_path.parent.parent
    temp_model_dir = temp_dir / source_model_dir.name
    temp_usd_dir = temp_model_dir / encrypted_usd_path.parent.name
    temp_usd_dir.mkdir(parents=True)
    for source_entry in source_model_dir.iterdir():
        if source_entry.name == encrypted_usd_path.parent.name:
            continue
        os.symlink(source_entry, temp_model_dir / source_entry.name)

    decrypted = Fernet(key_path.read_bytes()).decrypt(encrypted_usd_path.read_bytes())
    usd_path = temp_usd_dir / encrypted_usd_path.name.replace(".encrypted", "")
    usd_path.write_bytes(decrypted)
    _prepare_usd_for_newton(usd_path, scale=scale)
    return usd_path


def _copy_usd_for_preprocessing(source_path, temp_dir, scale=None):
    source_model_dir = source_path.parent.parent
    temp_model_dir = temp_dir / source_model_dir.name
    temp_usd_dir = temp_model_dir / source_path.parent.name
    temp_usd_dir.mkdir(parents=True)

    for source_entry in source_model_dir.iterdir():
        if source_entry.name == source_path.parent.name:
            continue
        os.symlink(source_entry, temp_model_dir / source_entry.name)

    for source_entry in source_path.parent.iterdir():
        if source_entry.name == source_path.name:
            continue
        os.symlink(source_entry, temp_usd_dir / source_entry.name)

    usd_path = temp_usd_dir / source_path.name
    shutil.copyfile(source_path, usd_path)
    _prepare_usd_for_newton(usd_path, scale=scale)
    return usd_path


def _prepare_usd_for_newton(usd_path, scale=None):
    scale_applied = _apply_root_scale(usd_path, scale)
    _ensure_authored_mass_properties(usd_path, refresh_existing=scale_applied)
    _hide_metalink_visuals(usd_path)


def _apply_root_scale(usd_path, scale):
    if scale is None:
        return False

    from pxr import Gf, Usd, UsdGeom

    values = tuple(float(v) for v in scale)
    if len(values) != 3:
        raise ValueError(f"Expected scalar or 3-vector root scale, got {scale!r}.")
    if values == (1.0, 1.0, 1.0):
        return False

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"USD failed to open for Newton root-scale preprocessing: {usd_path}")
    root = stage.GetDefaultPrim()
    if not root:
        raise RuntimeError(f"USD does not define a default prim for root-scale preprocessing: {usd_path}")

    xformable = UsdGeom.Xformable(root)
    scale_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale]
    if scale_ops:
        scale_ops[0].Set(Gf.Vec3d(*values))
    else:
        xformable.AddScaleOp().Set(Gf.Vec3d(*values))
    stage.Save()
    del stage
    return True


def _ensure_authored_mass_properties(usd_path, refresh_existing=False):
    # Some BEHAVIOR rigid bodies omit mass properties accepted by Newton/MuJoCo,
    # and native mass-property computation has failed for affected assets.
    # Author them on the temporary USD only, using a coarse bounding-box inertia
    # because near-zero rotational inertia destabilizes awakened furniture. See
    # workaround W5 in docs/other/newton_migration.md.
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"USD failed to open after decryption: {usd_path}")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )

    changed = False
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue

        mass_api = UsdPhysics.MassAPI(prim)
        if not prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI.Apply(prim)

        if not mass_api.GetMassAttr().HasAuthoredValue():
            mass_api.GetMassAttr().Set(1.0)
            changed = True
        if refresh_existing or not mass_api.GetCenterOfMassAttr().HasAuthoredValue():
            mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            changed = True
        if refresh_existing or not mass_api.GetDiagonalInertiaAttr().HasAuthoredValue():
            mass_api.GetDiagonalInertiaAttr().Set(_approximate_diagonal_inertia(prim, mass_api, bbox_cache))
            changed = True

    if changed:
        stage.Save()
    del stage


def _approximate_diagonal_inertia(prim, mass_api, bbox_cache):
    """Approximate missing rigid-body inertia without calling ComputeMassProperties."""
    from pxr import Gf

    mass = mass_api.GetMassAttr().Get()
    mass = float(mass) if mass is not None and mass > 0 else 1.0

    center_of_mass = mass_api.GetCenterOfMassAttr().Get() or Gf.Vec3f(0.0, 0.0, 0.0)
    try:
        local_range = bbox_cache.ComputeLocalBound(prim).ComputeAlignedRange()
        if local_range.IsEmpty():
            raise ValueError("empty local bound")
        extent = local_range.GetSize()
        center = local_range.GetMidpoint()
    except Exception:
        extent = Gf.Vec3d(0.05, 0.05, 0.05)
        center = Gf.Vec3d(0.0, 0.0, 0.0)

    sx, sy, sz = (max(float(value), 0.05) for value in extent)
    cx = float(center[0]) - float(center_of_mass[0])
    cy = float(center[1]) - float(center_of_mass[1])
    cz = float(center[2]) - float(center_of_mass[2])

    ix = mass * (sy * sy + sz * sz) / 12.0 + mass * (cy * cy + cz * cz)
    iy = mass * (sx * sx + sz * sz) / 12.0 + mass * (cx * cx + cz * cz)
    iz = mass * (sx * sx + sy * sy) / 12.0 + mass * (cx * cx + cy * cy)
    return Gf.Vec3f(max(ix, 1.0e-3), max(iy, 1.0e-3), max(iz, 1.0e-3))


def _hide_metalink_visuals(usd_path):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"USD failed to open for Newton preprocessing: {usd_path}")

    changed = False
    default_purpose = getattr(UsdGeom.Tokens, "default_", "default")

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath()).lower()
        if "meta" not in prim_path:
            continue
        if prim.GetTypeName() != "Mesh":
            continue

        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            continue

        purpose_attr = imageable.CreatePurposeAttr()
        visibility_attr = imageable.CreateVisibilityAttr()

        if any(metalink_type in prim_path for metalink_type in VISIBLE_METALINK_TYPES):
            if purpose_attr.Get() != default_purpose:
                purpose_attr.Set(default_purpose)
                changed = True
            if visibility_attr.Get() == UsdGeom.Tokens.invisible:
                visibility_attr.Set(UsdGeom.Tokens.inherited)
                changed = True
        elif any(metalink_type in prim_path for metalink_type in HIDDEN_METALINK_TYPES):
            if purpose_attr.Get() != UsdGeom.Tokens.guide:
                purpose_attr.Set(UsdGeom.Tokens.guide)
                changed = True
            if visibility_attr.Get() != UsdGeom.Tokens.invisible:
                visibility_attr.Set(UsdGeom.Tokens.invisible)
                changed = True

    if changed:
        stage.Save()
    del stage


@contextlib.contextmanager
def prepared_dataset_object_usd(source_path, data_path=None, scale=None):
    """Yield a loadable DatasetObject USD path, decrypting encrypted assets if needed."""
    source_path = Path(source_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"DatasetObject USD does not exist: {source_path}")

    # Imported mesh resources have retained native references to these USD-backed
    # files after model finalization. Keep them for the process lifetime until
    # repeated build/close memory tests satisfy workaround W4 in the migration
    # record.
    temp_dir = Path(tempfile.mkdtemp(prefix="og-newton-object-"))
    if source_path.name.endswith(".encrypted.usd"):
        yield _decrypt_usd(source_path, data_path, temp_dir, scale=scale)
    else:
        yield _copy_usd_for_preprocessing(source_path, temp_dir, scale=scale)
