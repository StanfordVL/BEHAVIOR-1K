"""Config helpers for the native Newton backend."""

from pathlib import Path
import re

import yaml

from omnigibson.newton.assets import DatasetObjectSpec, RobotSpec, resolve_robot_fixed_base_default
from omnigibson.objects import DatasetObject, LightObject, PrimitiveObject, USDObject
from omnigibson.scenes.scene_base import NewtonRobotSpec, NewtonSceneSpec


def load_newton_config(config):
    """Load a BEHAVIOR-style YAML config or return a provided mapping."""
    if isinstance(config, dict):
        return config

    path = Path(config).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Newton config does not exist: {path}")
    with path.open("r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def simulator_from_config(config, *, data_path=None, dataset_usd_path=None, robot_asset_path=None):
    """Create a NewtonSceneSimulator from a minimal OmniGibson-style config.

    Supported config forms:

    - Native fields under ``newton.scene``, ``newton.object``, ``newton.robot``, and
      ``newton.simulation``.
    - Existing top-level ``objects`` and ``robots`` lists.
    """
    cfg = load_newton_config(config)

    from omnigibson.simulator.newton import NewtonSceneSimulator

    scene = scene_from_config(
        cfg,
        data_path=data_path,
        dataset_usd_path=dataset_usd_path,
        robot_asset_path=robot_asset_path,
    )
    sim_cfg = simulation_config_from_config(cfg)

    return NewtonSceneSimulator(
        scene,
        data_path=data_path,
        config=sim_cfg,
    )


def scene_from_config(config, *, data_path=None, dataset_usd_path=None, robot_asset_path=None):
    """Return a NewtonSceneSpec from a Newton or legacy-style config."""
    cfg = load_newton_config(config)
    newton_cfg = cfg.get("newton", {}) or {}
    scene_cfg = cfg.get("scene", {}) or {}
    native_scene_cfg = newton_cfg.get("scene", {}) or {}
    simulation_cfg = newton_cfg.get("simulation", {}) or {}

    name = native_scene_cfg.get("name", scene_cfg.get("model", scene_cfg.get("type", "scene")))
    use_ground_plane = native_scene_cfg.get(
        "use_ground_plane",
        scene_cfg.get("use_floor_plane", scene_cfg.get("use_ground_plane", True)),
    )
    default_object_position = tuple(simulation_cfg.get("object_position", (1.0, 0.0, 0.5)))
    default_robot_position = tuple(simulation_cfg.get("robot_position", (-1.0, 0.0, 0.0)))

    objects_cfg = _object_configs(cfg, newton_cfg)
    robots_cfg = _robot_configs(cfg, newton_cfg)
    robots = tuple(
        _newton_robot_instance(robot_cfg, idx, default_robot_position, robot_asset_path)
        for idx, robot_cfg in enumerate(robots_cfg)
    )
    if scene_cfg.get("type") == "InteractiveTraversableScene" and not objects_cfg:
        from omnigibson.scenes.scene_loader import scene_spec_from_behavior_scene

        loaded_scene = scene_spec_from_behavior_scene(
            scene_cfg["scene_model"],
            data_path=data_path,
            scene_instance=scene_cfg.get("scene_instance"),
            load_object_categories=scene_cfg.get("load_object_categories"),
        )
        return NewtonSceneSpec(
            name=loaded_scene.name,
            use_ground_plane=loaded_scene.use_ground_plane,
            objects=loaded_scene.objects,
            robots=robots,
            lights=loaded_scene.lights,
        )

    object_instances = []
    light_instances = []
    for idx, obj_cfg in enumerate(objects_cfg):
        if obj_cfg.get("type") == "LightObject":
            light_instances.append(_newton_light_instance(obj_cfg, idx))
        else:
            object_instances.append(_newton_object_instance(obj_cfg, idx, default_object_position, dataset_usd_path))

    return NewtonSceneSpec(
        name=name,
        use_ground_plane=use_ground_plane,
        objects=tuple(object_instances),
        robots=robots,
        lights=tuple(light_instances),
    )


def specs_from_config(config):
    """Return DatasetObjectSpec and RobotSpec from a Newton config."""
    scene = scene_from_config(config)
    if not scene.objects or not scene.robots:
        raise ValueError("Config must define at least one object and one robot to use specs_from_config().")
    return scene.objects[0].asset, scene.robots[0].asset


def simulation_config_from_config(config):
    """Return NewtonSimulationConfig from a Newton config."""
    from omnigibson.simulator.newton import NewtonSimulationConfig

    cfg = load_newton_config(config)
    newton_cfg = cfg.get("newton", {}) or {}
    simulation = _normalize_simulation_cfg(newton_cfg.get("simulation", {}) or {})
    allowed = {field for field in NewtonSimulationConfig.__dataclass_fields__}
    kwargs = {key: value for key, value in simulation.items() if key in allowed}
    return NewtonSimulationConfig(**kwargs)


def _normalize_simulation_cfg(simulation):
    """Normalize Isaac-Lab-style Newton config aliases into NewtonSimulationConfig fields."""
    normalized = dict(simulation)

    # Isaac Lab names the step subdivision ``num_substeps``. OmniGibson already
    # exposed ``sim_substeps`` during the Newton migration, so accept both while
    # keeping the internal field name stable.
    if "num_substeps" in normalized and "sim_substeps" not in normalized:
        normalized["sim_substeps"] = normalized["num_substeps"]

    solver_cfg = normalized.pop("solver_cfg", None) or {}
    solver_aliases = {
        "iterations": "solver_iterations",
        "ls_iterations": "solver_ls_iterations",
        "ccd_iterations": "solver_ccd_iterations",
        "sdf_iterations": "solver_sdf_iterations",
        "sdf_initpoints": "solver_sdf_initpoints",
        "method": "solver_method",
        "solver": "solver_method",
        "integrator": "solver_integrator",
        "cone": "solver_cone",
        "jacobian": "solver_jacobian",
        "impratio": "solver_impratio",
        "tolerance": "solver_tolerance",
        "ls_tolerance": "solver_ls_tolerance",
        "ccd_tolerance": "solver_ccd_tolerance",
        "ls_parallel": "solver_ls_parallel",
    }
    for source_key, target_key in solver_aliases.items():
        if source_key in solver_cfg and target_key not in normalized:
            normalized[target_key] = solver_cfg[source_key]
    for direct_key in (
        "njmax",
        "nconmax",
        "use_mujoco_contacts",
        "use_mujoco_cpu",
        "enable_multiccd",
        "disable_contacts",
        "update_data_interval",
    ):
        if direct_key in solver_cfg and direct_key not in normalized:
            normalized[direct_key] = solver_cfg[direct_key]

    shape_cfg = normalized.pop("default_shape_cfg", None) or {}
    shape_aliases = {
        "ke": "default_shape_ke",
        "kd": "default_shape_kd",
        "kf": "default_shape_kf",
        "mu": "default_shape_mu",
    }
    for source_key, target_key in shape_aliases.items():
        if source_key in shape_cfg and target_key not in normalized:
            normalized[target_key] = shape_cfg[source_key]

    return normalized


def _object_configs(cfg, newton_cfg):
    if "objects" in newton_cfg:
        return newton_cfg.get("objects") or []
    if "object" in newton_cfg:
        return [newton_cfg.get("object") or {}]
    return cfg.get("objects") or []


def _robot_configs(cfg, newton_cfg):
    if "robots" in newton_cfg:
        return newton_cfg.get("robots") or []
    if "robot" in newton_cfg:
        return [newton_cfg.get("robot") or {}]
    return cfg.get("robots") or []


def _dataset_object_from_config(object_cfg):
    if "category" not in object_cfg or "model" not in object_cfg:
        raise ValueError(f"DatasetObject config must define 'category' and 'model': {object_cfg}")
    return DatasetObjectSpec(
        category=object_cfg["category"],
        model=object_cfg["model"],
        dataset_name=object_cfg.get("dataset_name", "behavior-1k-assets"),
    )


def _robot_from_config(robot_cfg):
    model = robot_cfg.get("model")
    if model is None:
        if "type" not in robot_cfg:
            raise ValueError(f"Robot config must define either 'model' or 'type': {robot_cfg}")
        model = _robot_type_to_model(robot_cfg["type"])

    return RobotSpec(
        model=model,
        asset_format=robot_cfg.get("asset_format", "usd"),
        dataset_name=robot_cfg.get("dataset_name", "omnigibson-robot-assets"),
    )


def _newton_object_instance(object_cfg, idx, default_position, dataset_usd_path):
    object_type = object_cfg.get("type", "DatasetObject")
    usd_path = object_cfg.get("usd_path", dataset_usd_path if idx == 0 else None)
    if object_type == "DatasetObject":
        asset = _dataset_object_from_config(object_cfg)
        fallback_name = f"{asset.category}_{asset.model}" if idx == 0 else f"{asset.category}_{asset.model}_{idx}"
        cls = DatasetObject
        kwargs = {
            "category": asset.category,
            "model": asset.model,
            "dataset_name": asset.dataset_name,
        }
    elif object_type == "USDObject":
        if usd_path is None:
            raise ValueError(f"USDObject config must define 'usd_path': {object_cfg}")
        fallback_name = f"usd_object_{idx}"
        cls = USDObject
        kwargs = {"usd_path": usd_path}
    elif object_type == "PrimitiveObject":
        fallback_name = f"primitive_object_{idx}"
        cls = PrimitiveObject
        kwargs = {"primitive_type": object_cfg.get("primitive_type", "Cube")}
    else:
        raise ValueError(f"Unsupported Newton object type: {object_type}")

    name = object_cfg.get("name", fallback_name)
    return cls(
        name=name,
        **kwargs,
        position=tuple(object_cfg.get("position", default_position)),
        orientation=tuple(object_cfg.get("orientation", (0.0, 0.0, 0.0, 1.0))),
        scale=_scale_from_config(object_cfg.get("scale", 1.0)),
        fixed_base=bool(object_cfg.get("fixed_base", False)),
        visual_only=bool(object_cfg.get("visual_only", False)),
    )


def _newton_robot_instance(robot_cfg, idx, default_position, robot_asset_path):
    asset = _robot_from_config(robot_cfg)
    name = robot_cfg.get("name", f"robot_{asset.model}" if idx == 0 else f"robot_{asset.model}_{idx}")
    asset_path = robot_cfg.get("asset_path", robot_asset_path if idx == 0 else None)
    fixed_base = bool(robot_cfg["fixed_base"]) if "fixed_base" in robot_cfg else resolve_robot_fixed_base_default(asset)
    return NewtonRobotSpec(
        name=name,
        asset=asset,
        position=tuple(robot_cfg.get("position", default_position)),
        orientation=tuple(robot_cfg.get("orientation", (0.0, 0.0, 0.0, 1.0))),
        asset_path=Path(asset_path).expanduser().resolve() if asset_path is not None else None,
        fixed_base=fixed_base,
        action_normalize=bool(robot_cfg.get("action_normalize", True)),
        obs_modalities=_normalized_obs_modalities(robot_cfg.get("obs_modalities", ("rgb",))),
    )


def _normalized_obs_modalities(modalities):
    if isinstance(modalities, str):
        modalities = (modalities,)
    return tuple(modalities)


def _newton_light_instance(light_cfg, idx):
    return LightObject(
        name=light_cfg.get("name", f"light_{idx}"),
        light_type=light_cfg.get("light_type", "Sphere"),
        position=tuple(light_cfg.get("position", (0.0, 0.0, 2.0))),
        intensity=float(light_cfg.get("intensity", 1.0e5)),
        radius=float(light_cfg.get("radius", 0.01)),
    )


def _scale_from_config(scale):
    if isinstance(scale, (int, float)):
        return float(scale)
    values = tuple(float(v) for v in scale)
    if len(values) != 3:
        raise ValueError(f"Expected scalar or 3-vector scale, got {scale!r}.")
    return values


def _robot_type_to_model(robot_type):
    robot_type = str(robot_type)
    normalized = re.sub(r"(?<!^)(?=[A-Z])", "_", robot_type).replace("__", "_").lower()
    return {
        "r1_pro": "r1pro",
        "behavior_robot": "behavior_robot",
    }.get(normalized, normalized)
