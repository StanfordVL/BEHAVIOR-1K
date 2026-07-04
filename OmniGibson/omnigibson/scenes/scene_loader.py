"""BEHAVIOR scene JSON loading for the Newton runtime."""

import json

from omnigibson.newton.assets import DEFAULT_DATASET_NAME, resolve_data_path
from omnigibson.objects import DatasetObject
from omnigibson.scenes.scene_base import Scene


STRUCTURE_CATEGORIES = frozenset({"floors", "walls", "ceilings", "lawn", "driveway", "fence", "roof", "background"})


def get_available_behavior_1k_scenes(data_path=None, dataset_name=DEFAULT_DATASET_NAME):
    """Return available BEHAVIOR scene model names from the dataset directory."""
    scenes_dir = _scenes_dir(data_path, dataset_name)
    return sorted(
        entry.name
        for entry in scenes_dir.iterdir()
        if entry.is_dir() and not entry.name.startswith(".") and (entry / "json").exists()
    )


def scene_from_behavior_scene(
    scene_model,
    *,
    data_path=None,
    dataset_name=DEFAULT_DATASET_NAME,
    scene_instance=None,
    load_object_categories=None,
):
    """Convert a BEHAVIOR scene JSON into a Newton-native Scene."""
    scene_data = _load_scene_json(
        scene_model, data_path=data_path, dataset_name=dataset_name, scene_instance=scene_instance
    )
    object_filter = set(load_object_categories) if load_object_categories is not None else None
    object_registry = scene_data.get("state", {}).get("registry", {}).get("object_registry", {})

    objects = []
    for object_name, init_info in scene_data.get("objects_info", {}).get("init_info", {}).items():
        if init_info.get("class_name") != "DatasetObject":
            continue
        args = init_info.get("args", {})
        category = args.get("category")
        model = args.get("model")
        if category is None or model is None:
            continue
        if object_filter is not None and category not in object_filter:
            continue

        root_link = object_registry.get(object_name, {}).get("root_link", {})
        objects.append(
            DatasetObject(
                name=args.get("name", object_name),
                category=category,
                model=model,
                dataset_name=dataset_name,
                position=tuple(root_link.get("pos", (0.0, 0.0, 0.0))),
                orientation=tuple(root_link.get("ori", (0.0, 0.0, 0.0, 1.0))),
                scale=_scale_from_json(args.get("scale", 1.0)),
                fixed_base=bool(args.get("fixed_base", False)),
                visual_only=bool(args.get("visual_only", False)),
            )
        )

    return Scene(
        name=scene_model,
        use_ground_plane=False,
        objects=tuple(objects),
    )


scene_spec_from_behavior_scene = scene_from_behavior_scene


def _load_scene_json(scene_model, *, data_path=None, dataset_name=DEFAULT_DATASET_NAME, scene_instance=None):
    scene_dir = _scenes_dir(data_path, dataset_name) / scene_model / "json"
    filename = f"{scene_model}_best.json" if scene_instance is None else f"{scene_instance}.json"
    json_path = scene_dir / filename
    if not json_path.exists():
        raise FileNotFoundError(f"BEHAVIOR scene JSON does not exist: {json_path}")
    with json_path.open("r") as f:
        return json.load(f)


def _scenes_dir(data_path=None, dataset_name=DEFAULT_DATASET_NAME):
    scenes_dir = resolve_data_path(data_path) / dataset_name / "scenes"
    if not scenes_dir.exists():
        raise FileNotFoundError(f"BEHAVIOR scenes directory does not exist: {scenes_dir}")
    return scenes_dir


def _scale_from_json(scale):
    if isinstance(scale, (int, float)):
        return float(scale)
    values = tuple(float(v) for v in scale)
    if len(values) != 3:
        raise ValueError(f"Expected scalar or 3-vector scale, got {scale!r}.")
    return values
