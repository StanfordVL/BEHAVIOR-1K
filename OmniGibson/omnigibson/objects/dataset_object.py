"""Newton-native DatasetObject descriptor."""

from __future__ import annotations

import random
from pathlib import Path

from omnigibson.objects.usd_object import REGISTERED_OBJECTS, USDObject


class DatasetObject(USDObject):
    """Object descriptor for BEHAVIOR dataset assets."""

    def __init__(
        self,
        name,
        relative_prim_path=None,
        category="object",
        model=None,
        dataset_name="behavior-1k-assets",
        scale=None,
        visible=True,
        fixed_base=False,
        visual_only=False,
        kinematic_only=None,
        self_collisions=False,
        prim_type=None,
        link_physics_materials=None,
        load_config=None,
        abilities=None,
        include_default_states=True,
        bounding_box=None,
        in_rooms=None,
        expected_file_hash=None,
        position=None,
        orientation=None,
        **kwargs,
    ):
        if isinstance(in_rooms, str):
            in_rooms = [in_rooms]
        if bounding_box is not None and scale is not None:
            raise ValueError("Cannot define both scale and bounding_box for a DatasetObject.")
        if model is None:
            models = get_all_object_category_models(category=category, dataset_name=dataset_name)
            if not models:
                raise ValueError(f"No available models found for category {category!r}.")
            model = random.choice(models)

        self._model = model
        self.dataset_name = dataset_name
        self.bounding_box = bounding_box
        self.in_rooms = [] if in_rooms is None else list(in_rooms)

        load_config = {} if load_config is None else dict(load_config)
        load_config["bounding_box"] = bounding_box
        load_config["dataset_name"] = dataset_name

        super().__init__(
            name=name,
            usd_path=get_usd_path(category=category, model=model, dataset_name=dataset_name),
            encrypted=dataset_name == "behavior-1k-assets",
            relative_prim_path=relative_prim_path,
            category=category,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            kinematic_only=kinematic_only,
            self_collisions=self_collisions,
            prim_type=prim_type,
            link_physics_materials=link_physics_materials,
            load_config=load_config,
            abilities=abilities,
            include_default_states=include_default_states,
            expected_file_hash=expected_file_hash,
            position=position,
            orientation=orientation,
            **kwargs,
        )
        self.object_type = "DatasetObject"

    @property
    def model(self):
        return self._model

    @property
    def asset(self):
        return self

    def get_init_info(self):
        info = super().get_init_info()
        info["args"].update(
            {
                "category": self.category,
                "model": self.model,
                "dataset_name": self.dataset_name,
                "bounding_box": self.bounding_box,
                "in_rooms": self.in_rooms,
            }
        )
        return info

    @classmethod
    def get_usd_path(cls, category, model, dataset_name="behavior-1k-assets"):
        return get_usd_path(category=category, model=model, dataset_name=dataset_name)


def get_usd_path(category, model, dataset_name="behavior-1k-assets"):
    base = _data_path() / dataset_name / "objects" / category / model / "usd"
    usd_path = base / f"{model}.usd"
    encrypted_usd_path = base / f"{model}.encrypted.usd"
    if usd_path.exists():
        return usd_path
    if encrypted_usd_path.exists():
        return encrypted_usd_path
    raise FileNotFoundError(f"Could not find DatasetObject USD for {category}/{model} under {base}")


def get_all_object_category_models(category, dataset_name="behavior-1k-assets"):
    category_dir = _data_path() / dataset_name / "objects" / category
    if not category_dir.exists():
        return []
    return sorted(entry.name for entry in category_dir.iterdir() if entry.is_dir() and not entry.name.startswith("."))


def _data_path():
    import os

    if "OMNIGIBSON_DATA_PATH" in os.environ:
        return Path(os.environ["OMNIGIBSON_DATA_PATH"]).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "datasets"


REGISTERED_OBJECTS["DatasetObject"] = DatasetObject
