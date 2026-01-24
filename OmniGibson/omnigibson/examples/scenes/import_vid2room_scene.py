import pathlib
import json
import torch as th
from collections import Counter

import omnigibson as og
from omnigibson.objects import DatasetObject
from omnigibson.scenes import Scene
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.asset_utils import get_dataset_path

log = create_module_logger(module_name=__name__)


def load_vid2room_scene(scene_input_json):
    scene_input_json = pathlib.Path(scene_input_json)

    # Load the scene JSON
    scene_contents = json.loads(scene_input_json.read_text())

    # Load all the objects manually into a scene
    scene = Scene(use_floor_plane=True, floor_plane_visible=False)
    og.sim.import_scene(scene)

    walls = DatasetObject(
        name="walls",
        category="walls",
        model="walls41bliving",
        fixed_base=True,
        dataset_name="vid2room",
    )
    scene.add_object(walls)

    floor = DatasetObject(
        name="floor",
        category="floor",
        model="floor41bliving",
        fixed_base=True,
        dataset_name="vid2room",
    )
    scene.add_object(floor)

    category_counts = Counter()
    for obj_name, obj_data in scene_contents.items():
        if obj_name.startswith("door") or obj_name.startswith("window"):
            continue
        else:
            segment, category, idx = obj_name.split("-")
            category = "".join(c if c.isalnum() else "_" for c in category.lower())
            model = "".join(c if c.isalnum() else "" for c in (category + idx + segment).lower())

        obj = DatasetObject(
            name=f"{category}_{category_counts[category]}",
            category=category,
            model=model,
            bounding_box=th.as_tensor(obj_data["bbox_extents"]),
            fixed_base=True,
            dataset_name="vid2room",
        )
        scene.add_object(obj)
        obj.set_bbox_center_position_orientation(
            position=th.as_tensor(obj_data["bbox_center"]), orientation=th.as_tensor(obj_data["rotation"])
        )
        category_counts[category] += 1

    # Play the simulator, then save
    og.sim.play()

    # Take a sim step
    for _ in range(500):
        og.sim.step()

    return scene


if __name__ == "__main__":
    if og.sim:
        og.clear()
    else:
        og.launch()

    scene = load_vid2room_scene(r"D:\vid2room-scene\scene.json")
    scene_dir = pathlib.Path(get_dataset_path("vid2room")) / "scenes" / "scene" / "json"
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene.save(str(scene_dir / "scene_best.json"))

    while True:
        og.sim.step()
