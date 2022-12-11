import omnigibson as og
from refactor_scripts.old_scene_classes import InteractiveTraversableSceneOld
from pathlib import Path
import os
import json
from omnigibson.utils.config_utils import NumpyEncoder

# SCENE_ID = "Pomaria_0_int"


# if __name__ == "__main__":
for scene_model in os.listdir(os.path.join(og.og_dataset_path, "scenes")):
    json_path = f"{og.og_dataset_path}/scenes/{scene_model}/json/{scene_model}_best.json"
    if "background" not in scene_model and not os.path.exists(json_path):
        scene = InteractiveTraversableSceneOld(scene_model=scene_model)

        og.sim.stop()
        og.sim.import_scene(scene)
        og.sim.play()
        json_path = f"{og.og_dataset_path}/scenes/{scene_model}/json/{scene_model}_best.json"
        Path(os.path.dirname(json_path)).mkdir(parents=True, exist_ok=True)
        og.sim.save(json_path=json_path)

        # Load the json, remove the init_info because we don't need it, then save it again
        with open(json_path, "r") as f:
            scene_info = json.load(f)

        scene_info.pop("init_info")

        with open(json_path, "w+") as f:
            json.dump(scene_info, f, cls=NumpyEncoder, indent=4)

og.shutdown()

# scene = InteractiveTraversableSceneOld(
#     scene_model=SCENE_ID,
# )
#
# og.sim.import_scene(scene)
# og.sim.play()
# json_path = f"{og.og_dataset_path}/scenes/{SCENE_ID}/json/{SCENE_ID}_best.json"
# Path(os.path.dirname(json_path)).mkdir(parents=True, exist_ok=True)
# og.sim.save(json_path=json_path)
#
# # Load the json, remove the init_info because we don't need it, then save it again
# with open(json_path, "r") as f:
#     scene_info = json.load(f)
#
# scene_info.pop("init_info")
#
# with open(json_path, "w+") as f:
#     json.dump(scene_info, f, cls=NumpyEncoder, indent=4)
