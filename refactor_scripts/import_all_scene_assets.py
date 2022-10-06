"""
Script to import scene's objects
"""
from igibson import app, ig_dataset_path, assets_path
from refactor_scripts.import_urdfs_from_scene import import_objects_from_scene_urdf
from refactor_scripts.import_metadata import import_models_metadata_from_scene
from refactor_scripts.import_scene_template import import_models_template_from_scene
from refactor_scripts.preprocess_ig2_building_assets import copy_building_assets_to_objects_folder
import os


##### SET THIS ######
IMPORT_RENDER_CHANNELS = True
#### YOU DONT NEED TO TOUCH ANYTHING BELOW HERE IDEALLY :) #####


if __name__ == "__main__":
    for scene in os.listdir(os.path.join(ig_dataset_path, "scenes")):
        if "background" not in scene:
            urdf = f"{ig_dataset_path}/scenes/{scene}/urdf/{scene}_best.urdf"
            usd_out = f"{ig_dataset_path}/scenes/{scene}/usd/{scene}_best_template.usd"

            copy_building_assets_to_objects_folder(scene)
            import_objects_from_scene_urdf(urdf=urdf)
            import_models_metadata_from_scene(urdf=urdf, import_render_channels=IMPORT_RENDER_CHANNELS)
            import_models_template_from_scene(urdf=urdf, usd_out=usd_out)

    app.close()
