"""
Script to import scene and objects
"""
from omnigibson import app, og_dataset_path, assets_path
from refactor_scripts.import_urdfs_from_scene import import_obj_urdf
from refactor_scripts.import_metadata import import_obj_metadata
import os


##### SET THIS ######
IMPORT_RENDER_CHANNELS = True
#### YOU DONT NEED TO TOUCH ANYTHING BELOW HERE IDEALLY :) #####


if __name__ == "__main__":
    for obj_category in os.listdir(os.path.join(og_dataset_path, "objects")):
        for obj_model in os.listdir(os.path.join(og_dataset_path, "objects", obj_category)):
            print(f"IMPORTING CATEGORY/MODEL {obj_category}/{obj_model}...")
            import_obj_urdf(obj_category=obj_category, obj_model=obj_model, skip_if_exist=False)
            import_obj_metadata(
                obj_category=obj_category, obj_model=obj_model, import_render_channels=IMPORT_RENDER_CHANNELS
            )

    app.close()
