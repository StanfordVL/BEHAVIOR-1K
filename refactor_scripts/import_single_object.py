"""
Script to import scene and objects
"""
from igibson import app, ig_dataset_path, assets_path
from refactor_scripts.import_urdfs_from_scene import import_obj_urdf
from refactor_scripts.import_metadata import import_obj_metadata


##### SET THIS ######
OBJECT_CATEGORIES = ["apple", "apple", "table_knife"]
OBJECT_MODELS = ["00_0", "00_1", "1"]

IMPORT_RENDER_CHANNELS = True
#### YOU DONT NEED TO TOUCH ANYTHING BELOW HERE IDEALLY :) #####


if __name__ == "__main__":
    for obj_category, obj_model in zip(OBJECT_CATEGORIES, OBJECT_MODELS):
        import_obj_urdf(obj_category=obj_category, obj_model=obj_model, skip_if_exist=False)
        import_obj_metadata(
            obj_category=obj_category, obj_model=obj_model, import_render_channels=IMPORT_RENDER_CHANNELS
        )

    app.close()
