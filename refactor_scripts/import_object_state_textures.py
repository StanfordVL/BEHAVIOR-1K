"""
Script to import scene and objects
"""
from omnigibson import app, og_dataset_path, assets_path
from refactor_scripts.import_metadata import copy_object_state_textures
import os


##### SET THIS ######

#### YOU DONT NEED TO TOUCH ANYTHING BELOW HERE IDEALLY :) #####


if __name__ == "__main__":
    for obj_category in os.listdir(os.path.join(og_dataset_path, "objects")):
        for obj_model in os.listdir(os.path.join(og_dataset_path, "objects", obj_category)):
            print(f"IMPORTING OBJECT STATE TEXTURES FOR CATEGORY/MODEL {obj_category}/{obj_model}...")
            copy_object_state_textures(obj_category=obj_category, obj_model=obj_model)

    app.close()
