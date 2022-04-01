"""
Script to import scene and objects
"""
from igibson import app, ig_dataset_path, assets_path
from refactor_scripts.import_urdfs_from_scene import import_objects_from_scene_urdf
from refactor_scripts.import_metadata import import_models_metadata_from_scene
from refactor_scripts.import_scene_template import import_models_template_from_scene


##### SET THIS ######
SCENE = "Rs_int"
#### YOU DONT NEED TO TOUCH ANYTHING BELOW HERE IDEALLY :) #####


if __name__ == "__main__":
    urdf = f"{ig_dataset_path}/scenes/{SCENE}/urdf/{SCENE}_best.urdf"
    usd_out = f"{ig_dataset_path}/scenes/{SCENE}/urdf/{SCENE}_best_template.usd"

    import_objects_from_scene_urdf(urdf=urdf)
    import_models_metadata_from_scene(urdf=urdf)
    import_models_template_from_scene(urdf=urdf, usd_out=usd_out)

    app.close()
