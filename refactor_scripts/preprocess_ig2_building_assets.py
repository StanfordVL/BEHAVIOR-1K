from omnigibson import og_dataset_path
import omnigibson.utils.transform_utils as T
import xml.etree.ElementTree as ET
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime
import shutil


##### SET THIS ######
SCENE = "Rs_int"
#### YOU DONT NEED TO TOUCH ANYTHING BELOW HERE IDEALLY :) #####


def copy_building_assets_to_objects_folder(scene_id):
    for category in {"floors", "ceilings", "walls"}:
        cat_nonplural = category[:-1]
        # Copy the URDF, mesh files, and material files to the object folder, so that these have the same
        # structure as other objects in og_dataset
        old_dir = f"{og_dataset_path}/scenes/{scene_id}"
        old_path = f"{old_dir}/urdf/{scene_id}_{category}.urdf"
        obj_model = f"{scene_id}"
        new_dir = f"{og_dataset_path}/objects/{category}/{obj_model}"
        usd_path = f"{og_dataset_path}/objects/{category}/{obj_model}/usd/{obj_model}.usd"

        # Create new directories
        Path(new_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(new_dir, "material")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(new_dir, "misc")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(new_dir, "shape/visual")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(new_dir, "shape/collision")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(new_dir, "visualizations")).mkdir(parents=True, exist_ok=True)

        # Copy URDF file
        shutil.copy(old_path, os.path.join(new_dir, f"{obj_model}.urdf"))

        # Copy shape files
        for shape_subdir in os.listdir(os.path.join(old_dir, "shape")):
            for shape_file in os.listdir(os.path.join(old_dir, "shape", shape_subdir)):
                if cat_nonplural in shape_file:
                    old_shape_fpath = os.path.join(old_dir, "shape", shape_subdir, shape_file)
                    new_shape_fpath = os.path.join(new_dir, "shape", shape_subdir, shape_file)
                    if ".mtl" in shape_file:
                        # We need to update the map naming for this
                        with open(old_shape_fpath, "r") as f:
                            old_lines = f.readlines()
                        lines = []
                        for line in old_lines:
                            if "map" in line:
                                line_part_1 = "/".join(line.split("/")[:-1])
                                line_part_2 = line.split("/")[-1]
                                lines.append(f"{line_part_1}_{line_part_2}")
                            else:
                                lines.append(line)
                        with open(new_shape_fpath, "w+") as f:
                            f.writelines(lines)
                    else:
                        shutil.copy(old_shape_fpath, new_shape_fpath)

        # Copy material files
        for mat_subdir in os.listdir(os.path.join(old_dir, "material")):
            if cat_nonplural in mat_subdir:
                for mat_file in os.listdir(os.path.join(old_dir, "material", mat_subdir)):
                    old_mat_fpath = os.path.join(old_dir, "material", mat_subdir, mat_file)
                    shutil.copy(old_mat_fpath,
                                    os.path.join(new_dir, "material", f"{mat_subdir}_{mat_file}"))

        # Copy material groups
        for misc_file in os.listdir(os.path.join(old_dir, "misc")):
            if category in misc_file:
                shutil.copy(os.path.join(old_dir, "misc", misc_file),
                            os.path.join(new_dir, "misc", "material_groups.json"))

        # Write dummy metadata
        metadata = dict()
        with open(os.path.join(new_dir, "misc", "metadata.json"), "w+") as f:
            json.dump(metadata, f)
