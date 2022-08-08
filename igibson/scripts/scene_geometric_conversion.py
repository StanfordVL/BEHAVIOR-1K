import glob
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

import igibson

if __name__ == "__main__":
    object_categories = glob.glob(os.path.join(igibson.ig_dataset_path, "objects/*"))

    objects_to_process = {}
    idx = 0
    for category_path in object_categories:
        if os.path.isdir(category_path):
            for object_instance in os.listdir(category_path):
                heights_per_link = os.path.join(category_path, object_instance, "misc", "heights_per_link.json")
                if os.path.exists(heights_per_link):
                    with open(heights_per_link, "r") as f:
                        link_data = json.load(f)
                    height_maps = {}
                    if "under" in link_data:
                        del link_data["under"]
                    for link_folder in link_data:
                        height_maps[link_folder] = {}
                        for link in link_data[link_folder]:
                            height_map_dir = os.path.join(
                                category_path, object_instance, "misc", "height_maps_per_link", link_folder, link
                            )
                            link_height_maps = []
                            for link_png in os.listdir(height_map_dir):
                                link_height_maps.append(
                                    os.path.join(
                                        category_path,
                                        object_instance,
                                        "misc",
                                        "height_maps_per_link",
                                        link_folder,
                                        link,
                                        link_png,
                                    )
                                )
                            height_maps[link_folder][link] = link_height_maps

                    objects_to_process[idx] = {
                        "category": os.path.basename(category_path),
                        "category_path": category_path,
                        "object_instance": object_instance,
                        "urdf": os.path.join(category_path, object_instance, f"{object_instance}.urdf"),
                        "height_maps": height_maps,
                        "offset": link_data,
                    }
                    idx += 1

    insert_visual_mesh = True
    for obj_name, obj_properties in objects_to_process.items():
        tree = ET.parse(obj_properties["urdf"])
        links = tree.findall("link")
        succesful_write = False
        for support_type in obj_properties["height_maps"]:
            for child_link in obj_properties["height_maps"][support_type]:
                for idx, supporting_suface in enumerate(obj_properties["height_maps"][support_type][child_link]):
                    image = Image.open(supporting_suface)
                    image_height, image_width = np.array(image).shape
                    row, column = np.where(image)
                    max_row = np.max(row)
                    min_row = np.min(row)
                    max_column = np.max(column)
                    min_column = np.min(column)
                    center_height = np.mean([max_row, min_row])
                    center_width = np.mean([max_column, min_column])

                    x_offset = (center_width - image_width / 2) * 0.01
                    y_offset = (center_height - image_height / 2) * 0.01

                    depth = (max_row - min_row) * 0.01 * 0.98
                    width = (max_column - min_column) * 0.01 * 0.98
                    euler = [0, 0, 0]
                    # offset = [x_offset, y_offset, obj_properties["offset"][support_type][child_link][0]]
                    offset = [0, y_offset, obj_properties["offset"][support_type][child_link][idx]]
                    size = np.array([width, depth, 0.01])
                    for urdf_link in tree.findall("link"):
                        link_name = urdf_link.get("name")
                        if link_name == child_link:

                            if insert_visual_mesh:
                                visual_mesh = ET.fromstring(
                                    f"""
                                    <visual>
                                      <origin rpy="{euler[0]} {euler[1]} {euler[2]}" xyz="{offset[0]} {offset[1]} {offset[2]}"/>
                                      <geometry>
                                        <box size="{size[0]} {size[1]} {size[2]}"/>
                                      </geometry>
                                    </visual>
                                    """
                                )
                                urdf_link.insert(1, visual_mesh)

                            collision_mesh = ET.fromstring(
                                f"""
                                <collision>
                                  <origin rpy="{euler[0]} {euler[1]} {euler[2]}" xyz="{offset[0]} {offset[1]} {offset[2]}"/>
                                  <geometry>
                                    <box size="{size[0]} {size[1]} {size[2]}"/>
                                  </geometry>
                                </collision>
                                """
                            )
                            urdf_link.insert(2, collision_mesh)
                            succesful_write = True

        if succesful_write:
            urdf_path = os.path.join(
                obj_properties["category_path"],
                obj_properties["object_instance"],
                f"{obj_properties['object_instance']}_simplified.urdf",
            )
            with open(urdf_path, "w") as f:
                tree.write(f, encoding="unicode")
