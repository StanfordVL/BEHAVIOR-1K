import glob
import json
import os
import xml.etree.ElementTree as ET

import numpy as np

import igibson
import igibson.utils.transform_utils as T


def insert_geometric_primitive(obj, insert_visual_mesh=True):
    tree = ET.parse(obj["urdf"])
    links = tree.findall("link")
    size = obj["size"]
    euler = obj["euler"]
    offset = obj["base_link_offset"]

    for link in links:
        import pdb

        pdb.set_trace()
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
            link.insert(1, visual_mesh)

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
        link.insert(2, collision_mesh)

    return tree


if __name__ == "__main__":
    object_categories = glob.glob(os.path.join(igibson.ig_dataset_path, "objects/*"))

    idx = 0
    objects_to_process = {}
    for category_path in object_categories:
        print(category_path)
        if os.path.isdir(category_path):
            for object_instance in os.listdir(category_path):
                mvbb_meta = os.path.join(category_path, object_instance, "misc", "mvbb_meta.json")
                metadata_json = os.path.join(category_path, object_instance, "misc", "metadata.json")
                if os.path.exists(mvbb_meta):
                    with open(mvbb_meta, "r") as f:
                        object_data = json.load(f)
                    with open(metadata_json, "r") as f:
                        metadata = json.load(f)
                    size = np.array(object_data["max_vert"]) - np.array(object_data["min_vert"])
                    if object_data["ratio"] >= 0.9:
                        objects_to_process[idx] = {
                            "category": os.path.basename(category_path),
                            "category_path": category_path,
                            "object_instance": object_instance,
                            "urdf": os.path.join(category_path, object_instance, f"{object_instance}.urdf"),
                            "mvbb_meta": mvbb_meta,
                            "base_link_offset": np.array(metadata["base_link_offset"]),
                            "size": size,
                            "euler": T.mat2euler(np.reshape(object_data["transform"], [3, 3])),
                        }
                        idx += 1
    new_urdfs = []
    for obj_name, obj_properties in objects_to_process.items():
        tree = insert_geometric_primitive(obj_properties)
        urdf_path = os.path.join(
            obj_properties["category_path"],
            obj_properties["object_instance"],
            f"{obj_properties['object_instance']}_simplified.urdf",
        )
        new_urdfs.append(urdf_path)
        with open(urdf_path, "w") as f:
            tree.write(f, encoding="unicode")
