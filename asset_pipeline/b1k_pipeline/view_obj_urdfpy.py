from ast import Dict
import sys

import numpy as np
import trimesh
from b1k_pipeline.urdfpy import URDF
import json
import os


def get_cube(limits=None):
    """get the vertices, edges, and faces of a cuboid defined by its limits

    limits = np.array([[x_min, x_max],
                       [y_min, y_max],
                       [z_min, z_max]])
    """
    v = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=int,
    )

    if limits is not None:
        v = limits[np.arange(3)[np.newaxis, :].repeat(8, axis=0), v]

    e = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [1, 3],
            [1, 5],
            [2, 3],
            [2, 6],
            [3, 7],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
        ],
        dtype=int,
    )

    f = np.array(
        [
            [0, 2, 3, 1],
            [0, 4, 5, 1],
            [0, 4, 6, 2],
            [1, 5, 7, 3],
            [2, 6, 7, 3],
            [4, 6, 7, 5],
        ],
        dtype=int,
    )

    return v, e, f


def main(model_dir):
    # Load the URDF file into urdfpy
    model_id = os.path.basename(model_dir)
    urdf_filename = f"{model_id}.urdf"
    urdf_path = os.path.join(model_dir, "urdf", urdf_filename)
    robot = URDF.load(urdf_path)

    # Load the metadata file too
    metadata_path = os.path.join(model_dir, "misc", "metadata.json")
    with open(metadata_path, "r") as metadata_file:
        metadata = json.load(metadata_file)

    # Do FK with everything at the lower position
    joint_cfg = {
        joint.name: joint.limit.lower
        for joint in robot.joints
        if joint.joint_type in ("prismatic", "revolute")
    }
    vfk: Dict[trimesh.Trimesh, np.ndarray] = robot.visual_trimesh_fk(cfg=joint_cfg)

    meshes = []
    for mesh, transform in vfk.items():
        mesh_copy = mesh.copy()
        mesh_copy.apply_transform(transform)
        meshes.append(mesh_copy)
    final_mesh = trimesh.util.concatenate(meshes)
    final_mesh.show()
    os.makedirs(f"export/{model_id}", exist_ok=True)
    final_mesh.export("export/{model_id}/export.obj")

    # sphere = trimesh.creation.uv_sphere(radius=0.01)
    # bbox_size = np.array(metadata["bbox_size"])
    # print("bb size", bbox_size)
    # bbox_ctr = np.array(metadata["base_link_offset"])
    # print("bb ctr", bbox_ctr)
    # bbox_min = bbox_ctr - bbox_size / 2
    # bbox_max = bbox_ctr + bbox_size / 2
    # limits = np.stack([bbox_min, bbox_max], axis=1)
    # v, e, f = get_cube(limits)
    # for point in v:
    #     p_transform = trimesh.transformations.translation_matrix(point)
    #     scene.add_geometry(geometry=sphere, transform=p_transform)
    # scene.show()


if __name__ == "__main__":
    main(sys.argv[1])
