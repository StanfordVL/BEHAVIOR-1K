from ast import Dict
import sys

import numpy as np
import trimesh
from b1k_pipeline.urdfpy import URDF
import os


def main(model_dir):
    # Load the URDF file into urdfpy
    model_id = os.path.basename(model_dir)
    urdf_filename = f"{model_id}.urdf"
    urdf_path = os.path.join(model_dir, "urdf", urdf_filename)
    robot = URDF.load(urdf_path)

    # Do FK with everything at the lower position
    joint_cfg = {
        joint.name: joint.limit.lower
        for joint in robot.joints
        if joint.joint_type in ("prismatic", "revolute")
    }
    cfk: Dict[trimesh.Trimesh, np.ndarray] = robot.collision_trimesh_fk(cfg=joint_cfg)

    meshes = []
    for mesh, transform in cfk.items():
        mesh_copy = mesh.copy()
        mesh_copy.apply_transform(transform)
        meshes.append(mesh_copy)
    final_mesh = trimesh.boolean.union(meshes, engine="manifold")
    final_mesh.show()
    final_mesh.export(os.path.join(model_dir, f"{model_id}.stl"))


if __name__ == "__main__":
    main(sys.argv[1])
