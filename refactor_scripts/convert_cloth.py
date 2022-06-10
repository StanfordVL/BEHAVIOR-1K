# from omni.isaac.kit import SimulationApp
#
# app = SimulationApp({"headless": False})
#
# from omni.isaac.core import World as Simulator

import os

from igibson import app, ig_dataset_path
import trimesh
import numpy as np
import shutil
from import_urdfs_from_scene import import_obj_urdf
from import_metadata import import_obj_metadata
from omni.isaac.core.utils.stage import open_stage, get_current_stage
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omni.kit.commands import execute


CLOTH_CATEGORIES = ["carpet"]
THRESH = 0.05

for category in CLOTH_CATEGORIES:
    category_dir = os.path.join(ig_dataset_path, "objects", category)
    for model in os.listdir(category_dir):

        model_dir = os.path.join(category_dir, model)
        visual_mesh_dir = os.path.join(model_dir, "shape", "visual")
        obj_files = [fname for fname in os.listdir(visual_mesh_dir) if fname.endswith(".obj")]
        if len(obj_files) > 1:
            print(f"WARNING: there are multiple visual meshes in {visual_mesh_dir}")
            continue

        # Get the single visual mesh OBJ file
        original_obj_file = os.path.join(visual_mesh_dir, obj_files[0])

        # Sub-divide to enough resolution
        mesh = trimesh.load(original_obj_file)
        while True:
            # face_indices_from_area = np.where(mesh.area_faces > AREA_THRESH)[0]
            edge_indices = np.where(mesh.edges_unique_length > THRESH)[0]

            # Faces that have any edges longer than the threshold
            face_indices_from_edge_length = np.where([len(np.intersect1d(face, edge_indices)) != 0 for face in mesh.faces_unique_edges])[0]
            # face_indices = np.union1d(face_indices_from_area, face_indices_from_edge_length)
            face_indices = face_indices_from_edge_length
            if len(face_indices) == 0:
                break

            mesh = mesh.subdivide(face_indices)

        print(f"Final face count: {len(mesh.faces)}")

        # Somehow we need to manually write the vertex normals to cache
        mesh._cache.cache["vertex_normals"] = mesh.vertex_normals

        # Export the new OBJ file into a tmp directory
        tmp_visual_mesh_dir = os.path.join(model_dir, "shape", "visual_tmp")
        os.makedirs(tmp_visual_mesh_dir, exist_ok=True)
        new_obj_file = os.path.join(tmp_visual_mesh_dir, "test.obj")
        mesh.export(new_obj_file, file_type="obj")

        # Copy the original visual mesh OBJ file to a tmp file
        tmp_original_obj_file = original_obj_file[:-4] + "_original.obj"
        shutil.move(original_obj_file, tmp_original_obj_file)

        # Create the new visual mesh that uses the original mtl
        with open(original_obj_file, "w+") as dst:
            with open(new_obj_file, "r") as src:
                for line in src.readlines():
                    line = line.replace("mtllib material_0.mtl", "mtllib default.mtl")
                    line = line.replace("usemtl material_0", "usemtl default")
                    dst.write(line)

        usd_file = os.path.join(category_dir, model, "usd", f"{model}.usd")
        rigid_usd_file = os.path.join(category_dir, model, "usd", f"{model}_rigid.usd")
        cloth_usd_file = os.path.join(category_dir, model, "usd", f"{model}_cloth.usd")

        # Rename the original rigid usd if exists
        if os.path.isfile(usd_file):
            shutil.move(usd_file, rigid_usd_file)

        # Use the import_obj_urdf to output to usd_file
        import_obj_urdf(category, model)
        import_obj_metadata(obj_category=category, obj_model=model, import_render_channels=True)

        # Rename it to the cloth version
        shutil.move(usd_file, cloth_usd_file)

        # Rename back model.usd if exists
        if os.path.isfile(rigid_usd_file):
            shutil.move(rigid_usd_file, usd_file)

        # Rename back original visual mesh OBJ file
        shutil.move(tmp_original_obj_file, original_obj_file)
        # Remove tmp directory
        shutil.rmtree(tmp_visual_mesh_dir)

        # Load the newly created cloth usd and simplify the hierarchy
        assert open_stage(cloth_usd_file)
        stage = get_current_stage()
        prim = stage.GetDefaultPrim()

        links = [child for child in prim.GetChildren() if child.GetTypeName() == "Xform"]
        assert len(links) == 1, f"[{cloth_usd_file}] has more than one link"
        link = links[0]

        tmp_base_link_path = prim.GetPrimPath().AppendPath("base_link_backup")
        execute("MovePrim", path_from=link.GetPrimPath(), path_to=tmp_base_link_path)

        link = get_prim_at_path(tmp_base_link_path)

        visual_mesh_path = tmp_base_link_path.AppendPath("visuals")
        assert is_prim_path_valid(visual_mesh_path), f"WARNING: visual mesh path [{visual_mesh_path}] does not exist"

        visual_mesh = get_prim_at_path(visual_mesh_path)
        assert visual_mesh.GetTypeName() == "Mesh", \
            f"WARNING: visual mesh path [{visual_mesh_path}] does not have type Mesh " \
            f"(likely because there are multiple visual meshes)."

        execute("MovePrim", path_from=visual_mesh.GetPrimPath(), path_to=prim.GetPrimPath().AppendPath("base_link"))

        stage.RemovePrim(tmp_base_link_path)
        stage.Save()

        print(f"Done: {cloth_usd_file}")

app.close()
