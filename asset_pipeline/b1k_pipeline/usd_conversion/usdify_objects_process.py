"""
Script to import scene and objects
"""

import glob
import os
import pathlib
import shutil
import subprocess
import sys

from omnigibson.macros import gm

# Set some macros. Is this kosher?
gm.HEADLESS = True
gm.ENABLE_FLATCACHE = False
gm.USE_GPU_DYNAMICS = True
gm.USE_ENCRYPTED_ASSETS = True
gm.FORCE_LIGHT_INTENSITY = None
gm.ENABLE_TRANSITION_RULES = False

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.prims import ClothPrim
from omnigibson.scenes import Scene
from omnigibson.utils.asset_utils import encrypt_file
from omnigibson.utils.asset_conversion_utils import (
    import_obj_metadata,
    convert_urdf_to_usd,
)
from bddl.object_taxonomy import ObjectTaxonomy

if __name__ == "__main__":
    og.launch()

    # Here we need to save the kwargs for when we clear later. This is because the
    # URDF importer will open a new stage and thus make the SimulationContext unusable,
    # so og.clear()'s default implementation of getting the kwargs from the existing
    # context will not work.
    clear_kwargs = dict(
        gravity=og.sim.gravity,
        physics_dt=og.sim.get_physics_dt(),
        rendering_dt=og.sim.get_rendering_dt(),
        sim_step_dt=og.sim.get_sim_step_dt(),
        viewer_width=og.sim.viewer_width,
        viewer_height=og.sim.viewer_height,
        device=og.sim.device,
    )

    ot = ObjectTaxonomy()

    dataset_root = str(pathlib.Path(sys.argv[1]))
    batch = sys.argv[2:]
    for path in batch:
        obj_category, obj_model = pathlib.Path(path).parts[-2:]
        model_dir = pathlib.Path(dataset_root) / "objects" / obj_category / obj_model
        assert model_dir.exists()
        print(f"IMPORTING CATEGORY/MODEL {obj_category}/{obj_model}...")
        convert_urdf_to_usd(
            urdf_path=str(model_dir / "urdf" / f"{obj_model}.urdf"),
            obj_category=obj_category,
            obj_model=obj_model,
            dataset_root=dataset_root,
        )
        print("Importing metadata")
        usd_path = str(model_dir / "usd" / f"{obj_model}.usd")
        import_obj_metadata(
            usd_path=usd_path,
            obj_category=obj_category,
            obj_model=obj_model,
            dataset_root=dataset_root,
            force_asset_pipeline_materials=True,
        )
        print("Done importing metadata")

        obj_synset = ot.get_synset_from_category_or_substance(obj_category)
        assert (
            obj_synset is not None
        ), f"Could not find synset for category {obj_category}"
        if "cloth" in ot.get_abilities(obj_synset):
            og.clear(**clear_kwargs)
            empty_scene = Scene()
            og.sim.import_scene(empty_scene)

            # Prepare to simulate the object by creating a reference
            # to the object in the scene.
            stage = lazy.omni.isaac.core.utils.stage.get_current_stage()
            prim = stage.DefinePrim("/World/scene_0/cloth", "Mesh")
            cloth_prim_path_in_usd = f"/{obj_model}/base_link/visuals"
            assert prim.GetReferences().AddReference(
                usd_path, cloth_prim_path_in_usd
            ), "Failed to add reference to cloth"

            # Wrap it in a cloth prim and generate some configurations.
            cloth_prim = ClothPrim(
                relative_prim_path="/cloth",
                name="cloth",
                load_config=dict(force_remesh=True),
            )
            cloth_prim.load(empty_scene)

            og.sim.play()
            cloth_prim.generate_settled_configuration()
            cloth_prim.generate_folded_configuration()
            cloth_prim.generate_crumpled_configuration()
            cloth_prim.reset_points_to_configuration("default")

            # Get all of the important attributes
            attribs_to_save = {
                "points",
                "faceVertexCounts",
                "faceVertexIndices",
                "normals",
                "primvars:st",
                "points_default",
                "points_settled",
                "points_folded",
                "points_crumpled",
            }
            attrib_types_and_values = {}
            for attrib_name in attribs_to_save:
                attrib = cloth_prim.prim.GetAttribute(attrib_name)
                attrib_types_and_values[attrib_name] = (
                    attrib.GetTypeName(),
                    attrib.Get(),
                )

            # Clear the simulation again to remove the reference
            og.clear(**clear_kwargs)

            # Open the USD file and add the attributes
            cloth_stage = lazy.pxr.Usd.Stage.Open(usd_path)
            prim = cloth_stage.GetPrimAtPath(cloth_prim_path_in_usd)
            for attrib_name, (
                attrib_type,
                attrib_value,
            ) in attrib_types_and_values.items():
                attrib = (
                    prim.GetAttribute(attrib_name)
                    if prim.HasAttribute(attrib_name)
                    else prim.CreateAttribute(attrib_name, attrib_type)
                )
                attrib.Set(attrib_value)
            cloth_stage.Save()

        # Encrypt the output files.
        print("Encrypting")
        for usd_path in glob.glob(
            os.path.join(
                dataset_root, "objects", obj_category, obj_model, "usd", "*.usd"
            )
        ):
            encrypted_usd_path = usd_path.replace(".usd", ".encrypted.usd")
            encrypt_file(usd_path, encrypted_filename=encrypted_usd_path)
            os.remove(usd_path)
        print("Done encrypting")

    og.shutdown()
