import json
import pathlib

from concurrent.futures import as_completed
import fs.copy
import fs.path
from fs.tempfs import TempFS
import tqdm

from b1k_pipeline.utils import (
    ParallelZipFS,
    PipelineFS,
    TMP_DIR,
    launch_cluster,
    og_context,
    submit_og_task,
)

WORKER_COUNT = 6
MAX_ATTEMPTS = 3

OG_MACROS = {
    "HEADLESS": True,
    "USE_GPU_DYNAMICS": True,
    "USE_ENCRYPTED_ASSETS": True,
    "FORCE_LIGHT_INTENSITY": None,
    "ENABLE_TRANSITION_RULES": False,
}


def process_obj(dataset_root, obj_path):
    """Convert one URDF object to an encrypted USD on a worker."""
    import glob
    import os

    import omnigibson as og
    import omnigibson.lazy as lazy
    from omnigibson.prims import ClothPrim
    from omnigibson.scenes import Scene
    from omnigibson.utils.asset_utils import encrypt_file
    from omnigibson.utils.asset_conversion_utils import (
        import_obj_metadata,
        convert_urdf_to_usd,
    )
    from bddl.knowledge_base import KnowledgeBase

    ctx = og_context()
    clear_kwargs = ctx.clear_kwargs
    if "kb" not in ctx.cache:
        ctx.cache["kb"] = KnowledgeBase(populate=True)
    kb = ctx.cache["kb"]
    dataset_root = str(pathlib.Path(dataset_root))

    obj_category, obj_model = pathlib.Path(obj_path).parts[-2:]
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

    cat = kb.get_category(obj_category)
    ps = kb.get_particle_system(obj_category)
    obj_synset = cat.synset if cat else (ps.synset if ps else None)
    assert (
        obj_synset is not None
    ), f"Could not find synset for category {obj_category}"
    if "cloth" in obj_synset.abilities and False:
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


def main():
    failed_objects = set()
    with (
        PipelineFS() as pipeline_fs,
        ParallelZipFS("objects.zip") as objects_fs,
        TempFS(temp_dir=str(TMP_DIR)) as dataset_fs,
    ):
        with ParallelZipFS("objects_usd.zip", write=True) as out_fs:
            # Copy everything over to the dataset FS
            print("Copying input to dataset fs...")
            objdir_glob = list(objects_fs.glob("objects/*/*/"))
            for item in tqdm.tqdm(objdir_glob):
                if (
                    objects_fs.opendir(item.path)
                    .opendir("urdf")
                    .glob("*.urdf")
                    .count()
                    .files
                    == 0
                ):
                    continue
                fs.copy.copy_fs(
                    objects_fs.opendir(item.path),
                    dataset_fs.makedirs(item.path, recreate=True),
                )

            print("Launching cluster...")
            executor = launch_cluster(WORKER_COUNT, og_macros=OG_MACROS)

            object_glob = [x.path for x in dataset_fs.glob("objects/*/*/")]
            print("Total count: ", len(object_glob))

            # Workers can segfault inside Isaac Sim's RTX plugin after
            # a handful of og.clear cycles, so retry each object up to
            # MAX_ATTEMPTS times (loky respawns the worker each time).
            remaining = list(object_glob)
            for attempt in range(1, MAX_ATTEMPTS + 1):
                if not remaining:
                    break
                print(
                    f"Attempt {attempt}/{MAX_ATTEMPTS}: queueing {len(remaining)} objects."
                )
                futures = {
                    submit_og_task(
                        executor,
                        process_obj,
                        dataset_fs.getsyspath("/"),
                        obj_path,
                    ): obj_path
                    for obj_path in remaining
                }

                next_remaining = []
                for future in tqdm.tqdm(
                    as_completed(futures.keys()), total=len(futures)
                ):
                    obj_path = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Object {obj_path} attempt {attempt} raised: {e}")

                    obj_dir = dataset_fs.opendir(obj_path)
                    if obj_dir.glob("usd/*.encrypted.usd").count().files != 1:
                        next_remaining.append(obj_path)
                        if obj_dir.exists("usd"):
                            obj_dir.removetree("usd")

                remaining = next_remaining
                print(
                    f"After attempt {attempt}: {len(remaining)} objects still failing."
                )

            failed_objects.update(remaining)

            # Move the USDs to the output FS
            print("Copying USDs to output FS...")
            usd_glob = [
                x.path for x in dataset_fs.glob("objects/*/*/usd/*.encrypted.usd")
            ]
            for item in tqdm.tqdm(usd_glob):
                itemdir = fs.path.dirname(item)
                fs.copy.copy_fs(dataset_fs.opendir(itemdir), out_fs.makedirs(itemdir))

            print("Done processing. Archiving things now.")

        # Save the logs
        with pipeline_fs.pipeline_output().open("usdify_objects.json", "w") as f:
            json.dump(
                {
                    "success": len(failed_objects) == 0,
                    "failed_objects": sorted(failed_objects),
                },
                f,
            )


if __name__ == "__main__":
    main()
