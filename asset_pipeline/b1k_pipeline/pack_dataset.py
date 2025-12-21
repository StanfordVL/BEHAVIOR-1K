import glob
import json
import logging
import os
import pathlib
import shutil
import traceback
import fs
import fs.copy
import fs.zipfs
import fs.multifs
import fs.osfs
import tqdm

import b1k_pipeline.utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

OUTPUT_FILENAME = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "artifacts",
    "pipeline",
    "pack_dataset.json",
)
SUCCESS_FILENAME = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "artifacts",
    "pipeline",
    "pack_dataset.success",
)
IN_FILENAME_AGGREGATE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "artifacts", "aggregate"
)
PARALLELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "artifacts", "parallels"
)
OUT_FILENAME = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "artifacts", "og_dataset.zip"
)
DEMO_OUT_FILENAME = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "artifacts", "og_dataset_demo.zip"
)
VERSION_FILENAME = os.path.join(os.path.dirname(os.path.dirname(__file__)), "VERSION")
PARALLELS = [
    "objects.zip",
    "objects_usd.zip",
    "metadata.zip",
    "scenes_json.zip",
    "systems.zip",
]


def main():
    success = True
    error_msg = ""
    try:
        # Get a multi-FS view over all of the parallel filesystems.
        multi_fs = fs.multifs.MultiFS()
        # multi_fs.add_fs('aggregate', fs.osfs.OSFS(IN_FILENAME_AGGREGATE), priority=0)
        for parallel_zip_name in PARALLELS:
            parallel_zip = os.path.join(PARALLELS_DIR, parallel_zip_name)
            print("Adding", parallel_zip)
            multi_fs.add_fs(
                os.path.basename(parallel_zip), fs.zipfs.ZipFS(parallel_zip), priority=1
            )

        # Copy all the files to the output zip filesystem.
        print("Copying files")
        total_files = sum(1 for f in multi_fs.walk.files())
        with b1k_pipeline.utils.WriteOnly7ZipFS(OUT_FILENAME) as out_fs:
            with tqdm.tqdm(total=total_files) as pbar:
                fs.copy.copy_fs(multi_fs, out_fs, on_copy=lambda *args: pbar.update(1))

            # Rename any WIP scenes
            print("Renaming WIP scenes")
            if out_fs.exists("scenes"):
                scenes_dir = out_fs.opendir("scenes")
                for scene_dir in list(scenes_dir.listdir("/")):
                    if "scenes/" + scene_dir in b1k_pipeline.utils.get_targets(
                        "verified_scenes"
                    ):
                        continue

                    # First rename the main dir
                    renamed_dir = "WIP_" + scene_dir
                    print("Renaming", scene_dir, "to", renamed_dir)
                    scenes_dir.movedir(scene_dir, renamed_dir, create=True)

                    # Then rename the JSONs
                    json_dir = scenes_dir.opendir(renamed_dir).opendir("json")
                    for json_file in list(json_dir.listdir("/")):
                        json_dir.move(json_file, "WIP_" + json_file)

            # Nuke any systems' directories in the objects directory
            print("Removing object dirs of systems")
            objects_dir = out_fs.opendir("objects")
            systems_dir = out_fs.opendir("systems")
            for system_dir in list(systems_dir.listdir("/")):
                if objects_dir.exists(system_dir):
                    print("Removing", system_dir)
                    objects_dir.removetree(system_dir)

            # Delete the URDF and shape directories since they contain unencrypted objects
            print("Removing URDF directories")
            urdf_dirs = {x.path for x in out_fs.glob("*/*/*/urdf/")} | {
                x.path for x in out_fs.glob("*/*/*/shape/")
            }
            for urdf_dir in urdf_dirs:
                out_fs.removetree(urdf_dir)

            # Add the VERSION file
            out_fs.writetext("VERSION", pathlib.Path(VERSION_FILENAME).read_text())

            # Now create the demo zip
            # with fs.zipfs.ZipFS(DEMO_OUT_FILENAME, write=True) as demo_out_fs:
            #     # Copy over the metadata directory
            #     fs.copy.copy_fs(out_fs.opendir("metadata"), demo_out_fs.makedirs("metadata"))

            #     # Copy over the Rs_int scene directory
            #     fs.copy.copy_fs(out_fs.opendir("scenes/Rs_int"), demo_out_fs.makedirs("scenes/Rs_int"))

            #     # Copy over the water system directory
            #     fs.copy.copy_fs(out_fs.opendir("systems/water"), demo_out_fs.makedirs("systems/water"))

            #     # Copy over the object directories of ALL objects that are needed for Rs_int
            #     rs_int_object_list = json.loads(b1k_pipeline.utils.PipelineFS().target_output("scenes/Rs_int").readtext("object_list.json"))
            #     rs_int_needed_objects = {obj.split("-")[1] for obj in rs_int_object_list["needed_objects"]}
            #     objects_dir = out_fs.opendir("objects")
            #     for cat in objects_dir.listdir("/"):
            #         cat_dir = objects_dir.opendir(cat)
            #         for mdl in cat_dir.listdir("/"):
            #             mdl_dir = cat_dir.opendir(mdl)

            #             if mdl in rs_int_needed_objects:
            #                 fs.copy.copy_fs(mdl_dir, demo_out_fs.makedirs(f"objects/{cat}/{mdl}", recreate=True))

    except Exception as e:
        success = False
        error_msg = traceback.format_exc()

    with open(OUTPUT_FILENAME, "w") as f:
        json.dump({"success": success, "error_msg": error_msg}, f, indent=4)

    if success:
        with open(SUCCESS_FILENAME, "w") as f:
            pass


if __name__ == "__main__":
    main()
