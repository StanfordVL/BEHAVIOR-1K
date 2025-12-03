import hashlib
import json
import os
import pathlib
import shutil
import pymxs

rt = pymxs.runtime

import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

from b1k_pipeline.utils import parse_name


def get_maps(root_mat):
    maps = set()

    def _recursively_hash_materials_and_textures(mtl):
        # print("Recursing into", mtl)

        # We can check for submtls for material instances
        if rt.superClassOf(mtl) == rt.Material:
            for i in range(rt.getNumSubMtls(mtl)):
                sub_mtl = rt.getSubMtl(mtl, i + 1)
                if sub_mtl is not None:
                    _recursively_hash_materials_and_textures(sub_mtl)

        # We can check for subtexmaps for texture maps and materials
        if rt.superClassOf(mtl) == rt.textureMap or rt.superClassOf(mtl) == rt.Material:
            found_subs = False
            for i in range(rt.getNumSubTexmaps(mtl)):
                sub_texmap = rt.getSubTexmap(mtl, i + 1)
                if sub_texmap is not None:
                    _recursively_hash_materials_and_textures(sub_texmap)
                    found_subs = True

            # Only track the texture map if it has no subtexmaps
            if rt.superClassOf(mtl) == rt.textureMap and not found_subs:
                maps.add(mtl)

    _recursively_hash_materials_and_textures(root_mat)
    return maps


def fix_maps():
    providers = {
        k.split("-")[-1]: v
        for k, v in json.loads(
            pathlib.Path(
                r"D:\BEHAVIOR-1K\asset_pipeline\artifacts\pipeline\object_inventory.json"
            ).read_text()
        )["providers"].items()
    }
    current_target = "/".join(pathlib.Path(rt.maxFilePath).parts[-2:])
    current_target_bakery = (
        pathlib.Path(r"D:\BEHAVIOR-1K\asset_pipeline\cad") / current_target / "bakery"
    )
    current_target_textures = (
        pathlib.Path(r"D:\BEHAVIOR-1K\asset_pipeline\cad") / current_target / "textures"
    )
    for obj in rt.objects:
        all_texmaps = get_maps(obj.material)

        for texmap in all_texmaps:
            if not hasattr(texmap, "filename") or not texmap.filename:
                print(
                    f"Map {texmap} of type {rt.classOf(texmap)} has no filename. Skipping."
                )
                continue

            # assert (
            #     rt.classOf(texmap) == rt.Bitmaptexture
            # ), f"Material map {texmap} has unexpected type {rt.classOf(texmap)}"

            # Check if the map is already under this target.
            texmap_filename = pathlib.Path(texmap.filename)
            if (
                current_target_bakery in texmap_filename.parents
                or current_target_textures in texmap_filename.parents
            ):
                print(
                    f"Map {texmap_filename} is already under the target {current_target}. Skipping.",
                )
                continue

            # If the map is not under the target, we need to move it.
            # First, check if it's found exactly at the provided path, or some other available path.
            if texmap_filename.exists():
                # If it's available at the provided path, use that.
                orig_map_path = texmap_filename
            elif (
                hasattr(texmap, "bitmap")
                and texmap.bitmap is not None
                and pathlib.Path(texmap.bitmap.filename).exists()
            ):
                # Otherwise, if the path was resolved somehow by 3ds Max, use that.
                orig_map_path = pathlib.Path(texmap.bitmap.filename)
            else:
                # Otherwise, try to resolve in the original provider's directory.
                # Get the original provider for the object
                parsed_name = parse_name(obj.name)
                if not parsed_name:
                    print(
                        f"Could not parse name for object {obj.name}. Skipping map {texmap_filename}."
                    )
                    continue

                model_id = parsed_name.group("model_id")
                original_provider = providers.get(model_id, None)
                if not original_provider:
                    print(
                        f"Could not find original provider for model ID {model_id}. Skipping map {texmap_filename}."
                    )
                    continue

                # Use os.path.abspath which normalizes + absolutifies the paths but does not resolve symlinks unlike pathlib (problem with dvc)
                map_filename = texmap_filename.name
                map_directory = (
                    "bakery" if "bakery" in texmap_filename.parts else "textures"
                )
                orig_map_path = (
                    pathlib.Path(r"D:\BEHAVIOR-1K\asset_pipeline\cad")
                    / original_provider
                    / map_directory
                    / map_filename
                )
            if not orig_map_path.exists():
                print(
                    f"Map {texmap_filename} does not exist at {orig_map_path}. Skipping.",
                )
                continue

            # Compute the hash of the map file
            with open(orig_map_path, "rb") as f:
                map_hash = hashlib.md5(f.read()).hexdigest()

            # Generate a brand new path for the map file. Prefix the filename with the hash if it's not in the bakery directory.
            new_map_directory = (
                "bakery" if "bakery" in orig_map_path.parts else "textures"
            )
            new_map_filename = (
                f"{map_hash}_{orig_map_path.name}"
                if new_map_directory == "textures"
                else orig_map_path.name
            )
            new_map_path = (
                pathlib.Path(r"D:\BEHAVIOR-1K\asset_pipeline\cad")
                / current_target
                / new_map_directory
                / new_map_filename
            )

            if not new_map_path.exists():
                print("Copying map from", orig_map_path, "to", new_map_path)
                shutil.copyfile(orig_map_path, new_map_path)

            # Then update the path in the bitmap texture
            texmap.filename = str(new_map_path)


if __name__ == "__main__":
    fix_maps()
