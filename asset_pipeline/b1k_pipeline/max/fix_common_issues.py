import csv
import hashlib
import json
import os
import shutil
import sys

import numpy as np

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import glob
import pathlib
import random
import re
import string
from collections import defaultdict

import pymxs
import tqdm

import b1k_pipeline.utils
import b1k_pipeline.max.apply_qa_fixes
import b1k_pipeline.max.import_fillable_meshes
import b1k_pipeline.max.extract_particle_objects
import b1k_pipeline.max.extract_school_objects
import b1k_pipeline.max.prebake_textures
import b1k_pipeline.max.replace_bad_object
import b1k_pipeline.max.collision_vertex_reduction
import b1k_pipeline.max.convex_decomposition
import b1k_pipeline.max.match_links
from b1k_pipeline.max.merge_collision import merge_collision

rt = pymxs.runtime

PASS_FILENAME = "done-vrayconversionagain.success"
VRAY_LOG_FILENAME = pathlib.Path(r"D:/BEHAVIOR-1K/asset_pipeline/mtlconvert.log")
RENDER_PRESET_FILENAME = str(
    (b1k_pipeline.utils.PIPELINE_ROOT / "render_presets" / "objrender.rps").absolute()
)

from bddl.object_taxonomy import ObjectTaxonomy

OBJECT_TAXONOMY = ObjectTaxonomy()


def get_approved_room_types():
    approved = []
    with open(
        b1k_pipeline.utils.PIPELINE_ROOT / "metadata/allowed_room_types.csv", newline=""
    ) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            approved.append(row[0])

    return set(approved)


APPROVED_ROOM_TYPES = get_approved_room_types()


def get_renames():
    with open(
        b1k_pipeline.utils.PIPELINE_ROOT / "metadata/object_renames.csv", "r"
    ) as f:
        reader = csv.DictReader(f)
        renames = {}
        for row in reader:
            obj_id = row["ID (auto)"]
            old_cat = row["Original category (auto)"]
            new_cat = row["New Category"]
            in_name = f"{old_cat}-{obj_id}"
            out_name = f"{new_cat}-{obj_id}"
            renames[obj_id] = (in_name, out_name)

        return renames


RENAMES = get_renames()


def get_deletions():
    with open(
        b1k_pipeline.utils.PIPELINE_ROOT / "metadata/deletion_queue.csv", "r"
    ) as f:
        reader = csv.DictReader(f)
        deletions = set()
        for row in reader:
            obj_id = row["ID (auto)"]
            deletions.add(obj_id)

        return deletions


DELETIONS = get_deletions()


def fix_layers(filename) -> bool:
    # If this is an object file, remove all layers
    if filename.parts[-3] == "objects":
        zero_layer = rt.LayerManager.getLayer(0)
        zero_layer.current = True
        for obj in rt.objects:
            zero_layer.addNode(obj)

        layers_to_remove = []
        for layer_id in range(1, rt.LayerManager.count):
            layers_to_remove.append(rt.LayerManager.getLayer(layer_id).name)

        for layer_name in layers_to_remove:
            rt.LayerManager.deleteLayerByName(layer_name)

    else:
        # Load and apply room assignments.
        # First, load from the file.
        old_room_assignment_file = (
            pathlib.Path(r"D:\rooms") / f"{filename.parts[-2]}.json"
        )
        old_room_assignments = json.loads(old_room_assignment_file.read_text())

        # Unfold the assignments by model ID
        model_centers = defaultdict(list)
        model_rooms = defaultdict(list)
        for name, room, bbmin, bbmax in old_room_assignments:
            m = b1k_pipeline.utils.parse_name(name)
            if not m:
                continue
            model_id = m.group("model_id")
            center = (np.array(bbmin) + np.array(bbmax)) / 2
            model_centers[model_id].append(center)
            model_rooms[model_id].append(room)
        model_centers = {k: np.array(v) for k, v in model_centers.items()}

        # Then, for each object instance base link in the scene, get the nearest room assignment
        rooms_by_model_and_instance = {}
        for obj in rt.objects:
            if rt.classOf(obj) != rt.Editable_Poly:
                continue
            m = b1k_pipeline.utils.parse_name(obj.name)
            if not m:
                continue
            model_id = m.group("model_id")
            instance_id = m.group("instance_id")
            link_name = m.group("link_name") or "base_link"
            if link_name != "base_link":
                continue
            if m.group("meta_type"):
                continue

            bbox_min, bbox_max = rt.NodeGetBoundingBox(obj, rt.Matrix3(1))
            center = (np.array(bbox_min) + np.array(bbox_max)) / 2

            # Get the nearest room assignment (and make sure it's less than 1m away)
            if model_id not in model_centers:
                continue
            all_centers = model_centers[model_id]
            all_rooms = model_rooms[model_id]
            nearest_center_idx = np.argmin(np.linalg.norm(all_centers - center, axis=1))
            nearest_center = all_centers[nearest_center_idx]
            nearest_room = all_rooms[nearest_center_idx]
            if np.linalg.norm(nearest_center - center) > 1000:
                continue
            rooms_by_model_and_instance[(model_id, instance_id)] = nearest_room

        # Then finally apply the room assignments
        for obj in rt.objects:
            m = b1k_pipeline.utils.parse_name(obj.name)
            if not m:
                continue
            model_id = m.group("model_id")
            instance_id = m.group("instance_id")
            if (model_id, instance_id) not in rooms_by_model_and_instance:
                continue
            room = rooms_by_model_and_instance[(model_id, instance_id)]
            layer = rt.LayerManager.getLayerFromName(
                room
            ) or rt.LayerManager.newLayerFromName(room)
            layer.addNode(obj)
            print("Reassigned", obj.name, "to", room)

    return True


def remove_root_level_meta_links(filename) -> bool:
    to_delete = []
    for obj in rt.objects:
        if obj.parent is not None:
            continue

        match = b1k_pipeline.utils.parse_name(obj.name)
        if match and match.group("meta_type"):
            to_delete.append(obj)

    if not to_delete:
        return False

    for obj in to_delete:
        rt.delete(obj)

    return True


def apply_deletions(filename):
    # No deletions on scenes.
    if "scenes" in str(filename):
        return False

    # Get the particle deletions
    particle_deletions = {
        mid
        for mids in b1k_pipeline.max.extract_particle_objects.MERGES_BY_TARGET.values()
        for mid in mids
    }

    to_delete = []
    for obj in rt.objects:
        match = b1k_pipeline.utils.parse_name(obj.name)
        if not match:
            continue

        model_id = match.group("model_id")
        if model_id in DELETIONS or (
            "substances-01" not in filename.parts and model_id in particle_deletions
        ):
            to_delete.append(obj)

    if not to_delete:
        return False

    for obj in to_delete:
        rt.delete(obj)

    return True


def apply_renames(filename):
    made_any_changes = False
    for obj in rt.objects:
        match = b1k_pipeline.utils.parse_name(obj.name)
        if not match:
            continue

        category = match.group("category")
        model_id = match.group("model_id")
        rename_key = f"{category}-{model_id}"
        if model_id in RENAMES:
            old_key, new_key = RENAMES[model_id]
            if rename_key == old_key:
                obj.name = obj.name.replace(old_key, new_key)
                made_any_changes = True

    return made_any_changes


def update_visibilities():
    # Exit isolate mode
    rt.IsolateSelection.ExitIsolateSelectionMode()

    # Unhide everything
    for obj in rt.objects:
        obj.isHidden = False

    # # Hide collision meshes
    for obj in rt.objects:
        if (
            "Mcollision" in obj.name
            or "Mfillable" in obj.name
            or "Mopenfillable" in obj.name
        ):
            obj.isHidden = True


def match_instance_materials():
    objs_by_base = defaultdict(list)
    for obj in rt.objects:
        if rt.classOf(obj) != rt.Editable_Poly:
            continue
        pn = b1k_pipeline.utils.parse_name(obj.name)
        if not pn:
            continue
        objs_by_base[obj.baseObject].append(obj)

    for base, objs in objs_by_base.items():
        if len(objs) < 2:
            continue
        if len({obj.material for obj in objs}) == 1:
            continue

        # Pick the good material instance
        good_instance = None
        for obj in objs:
            pn = b1k_pipeline.utils.parse_name(obj.name)
            if not pn:
                continue
            if pn.group("joint_side") == "upper":
                continue
            if pn.group("instance_id") != "0":
                continue
            assert not good_instance, f"Multiple good instances: {good_instance}, {obj}"
            good_instance = obj

        # Assign it to all the instances
        for obj in objs:
            obj.material = good_instance.material

        print("Fixed material for instances of", good_instance.name)


def remove_nested_shell_materials(filename, keep_top_level=True):
    def _replace_shell(mat):
        # If this is a multi material, recurse through its submaterials
        if rt.classOf(mat) == rt.MultiMaterial:
            for i in range(len(mat.materialList)):
                mat.materialList[i] = _replace_shell(mat.materialList[i])

        # If it's a shell material recurse down the original side
        if rt.classOf(mat) == rt.Shell_Material:
            mat.originalMaterial = _replace_shell(mat.originalMaterial)

        # If this is a shell material, return the unbaked material
        if rt.classOf(mat) == rt.Shell_Material:
            return mat.originalMaterial

        # Otherwise just return this material
        return mat

    for obj in rt.objects:
        # Note that we don't assign the return value here, in effect keeping the top-level
        # material and only replacing the nested ones.
        tmp_mtl = _replace_shell(obj.material)
        if not keep_top_level:
            obj.material = tmp_mtl

    # Save here to clear material library
    rt.saveMaxFile(str(filename))


def remove_unnecessary_lights():
    for light in list(rt.lights):
        match = b1k_pipeline.utils.parse_name(light.name)
        if not match:
            continue
        if match.group("bad") or match.group("instance_id") != "0":
            rt.delete(light)
            made_any_changes = True
    return made_any_changes


def merge_preexisting_fillable_meshes():
    processed_fillable = set()
    while True:
        for obj in list(rt.objects):
            if not rt.classOf(obj) == rt.Editable_Poly:
                continue

            if obj in processed_fillable:
                continue

            fillable_meshes = []
            for child in obj.children:
                if "Mfillable" in child.name:
                    fillable_meshes.append(child)

            if len(fillable_meshes) < 1:
                continue

            print("Processing", obj.name, "with fillable meshes", fillable_meshes)
            new_fillable = merge_collision(fillable_meshes, obj)
            new_fillable.name = re.sub(
                r"(-M([a-z]+)(?:_([A-Za-z0-9]+))?(?:_([0-9]+))?)",
                "-Mfillable",
                new_fillable.name,
            )
            made_any_changes = True

            processed_fillable.add(obj)

            # Break out of the for loop so that the iteration list restarts.
            # Otherwise, we might get a RuntimeError due to the list changing size.
            break
        else:
            # Break out of the while loop
            break
    return made_any_changes


def run_bad_object_replacement_for_legacy_scenes(filename):
    target_name = filename.parts[-2]
    if target_name.endswith("_int") or target_name.endswith("_garden"):
        comparison_data = b1k_pipeline.max.replace_bad_object.replace_all_bad_legacy_objects_in_open_file()
        print(f"Replaced {len(comparison_data)} bad objects in {filename}")
        with open(
            filename.parent / "artifacts" / "replaced_bad_objects.json", "w"
        ) as f:
            json.dump(comparison_data, f)


def convert_legacy_uv_unwrap_attribute_to_new():
    for obj in rt.objects:
        # Get the attribute that's currently on there
        saved_hash = b1k_pipeline.max.prebake_textures.get_recorded_uv_unwrapping_hash(
            obj
        )
        if saved_hash is None:
            continue

        # Otherwise generate the new hash and save it
        hash_digest = b1k_pipeline.max.prebake_textures.hash_object(obj)

        # Add the new attr
        b1k_pipeline.max.prebake_textures.set_recorded_uv_unwrapping_hash(
            obj, hash_digest
        )


def delete_parts_from_nonzero_instances():
    for obj in rt.objects:
        if not obj.parent:
            continue
        tags = {"subpart", "extrapart", "connectedpart"}
        if not any("T" + tag in obj.name for tag in tags):
            continue
        match = b1k_pipeline.utils.parse_name(obj.parent.name)
        if not match:
            continue
        if match.group("instance_id") == "0":
            continue
        rt.delete(obj)


def delete_upper_links_from_nonzero_instances():
    for obj in rt.objects:
        match = b1k_pipeline.utils.parse_name(obj.name)
        if not match:
            continue
        if not match.group("bad") and match.group("instance_id") == "0":
            continue
        if match.group("joint_side") != "upper":
            continue
        rt.delete(obj)


def delete_meta_links_from_unnecessary_instances():
    for obj in rt.objects:
        match = b1k_pipeline.utils.parse_name(obj.name)
        if not match:
            continue
        if not match.group("bad") and match.group("instance_id") == "0":
            continue
        if not match.group("meta_type"):
            continue
        rt.delete(obj)


def fix_extracted_school_objects():
    ids_to_bad = b1k_pipeline.max.extract_school_objects.IDS_TO_MERGE
    for obj in rt.objects:
        m = b1k_pipeline.utils.parse_name(obj.name)
        if not m:
            continue
        if m.group("model_id") in ids_to_bad and not m.group("bad"):
            obj.name = "B-" + obj.name
            made_any_changes = True
    return made_any_changes


def fix_triangulation():
    for obj in tqdm.tqdm(list(rt.objects)):
        if rt.classOf(obj) != rt.Editable_Poly:
            continue

        # Check all faces are triangular
        all_triangular = all(
            len(rt.polyop.getFaceVerts(obj, i + 1)) == 3
            for i in range(rt.polyop.GetNumFaces(obj))
        )
        if not all_triangular:
            print("Need to triangulate", obj.name)

            # Triangulate
            ttp = rt.Turn_To_Poly()
            ttp.limitPolySize = True
            ttp.maxPolySize = 3
            rt.addmodifier(obj, ttp)
            rt.maxOps.collapseNodeTo(obj, 1, True)

        # Check that there are no dead elements
        if rt.polyop.GetHasDeadStructs(obj) != 0:
            # Remove dead structs
            # print("Need to collapse", obj.name)
            rt.polyop.CollapseDeadStructs(obj)


def convert_old_model_ids_to_new():
    objs_by_model = defaultdict(list)
    for obj in rt.objects:
        result = b1k_pipeline.utils.parse_name(obj.name)
        if result is None:
            print("{} does not match naming convention".format(obj.name))
            continue

        if re.fullmatch("[a-z]{6}", result.group("model_id")) is None:
            objs_by_model[
                (
                    result.group("category"),
                    result.group("model_id"),
                    result.group("bad"),
                )
            ].append(obj)

    for category, model_id, bad in objs_by_model:
        if bad:
            random_str = "todo" + random.choices(string.ascii_lowercase, k=2)
        else:
            random_str = "".join(
                random.choice(string.ascii_lowercase) for _ in range(6)
            )
        for obj in objs_by_model[(category, model_id, bad)]:
            old_str = "-".join([category, model_id])
            new_str = "-".join([category, random_str])
            obj.name = obj.name.replace(old_str, new_str)


def rename_hallway_to_corridor():
    existing_layer_names = {
        rt.LayerManager.getLayer(x).name for x in range(rt.LayerManager.count)
    }
    for layer_id in range(rt.LayerManager.count):
        layer = rt.LayerManager.getLayer(layer_id)
        if "hallway_" in layer.name:
            to_name = layer.name.replace("hallway_", "corridor_")
            assert (
                to_name not in existing_layer_names
            ), f"Layer {to_name} already exists"
            layer.setName(to_name)


def remove_soft_tags():
    for obj in rt.objects:
        if "-Tsoft" in obj.name:
            obj.name = obj.name.replace("-Tsoft", "")


def load_vray_renderer():
    preset_categories = rt.renderpresets.LoadCategories(RENDER_PRESET_FILENAME)
    assert rt.renderpresets.Load(0, RENDER_PRESET_FILENAME, preset_categories)


def convert_materials_to_vray(filename):
    # First, remove all baked materials and save
    # remove_nested_shell_materials(filename, keep_top_level=False)

    rt.orig_mtls = rt.Array()
    rt.new_mtls = rt.Array()
    rt.orig_texmaps = rt.Array()
    rt.new_texmaps = rt.Array()

    # Get the objects to process. This doesnt need to be in any particular order because we call
    # a per-node processing function rather htan the recursive one.
    objects = list(rt.objects)

    # Get each object's prior materials first for tracking purposes
    obj_materials_and_texmaps = {}
    for obj in objects:
        recursive_materials_and_texmaps = set()

        def _recursively_get_materials(mtl_or_texmap):
            if mtl_or_texmap is None:
                return

            recursive_materials_and_texmaps.add(mtl_or_texmap)

            # We can check for submtls for material instances
            if rt.superClassOf(mtl_or_texmap) == rt.Material:
                for i in range(rt.getNumSubMtls(mtl_or_texmap)):
                    sub_mtl = rt.getSubMtl(mtl_or_texmap, i + 1)
                    if sub_mtl is not None:
                        _recursively_get_materials(sub_mtl)

            # We can check for subtexmaps for texture maps and materials
            if (
                rt.superClassOf(mtl_or_texmap) == rt.textureMap
                or rt.superClassOf(mtl_or_texmap) == rt.Material
            ):
                for i in range(rt.getNumSubTexmaps(mtl_or_texmap)):
                    sub_texmap = rt.getSubTexmap(mtl_or_texmap, i + 1)
                    if sub_texmap is not None:
                        _recursively_get_materials(sub_texmap)

        _recursively_get_materials(obj.material)
        obj_materials_and_texmaps[obj] = recursive_materials_and_texmaps

    # Then do the actual conversion
    for obj in objects:
        converted_mtl = rt.createVRayMtl(obj.material)
        obj.material = converted_mtl

        # Check if any materials were converted, and if so, require rebake by removing the shell material
        converted_so_far = set(rt.orig_mtls) | set(rt.orig_texmaps)
        converted_this_object_uses = obj_materials_and_texmaps[obj] & converted_so_far
        if len(converted_this_object_uses) > 0:
            print(f"Object {obj.name} had materials converted, removing shell material")
            VRAY_LOG_FILENAME.write_text(
                VRAY_LOG_FILENAME.read_text()
                + f"{obj.name}: {','.join(x.name for x in converted_this_object_uses)}\n"
            )
            if rt.classOf(obj.material) == rt.Shell_Material:
                obj.material = obj.material.originalMaterial

    # Clear these arrays the same way as in the original script
    rt.orig_mtls = rt.Array()
    rt.new_mtls = rt.Array()
    rt.orig_texmaps = rt.Array()
    rt.new_texmaps = rt.Array()

    # Save again to get rid of all the unused materials
    rt.saveMaxFile(str(filename))

    # # Obtain all the materials used in the scene somehow
    # materials_directly_in_use = list(rt.scenematerials) + [obj.material for obj in rt.objects if rt.classOf(obj) == rt.Editable_Poly and obj.material is not None]
    # recursive_materials = set()
    # def _recursively_get_materials(mtl):
    #     recursive_materials.add(mtl)
    #     for i in range(rt.getNumSubMtls(mtl)):
    #         sub_mtl = rt.getSubMtl(mtl, i + 1)
    #         if sub_mtl is not None:
    #             _recursively_get_materials(sub_mtl)
    # for mtl in materials_directly_in_use:
    #     _recursively_get_materials(mtl)

    # # Check the found materials for any materials that are not VrayMtl or MultiMaterial
    # found_bad = False
    # for mat in rt.sceneMaterials:
    #     if rt.classOf(mat) != rt.MultiMaterial and "vray" not in str(rt.classOf(mat)).lower():
    #         print("Non-Vray material", mat.name)
    #         found_bad = True
    # #assert not found_bad, "Non-Vray material found"


def update_texture_paths():
    for obj in rt.objects:
        mtl = obj.material
        if not mtl or rt.classOf(mtl) != rt.Shell_Material:
            continue
        baked_mtl = mtl.bakedMaterial
        for map_idx in range(rt.getNumSubTexmaps(baked_mtl)):
            sub_texmap = rt.getSubTexmap(baked_mtl, map_idx + 1)
            if sub_texmap is not None:
                sub_texmap_slot_name = rt.getSubTexmapSlotName(baked_mtl, map_idx + 1)
                assert (
                    rt.classOf(sub_texmap) == rt.Bitmaptexture
                ), f"Object {obj.name} baked material map {sub_texmap_slot_name} has unexpected type {rt.classOf(sub_texmap)}"

                # Use os.path.abspath which normalizes + absolutifies the paths but does not resolve symlinks unlike pathlib (problem with dvc)
                map_path = pathlib.Path(os.path.abspath(sub_texmap.filename))
                assert map_path.exists(), f"Object {obj.name} baked material map {sub_texmap_slot_name} does not exist at {map_path}"
                bakery_path = b1k_pipeline.utils.PIPELINE_ROOT / "bakery"

                if bakery_path in map_path.parents:
                    # This is the correct bakery path, so ignore this object
                    continue

                # Otherwise, we need to update the path.
                correct_path = bakery_path / map_path.name

                # The below is for actually moving the files around. Here we don't need to do that.
                if correct_path.exists():
                    # If the path already exists, check that it's the same file.
                    with open(map_path, "rb") as f:
                        map_hash = hashlib.md5(f.read()).hexdigest()
                    with open(correct_path, "rb") as f:
                        correct_hash = hashlib.md5(f.read()).hexdigest()
                    if map_hash != correct_hash:
                        print("\nHash mismatch for", map_path, "and", correct_path)
                        print("Correct path hash is", correct_hash)
                        print("Map path hash is", map_hash)
                        raise ValueError(
                            f"Hash mismatch for {map_path} and {correct_path}. Please check the files."
                        )

                else:
                    shutil.copyfile(map_path, correct_path)

                # Then update the path in the bitmap texture
                sub_texmap.filename = str(correct_path)


def convert_baked_material_to_vray_and_add_ior():
    MAP_TRANSLATION = {
        "Diffuse map": "Base Color Map",
        "Bump map": "Bump Map",
        "Refl. gloss.": "Roughness Map",
        "Reflect map": "Reflectivity Map",
        "Refract map": "Transparency Map",
        "Metalness": "Metalness Map",
        "Fresnel IOR": "IOR Map",
    }

    for obj in rt.objects:
        parsed_name = b1k_pipeline.utils.parse_name(obj.name)
        if not parsed_name:
            continue
        if parsed_name.group("meta_type"):
            continue
        if parsed_name.group("instance_id") != "0":
            continue
        if parsed_name.group("bad"):
            continue
        if parsed_name.group("joint_side") == "upper":
            continue

        mtl = obj.material
        if not mtl or rt.classOf(mtl) != rt.Shell_Material:
            continue

        baked_mtl = mtl.bakedMaterial
        assert (
            rt.classOf(baked_mtl) == rt.Physical_Material
        ), f"Object {obj.name} baked material is not a Physical Material, but {rt.classOf(baked_mtl)}"

        maps = {}
        for map_idx in range(rt.getNumSubTexmaps(baked_mtl)):
            channel_name = rt.getSubTexmapSlotName(baked_mtl, map_idx + 1)
            sub_texmap = rt.getSubTexmap(baked_mtl, map_idx + 1)
            if sub_texmap is not None:
                if channel_name == "Transparency Color Map":
                    channel_name = "Transparency Map"
                maps[channel_name] = sub_texmap

        # If there is no IOR map, try to guess it from the name
        if "IOR Map" not in maps:
            ior_map_filename = (
                pathlib.Path(rt.maxFilePath)
                / "bakery"
                / f"{obj.name}_VRayMtlReflectIORBake.exr"
            )
            assert (
                ior_map_filename.exists()
            ), f"IOR map {ior_map_filename} for object {obj.name} does not exist"

            # Create a new bitmap and use it as the IOR map
            ior_map = rt.Bitmaptexture()
            ior_map.filename = str(ior_map_filename)
            ior_map.coords.mapChannel = 99

            maps["IOR Map"] = ior_map

        # Check that we have ALL of the maps we need
        missing_keys = set(MAP_TRANSLATION.values()) - set(maps.keys())
        assert not missing_keys, f"Missing maps {missing_keys} for object {obj.name}. Found only {set(maps.keys())}"

        # Start converting to the new material
        new_mtl = rt.VRayMtl()
        new_mtl.name = obj.name + "__baked"
        new_mtl.reflection_lockIOR = False
        converted_keys = set()
        for map_idx in range(rt.getNumSubTexmaps(new_mtl)):
            channel_name = rt.getSubTexmapSlotName(new_mtl, map_idx + 1)
            if channel_name not in MAP_TRANSLATION:
                continue
            assert (
                channel_name not in converted_keys
            ), f"Duplicate channel name {channel_name} for object {obj.name}"

            old_map = maps[MAP_TRANSLATION[channel_name]]
            rt.setSubTexmap(new_mtl, map_idx + 1, old_map)
            old_map.name = f"{obj.name}__baked__{channel_name}"
            converted_keys.add(channel_name)
        missing_converted_keys = set(MAP_TRANSLATION.keys()) - converted_keys
        assert (
            not missing_converted_keys
        ), f"Not all maps converted for object {obj.name}: {missing_converted_keys}"

        mtl.bakedMaterial = new_mtl


def processFile(filename: pathlib.Path):
    # Load file, fixing the units
    print(f"\n\nProcessing {filename}")
    assert rt.loadMaxFile(str(filename), useFileUnits=False, quiet=True)
    assert rt.units.systemScale == 1, "System scale not set to 1mm."
    assert rt.units.systemType == rt.Name("millimeters"), "System scale not set to 1mm."

    # Switch to Vray
    # load_vray_renderer()

    # Apply deletions
    apply_deletions(filename)

    # Remove shell materials that might show up under multi materials
    remove_nested_shell_materials(filename)

    # Fix any bad materials
    convert_materials_to_vray(filename)

    # Remove root_level meta links
    remove_root_level_meta_links(filename)

    # Apply renames
    apply_renames(filename)

    # Fix layering (room assignments and object files)
    # fix_layers(filename)

    # Remove soft tags
    # remove_soft_tags()

    # Fix layers called hallway
    # rename_hallway_to_corridor()

    # Fix any old names
    # convert_old_model_ids_to_new()

    # Get all editable polies
    # fix_triangulation()

    # Convert extracted school objects to BAD on these scenes
    # fix_extracted_school_objects()

    # Delete meta links from non-zero or bad instances
    # delete_meta_links_from_unnecessary_instances()

    # Delete upper links from non-zero instances
    # delete_upper_links_from_nonzero_instances()

    # Delete parts from non-zero instances
    # delete_parts_from_nonzero_instances()

    # Update UV unwrap to use correct attribute
    # convert_legacy_uv_unwrap_attribute_to_new()

    # Run the bad object replacement system for legacy scenes
    # For this particular task we are only doing this to the legacy-containing scenes
    # run_bad_object_replacement_for_legacy_scenes(filename)

    # Generate all missing collision meshes
    # b1k_pipeline.max.convex_decomposition.generate_all_missing_collision_meshes()

    # Reduce collision mesh vertex counts
    # b1k_pipeline.max.collision_vertex_reduction.process_all_convex_meshes()

    # Match links
    # b1k_pipeline.max.match_links.process_all_objects()

    # Merge preexisting fillable meshes
    # merge_preexisting_fillable_meshes()

    # Import fillable meshes
    # b1k_pipeline.max.import_fillable_meshes.process_current_file()

    # Remove lights from bad objects and nonzero instances
    # remove_unnecessary_lights()

    # Apply the same material to all instances
    # match_instance_materials()

    # Remove materials from meta links
    # for obj in rt.objects:
    #     pn = b1k_pipeline.utils.parse_name(obj.name)
    #     if not pn:
    #         continue
    #     if pn.group("meta_type"):
    #         obj.material = None

    # Temporary hack for rebaking just the reflection channel
    # baked_mtls_by_object = {}
    # for obj in rt.objects:
    #     if obj.material and rt.classOf(obj.material) == rt.Shell_Material:
    #         # baked_mtls_by_object[obj] = obj.material.bakedMaterial
    #         obj.material = obj.material.originalMaterial
    #         assert rt.classOf(obj.material) != rt.Shell_Material, f"{obj} material should not be shell material before baking"
    b1k_pipeline.max.prebake_textures.process_open_file()
    # for obj, old_baked_mtl in baked_mtls_by_object.items():
    #     # Make sure it got baked again
    #     if not rt.classOf(obj.material) == rt.Shell_Material:
    #         print(f"{obj} material is not shell material after baking - meaning this object was not baked.")
    #         continue

    #     new_baked_mtl = obj.material.bakedMaterial

    #     # Copy the new material's reflection channel to the old one's slot too
    #     old_baked_mtl.reflectivity_map = new_baked_mtl.reflectivity_map

    #     # Switch back to the old shell material
    #     obj.material.bakedMaterial = old_baked_mtl

    # Update baked texture paths if the baked textures are not in the same folder as the max file
    # convert_baked_material_to_vray_and_add_ior()

    # Apply the orientation and scale changes
    # b1k_pipeline.max.apply_qa_fixes.apply_qa_fixes_in_open_file()

    # Update visibility settings
    update_visibilities()

    # Save again.
    rt.saveMaxFile(str(filename))

    with open(filename.parent / PASS_FILENAME, "w") as f:
        pass


def fix_common_issues_in_all_files():
    candidates = [
        x
        for x in pathlib.Path(r"D:\BEHAVIOR-1K\asset_pipeline").glob(
            "cad/*/*/processed.max"
        )
    ]

    for i, f in enumerate(tqdm.tqdm(candidates)):
        if (f.parent / PASS_FILENAME).exists():
            continue
        processFile(f)


if __name__ == "__main__":
    fix_common_issues_in_all_files()
