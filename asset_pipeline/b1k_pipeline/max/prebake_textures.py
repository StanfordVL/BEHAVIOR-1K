import pathlib
import sys
import traceback
import hashlib

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import os
import traceback
import numpy as np
from pymxs import runtime as rt
import time
import re
import b1k_pipeline.utils

_prebake_data_str = """attributes "prebakeData" attribID:#(0x6c7ac544, 0x36f63a5f) version:1
(
    parameters main rollout:params
    (
        hashDigest type:#string ui:hashDigest default:""
    )

    rollout params "Prebake Data"
    (
        edittext hashDigest "Mesh hash digest from last unwrap:" fieldWidth:300 labelOnTop:true
    )
)"""
PrebakeDataAttr = rt.execute(_prebake_data_str)

btt = rt.BakeToTexture

BATCH_MODE = False
USE_UNWRELLA = True
IMG_SIZE = 1024
HQ_IMG_SIZE = 4096
HQ_IMG_CATEGORIES = {"floors", "lawn", "driveway", "walls"}
NEW_UV_CHANNEL = 99

# PBRMetalRough
# CHANNEL_MAPPING = {
#     "VRayRawDiffuseFilterMap": "Base Color Map",
#     "VRayBumpNormalsMap": "Normal Map",
#     # "VRayMtlReflectGlossinessBake": "Roughness Map",
# }

# PhysicalMaterial
CHANNEL_MAPPING = {
    "VRayRawDiffuseFilterMap": "Diffuse map",  # or VRayDiffuseFilter
    "VRayNormalsMap": "Bump map",  # or VRayBumpNormals
    "VRayMtlReflectGlossinessBake": "Refl. gloss.",  # iGibson/Omniverse renderer expects we flip the glossiness map
    "VRayRawReflectionFilterMap": "Reflect map",
    "VRayRawRefractionFilterMap": "Refract map",
    "VRayMetalnessMap": "Metalness",  # requires V-ray 5, update 2.3
    "VRayMtlReflectIORBake": "Fresnel IOR",
}

CHANNEL_DATA_FORMAT_OVERRIDES = {
    "VRayMtlReflectIORBake": "exr",
}

# CHANNEL_MAPPING = {
#     "Color": "Base Color Map",
#     "Normal": "Bump Map",
# }

RENDER_PRESET_FILENAME = str(
    (
        b1k_pipeline.utils.PIPELINE_ROOT / "render_presets" / "no_sampler_no_gi.rps"
    ).absolute()
)

allow_list = []
black_list = []


def unlight_mats_recursively(mat):
    if mat:
        if hasattr(mat, "selfIllumination_multiplier"):
            mat.selfIllumination_multiplier = 0.0
        for i in range(mat.numsubs):
            submat = mat[i]
            unlight_mats_recursively(submat)


def unlight_all_mats():
    materials = {x.material for x in rt.objects}
    for mat in materials:
        unlight_mats_recursively(mat)


def hash_object(obj, verts=None, faces=None):
    # This optionally takes verts and faces so that the precached ones from sanitycheck can be used.

    if verts is None:
        # Create a numpy array containing the vertex positions and the faces
        # Notice by getting this from the baseObject we get them in the object's local space
        # which is NOT the pivot space.
        verts = np.array(
            rt.polyop.getVerts(
                obj.baseObject, rt.execute("#{1..%d}" % rt.polyop.getNumVerts(obj))
            )
        )

    if faces is None:
        faces = (
            np.array(
                rt.polyop.getFacesVerts(
                    obj, rt.execute("#{1..%d}" % rt.polyop.GetNumFaces(obj))
                )
            )
            - 1
        )

    # Notice that we are adding 0. here. That is to avoid an absolutely INSANE bug where
    # 3ds Max returns -0. sometimes and 0. other times, causing the hash to be different
    # for very few objects because very few bytes differ. Adding zero makes all zeros positive.
    verts = verts + 0.0
    assert faces.shape[1] == 3, f"{obj.name} has non-triangular faces"

    # Hash the vertices and faces
    return hashlib.sha256(verts.tobytes() + faces.tobytes()).hexdigest()


def get_recorded_uv_unwrapping_hash(obj):
    prebake_data_attr = rt.custAttributes.get(obj, PrebakeDataAttr)
    if not prebake_data_attr:
        return None

    hash_digest = prebake_data_attr.hashDigest
    if not hash_digest:
        return None

    return hash_digest


def set_recorded_uv_unwrapping_hash(obj, hash_digest):
    if not rt.custAttributes.get(obj, PrebakeDataAttr):
        rt.custAttributes.add(obj, PrebakeDataAttr)

    prebake_data_attr = rt.custAttributes.get(obj, PrebakeDataAttr)
    prebake_data_attr.hashDigest = hash_digest


class TextureBaker:
    def __init__(self, bakery):
        self.MAP_NAME_TO_IDS = self.get_map_name_to_ids()

        self.unwrap_times = {}

        self.bakery = bakery
        os.makedirs(bakery, exist_ok=True)

    @staticmethod
    def get_map_name_to_ids():
        map_name_to_ids = {}
        for item in btt.getCompatibleMapTypes():
            map_name = item.split(" : ")[0]
            class_id_str = item[item.rindex("(") + 1 : item.rindex(")")]
            id_tuples = tuple([int(item) for item in class_id_str.split(", ")])
            map_name_to_ids[map_name] = id_tuples
        return map_name_to_ids

    def get_process_objs(self):
        objs = []
        for obj in rt.objects if not rt.selection else rt.selection:
            if rt.classOf(obj) != rt.Editable_Poly:
                continue
            if allow_list and all(
                re.fullmatch(p, obj.name) is None for p in allow_list
            ):
                continue
            if black_list and any(
                re.fullmatch(p, obj.name) is not None for p in black_list
            ):
                continue
            parsed_name = b1k_pipeline.utils.parse_name(obj.name)
            if parsed_name.group("meta_type"):
                continue
            if parsed_name.group("instance_id") != "0":
                continue
            if parsed_name.group("bad"):
                continue
            if parsed_name.group("joint_side") == "upper":
                continue

            objs.append(obj)

        return objs

    def uv_unwrapping_unwrella(self, obj):
        u = rt.Unwrella()
        rt.addmodifier(obj, u)
        u.stretch = 0.15
        u.rescale = True
        u.prerotate = True
        u.packmode = 1  # Efficient
        u.map_channel = NEW_UV_CHANNEL
        u.keep_seams = False
        u.unwrap_mode = 0

        u.padding = 2
        u.width = 1024
        u.height = 1024
        u.preview = False

        assert u.unwrap(), f"Unwrapping error w/ {obj.name}: {u.error}"

    def uv_unwrapping_native(self, obj):
        rt.select(obj)

        # Select all faces in preparation for uv unwrap
        rt.polyop.setFaceSelection(obj, rt.name("all"))

        modifier = rt.unwrap_uvw()
        rt.addmodifier(obj, modifier)

        modifier.unwrap2.setApplyToWholeObject(True)

        # In case the object is already uv unwrapped, we will need to assign to a new UV channel
        modifier.setMapChannel(NEW_UV_CHANNEL)
        modifier.unwrap2.flattenMapNoParams()

    def uv_unwrapping(self, obj):
        # Check if the object is already unwrapped with the same geometry
        current_hash = hash_object(obj)
        recorded_hash = get_recorded_uv_unwrapping_hash(obj)
        if rt.polyop.getMapSupport(obj, 99) and recorded_hash == current_hash:
            print(f"Skipping unwrapping for {obj.name} as it has not changed.")
            return

        start_time = time.time()

        print("uv_unwrapping", obj.name)

        rt.polyop.setMapSupport(obj, 99, False)
        assert not rt.polyop.getMapSupport(
            obj, 99
        ), f"Failed to clear UV channel 99 for {obj.name} prior to unwrapping"
        if USE_UNWRELLA:
            self.uv_unwrapping_unwrella(obj)
            if not rt.polyop.getMapSupport(obj, 99):
                print(
                    f"Unwrella failed for {obj.name}, falling back to native unwrapping."
                )
                self.uv_unwrapping_native(obj)
        else:
            self.uv_unwrapping_native(obj)
        assert rt.polyop.getMapSupport(
            obj, 99
        ), f"Could not unwrap UVs for object {obj.name}"

        # Flatten the modifier stack
        rt.maxOps.collapseNodeTo(obj, 1, True)

        # Record metadata about the hash at the time of the unwrapping.
        post_unwrap_hash = hash_object(obj)
        assert (
            post_unwrap_hash == current_hash
        ), f"Hash mismatch before and after unwrapping: {post_unwrap_hash} != {current_hash}"
        set_recorded_uv_unwrapping_hash(obj, current_hash)

        # If it's assigned to a baked material, switch back to the unbaked material.
        if rt.classOf(obj.material) == rt.Shell_Material:
            obj.material = obj.material.originalMaterial

        rt.update(obj)

        end_time = time.time()
        self.unwrap_times[obj.name] = end_time - start_time

    def prepare_texture_baking(self, obj):
        # If the object is already connected to a baked material, skip the baking process
        if (
            rt.classOf(obj.material) == rt.Shell_Material
            and obj.material.renderMtlIndex == 1
        ):
            print(f"Skipping baking for {obj.name} as it is already baked.")
            return

        # Also skip objects whose material is not some kind of a Vray material
        # if "vray" not in str(rt.classOf(obj.material)).lower():
        #     print(f"Skipping baking for {obj.name} as it is not a Vray material.")
        #     return

        # Get the existing baseobject children that use the same material as this one
        siblings = []
        for candidate in rt.objects:
            if (
                candidate.baseObject == obj.baseObject
                # and candidate.material == obj.material  # TODO: Is this too aggressive?
            ):
                siblings.append(candidate)

        # Disconnect the existing baked material
        if rt.classOf(obj.material) == rt.Shell_Material:
            obj.material = obj.material.originalMaterial

        print("prepare_texture_baking", obj.name)

        for i, map_name in enumerate(CHANNEL_MAPPING.keys()):
            texture_map = btt.addMapByClassId(obj, self.MAP_NAME_TO_IDS[map_name])

            # Only the first channel requires creating the new PhysicalMaterial
            if i == 0:
                btt.setOutputTo(obj, "CreateNewMaterial", material=rt.VrayMtl())
                # btt.setOutputTo(obj, "CreateNewMaterial", material=rt.PBRMetalRough())

            # Make sure the new UV channel is selected
            texture_map.uvChannel = NEW_UV_CHANNEL
            img_size = IMG_SIZE
            if (
                b1k_pipeline.utils.parse_name(obj.name).group("category")
                in HQ_IMG_CATEGORIES
            ):
                img_size = HQ_IMG_SIZE
            texture_map.imageWidth = img_size
            texture_map.imageHeight = img_size
            texture_map.edgePadding = 4
            texture_map.fileType = CHANNEL_DATA_FORMAT_OVERRIDES.get(map_name, "png")

            # Set the apply color mapping option
            if (
                texture_map.getOptionsCount() > 0
                and texture_map.getOptionName(1) == "Apply color mapping"
            ):
                texture_map.setOptionValue(1, False)

            # Mapping from the original channel (of VRay, Corona, etc) to the new channel of PhysicalMaterial
            texture_map.setTargetMapSlot(CHANNEL_MAPPING[map_name])

        return siblings

    def postprocess_texture_baking(self, obj, siblings):
        # Set the object to render the baked material
        obj.material.renderMtlIndex = 1

        new_mtl = obj.material.bakedMaterial
        new_mtl.name = obj.name + "__baked"
        new_mtl.reflection_lockIOR = False
        for map_idx in range(rt.getNumSubTexmaps(new_mtl)):
            channel_name = rt.getSubTexmapSlotName(new_mtl, map_idx + 1)
            texmap = rt.getSubTexmap(new_mtl, map_idx + 1)
            if texmap:
                texmap.name = f"{obj.name}__baked__{channel_name}"

        # Update everything that has the same baseobject to use the same material
        for sibling in siblings:
            sibling.material = obj.material
            rt.update(sibling)

        rt.update(obj)

    def texture_baking(self):
        # vray = rt.renderers.current
        # vray.camera_autoExposure = False
        # vray.options_lights = False
        # vray.options_hiddenLights = False
        rt.sceneexposurecontrol.exposurecontrol = None

        # Bake textures
        btt.outputPath = self.bakery
        btt.autoCloseProgressDialog = True
        btt.showFrameBuffer = False
        btt.alwaysOverwriteExistingFiles = True

        # Do the actual baking
        start_time = time.time()
        print("start baking")
        assert btt.bake(), "baking failed"
        print("finish baking")
        end_time = time.time()
        print(f"baking took {end_time - start_time} seconds")

        # Clear the baking list again.
        btt.deleteAllMaps()

    def run(self):
        # assert rt.classOf(rt.renderers.current) == rt.V_Ray_5__update_2_3, f"Renderer should be set to V-Ray 5.2.3 CPU instead of {rt.classOf(rt.renderers.current)}"
        assert rt.execute("max modify mode")

        # Remove lights from all materials
        # unlight_all_mats()

        # Remove the texture baking list
        btt.deleteAllMaps()

        objs = self.get_process_objs()
        postprocessing = []  # (obj, siblings)
        for i, obj in enumerate(objs):
            print(f"{(i + 1)} / {len(objs)}: {obj.name}")

            rt.select([obj])
            rt.IsolateSelection.EnterIsolateSelectionMode()

            obj.isHidden = False
            for child in obj.children:
                child.isHidden = False

            # TODO: Can we bake all at once?
            self.uv_unwrapping(obj)
            siblings = self.prepare_texture_baking(obj)
            if siblings is None:  # not baking this object!
                continue

            if not BATCH_MODE:
                # If we're not in batch mode, bake now.
                self.texture_baking()
                self.postprocess_texture_baking(obj, siblings)
            else:
                # Otherwise, queue for postprocessing for after baking.
                postprocessing.append((obj, siblings))

        if BATCH_MODE:
            # Bake
            self.texture_baking()

            # Postprocessing
            for obj, siblings in postprocessing:
                self.postprocess_texture_baking(obj, siblings)


def process_open_file():
    preset_categories = rt.renderpresets.LoadCategories(RENDER_PRESET_FILENAME)
    assert rt.renderpresets.Load(0, RENDER_PRESET_FILENAME, preset_categories)

    rt.setvraysilentmode(True)

    # TODO: Assert that the bakery path is unprotected
    bakery_path = pathlib.Path(rt.maxFilePath) / "bakery"
    bakery_path.mkdir(exist_ok=True)
    exp = TextureBaker(str(bakery_path))
    exp.run()


if __name__ == "__main__":
    process_open_file()
